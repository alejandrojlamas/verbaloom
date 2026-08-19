"""
App self-updater

Runs `git pull` + `pip install` in a background thread and (optionally) triggers
a process restart. The Flask request handler reads progress through the shared
UpdateState singleton via /api/version/update/status.

Restart strategy: exit with code 42. The launch scripts (start.bat / start.sh)
loop on this exit code and relaunch the app. When the app was started directly
with `python translation_api.py`, the user must restart it manually; in that
case we still write the new code/deps to disk before exiting.
"""
import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

RESTART_EXIT_CODE = 42

# Inactivity grace period before the process exits, so the HTTP response from
# /api/version/update can reach the browser and the WebSocket can flush.
_RESTART_GRACE_SECONDS = 1.5


@dataclass
class UpdateState:
    """Thread-safe state for the in-flight update job."""
    state: str = "idle"  # idle | running | completed | failed
    step: str = ""
    output: List[str] = field(default_factory=list)
    error: Optional[str] = None
    requires_restart: bool = False
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "step": self.step,
                "output": list(self.output),
                "error": self.error,
                "requires_restart": self.requires_restart,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    def reset(self):
        with self._lock:
            self.state = "idle"
            self.step = ""
            self.output = []
            self.error = None
            self.requires_restart = False
            self.started_at = None
            self.finished_at = None

    def start(self):
        with self._lock:
            self.reset()
            self.state = "running"
            self.started_at = time.time()

    def set_step(self, step: str):
        with self._lock:
            self.step = step
            self.output.append(f"[{time.strftime('%H:%M:%S')}] {step}")

    def append(self, line: str):
        with self._lock:
            self.output.append(line)

    def fail(self, message: str):
        with self._lock:
            self.state = "failed"
            self.error = message
            self.finished_at = time.time()

    def complete(self, requires_restart: bool):
        with self._lock:
            self.state = "completed"
            self.requires_restart = requires_restart
            self.finished_at = time.time()


_state = UpdateState()
_thread_lock = threading.Lock()
_active_thread: Optional[threading.Thread] = None


def get_state() -> UpdateState:
    return _state


def is_running() -> bool:
    return _state.state == "running"


def is_git_repo(repo_root: Path) -> bool:
    return (repo_root / ".git").exists()


def get_install_kind(repo_root: Path) -> str:
    """Detect how the app is being run, so the UI can route updates correctly.

    Returns one of:
        'frozen-windows' / 'frozen-macos' / 'frozen-linux' - PyInstaller bundle:
            auto-update via git pull is impossible; the UI should point users
            to the release page so they can download a new binary.
        'git'    - cloned repo: in-place git pull + restart works.
        'source' - source tree without .git (e.g. extracted zip): same UX as
            'frozen' (point to the release page).
    """
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            return "frozen-windows"
        if sys.platform == "darwin":
            return "frozen-macos"
        return "frozen-linux"
    if is_git_repo(repo_root):
        return "git"
    return "source"


def _hash_requirements(req_path: Path) -> Optional[str]:
    if not req_path.exists():
        return None
    return hashlib.md5(req_path.read_bytes()).hexdigest()


def _run(cmd: List[str], cwd: Path, env: Optional[dict] = None, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a subprocess with merged stdout/stderr, capturing output as text."""
    logger.info(f"Updater running: {' '.join(cmd)} (cwd={cwd})")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _do_update(repo_root: Path, install_deps: bool, restart: bool) -> None:
    """Background worker: git pull -> pip install -> exit-for-restart."""
    state = _state
    try:
        if not is_git_repo(repo_root):
            state.fail("Not a git repository. Auto-update needs the app installed via git clone.")
            return

        req_path = repo_root / "requirements.txt"
        old_hash = _hash_requirements(req_path)

        state.set_step("Fetching updates from GitHub")
        fetch = _run(["git", "fetch", "--all", "--prune"], cwd=repo_root, timeout=120)
        if fetch.stdout:
            state.append(fetch.stdout.strip())
        if fetch.returncode != 0:
            state.fail(f"git fetch failed (exit {fetch.returncode}). See output for details.")
            return

        state.set_step("Pulling latest changes")
        pull = _run(["git", "pull", "--ff-only"], cwd=repo_root, timeout=120)
        if pull.stdout:
            state.append(pull.stdout.strip())
        if pull.returncode != 0:
            state.fail(
                "git pull failed. You may have local changes; resolve them manually "
                "(e.g. `git stash`) and retry."
            )
            return

        new_hash = _hash_requirements(req_path)
        deps_changed = (old_hash != new_hash)

        if install_deps and deps_changed:
            state.set_step("Updating Python dependencies (this may take a minute)")
            pip = _run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_path), "--upgrade"],
                cwd=repo_root,
                timeout=900,
            )
            if pip.stdout:
                tail = "\n".join(pip.stdout.strip().splitlines()[-40:])
                state.append(tail)
            if pip.returncode != 0:
                state.fail(f"pip install failed (exit {pip.returncode}). Code is updated but dependencies are stale.")
                return
        elif deps_changed:
            state.append("requirements.txt changed but dependency install was skipped.")

        state.set_step("Update applied successfully")
        state.complete(requires_restart=restart)

        if restart:
            _schedule_restart()

    except subprocess.TimeoutExpired as e:
        state.fail(f"Step timed out: {e.cmd}")
    except FileNotFoundError as e:
        state.fail(f"Required command not found: {e}. Is git installed and on PATH?")
    except Exception as e:
        logger.exception("Updater crashed")
        state.fail(f"Unexpected error: {e}")


def _schedule_restart() -> None:
    """Exit the process so the wrapper script (start.bat/start.sh) relaunches it.

    A short delay lets the HTTP response leave the wire before the process dies.
    Exit code RESTART_EXIT_CODE is what the wrappers loop on; under direct
    `python translation_api.py` the process simply exits and the user restarts
    it manually.
    """
    def _exit_later():
        time.sleep(_RESTART_GRACE_SECONDS)
        logger.info(f"Updater triggering restart (exit code {RESTART_EXIT_CODE})")
        os._exit(RESTART_EXIT_CODE)

    threading.Thread(target=_exit_later, daemon=True, name="updater-restart").start()


def trigger_update(repo_root: Path, install_deps: bool = True, restart: bool = True) -> bool:
    """Kick off the update in a background thread. Returns False if already running."""
    global _active_thread
    with _thread_lock:
        if _state.state == "running":
            return False
        _state.start()
        _active_thread = threading.Thread(
            target=_do_update,
            args=(repo_root, install_deps, restart),
            daemon=True,
            name="app-updater",
        )
        _active_thread.start()
    return True
