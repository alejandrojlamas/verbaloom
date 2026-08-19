"""
Version check & self-update routes.

Endpoints:
    GET  /api/version/check         - current vs latest GitHub release
    GET  /api/version/update/status - in-flight update progress
    POST /api/version/update        - kick off git pull + pip install + restart
"""
import logging
from pathlib import Path
from flask import Blueprint, jsonify, request

from src.utils import version_checker
from src.utils import app_updater

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Project root. CWD is set by translation_api.py / launcher.py."""
    return Path.cwd()


def _list_active_jobs(state_manager) -> list:
    """Return the subset of jobs that would be killed by a restart."""
    active = []
    try:
        for tid, tdata in state_manager.get_all_translations().items():
            status = tdata.get("status")
            if status in ("running", "queued"):
                active.append({
                    "id": tid,
                    "status": status,
                    "output_filename": tdata.get("config", {}).get("output_filename", "unknown"),
                })
    except Exception as e:
        logger.warning(f"Could not enumerate active jobs: {e}")
    return active


def create_version_blueprint(state_manager):
    """Create the version blueprint.

    state_manager is needed so the update endpoint can refuse to restart
    while translations are still running.
    """
    bp = Blueprint("version", __name__)

    @bp.route("/api/version/check", methods=["GET"])
    def check_version():
        """Return current and latest version info. `?force=1` bypasses cache."""
        force = request.args.get("force", "0") in ("1", "true", "yes")
        try:
            result = version_checker.check_for_update(force=force)
            repo_root = _repo_root()
            result["git_repo"] = app_updater.is_git_repo(repo_root)
            result["install_kind"] = app_updater.get_install_kind(repo_root)
            return jsonify(result)
        except Exception as e:
            logger.exception("Version check failed")
            repo_root = _repo_root()
            return jsonify({
                "current": version_checker.get_current_version(),
                "latest": None,
                "update_available": False,
                "error": str(e),
                "git_repo": app_updater.is_git_repo(repo_root),
                "install_kind": app_updater.get_install_kind(repo_root),
            }), 500

    @bp.route("/api/version/update/status", methods=["GET"])
    def update_status():
        """Return progress of the in-flight (or last) update job."""
        return jsonify(app_updater.get_state().snapshot())

    @bp.route("/api/version/update", methods=["POST"])
    def start_update():
        """Trigger git pull + (optional) pip install + restart.

        Refuses when:
            - an update is already running
            - any translation is running or queued
            - the install is not a git checkout
        """
        repo_root = _repo_root()

        if not app_updater.is_git_repo(repo_root):
            return jsonify({
                "error": "not_a_git_repo",
                "message": "Auto-update requires a git checkout. Reinstall via git clone or update manually.",
            }), 400

        if app_updater.is_running():
            return jsonify({
                "error": "already_running",
                "message": "An update is already in progress.",
                "status": app_updater.get_state().snapshot(),
            }), 409

        active = _list_active_jobs(state_manager)
        if active:
            return jsonify({
                "error": "active_translations",
                "message": "Cannot update while translations are running. Wait or interrupt them first.",
                "active_translations": active,
            }), 409

        data = request.get_json(silent=True) or {}
        install_deps = bool(data.get("install_deps", True))
        restart = bool(data.get("restart", True))

        started = app_updater.trigger_update(repo_root, install_deps=install_deps, restart=restart)
        if not started:
            return jsonify({
                "error": "already_running",
                "message": "An update is already in progress.",
            }), 409

        return jsonify({
            "success": True,
            "message": "Update started.",
            "status": app_updater.get_state().snapshot(),
        }), 202

    return bp
