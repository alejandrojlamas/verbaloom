"""
Version checker

Compares the locally installed version (src/__version__.py) with the latest
GitHub release of the repository so the UI can surface an update banner.
Result is cached in memory to avoid hammering the GitHub API.
"""
import logging
import os
import re
import threading
import time
from typing import Optional, Tuple

import requests

from src import __version__

logger = logging.getLogger(__name__)

GITHUB_REPO_OWNER = os.getenv("TBL_UPDATE_REPO_OWNER", "alejandrojlamas")
GITHUB_REPO_NAME = os.getenv("TBL_UPDATE_REPO_NAME", "Traductor-y-estilizador-de-libros")
GITHUB_RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/latest"
GITHUB_TAGS_API = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/tags"

# Cache TTL: 1 hour. GitHub rate-limits unauthenticated calls at 60/hour/IP.
CACHE_TTL_SECONDS = 3600

_cache_lock = threading.Lock()
_cache: dict = {"timestamp": 0.0, "data": None}


def _parse_version(v: str) -> Tuple[int, ...]:
    """Parse a semver-like string into a tuple of ints for comparison.

    Strips leading 'v' and any non-numeric pre-release suffix so that
    '1.3.6', 'v1.3.6', '1.3.6-rc1' all reduce to (1, 3, 6).
    Returns (0,) when no numeric component is found.
    """
    if not v:
        return (0,)
    cleaned = v.strip().lstrip("vV")
    parts = re.findall(r"\d+", cleaned)
    if not parts:
        return (0,)
    return tuple(int(p) for p in parts)


def _is_newer(remote: str, local: str) -> bool:
    """Return True if `remote` is strictly newer than `local`."""
    return _parse_version(remote) > _parse_version(local)


def get_current_version() -> str:
    """Return the locally installed version string."""
    return __version__


def _fetch_latest_from_github(timeout: float = 5.0) -> Optional[dict]:
    """Fetch the latest release metadata from GitHub.

    Falls back to the tags endpoint when no published release exists.
    Returns None on network/parse failure (caller decides UX).
    """
    headers = {"Accept": "application/vnd.github+json"}
    try:
        resp = requests.get(GITHUB_RELEASES_API, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "tag_name": data.get("tag_name", ""),
                "name": data.get("name", ""),
                "html_url": data.get("html_url", ""),
                "body": data.get("body", "") or "",
                "published_at": data.get("published_at", ""),
                "source": "release",
            }
        if resp.status_code == 404:
            # No releases yet, try tags as fallback
            tags_resp = requests.get(GITHUB_TAGS_API, headers=headers, timeout=timeout)
            if tags_resp.status_code == 200:
                tags = tags_resp.json()
                if tags:
                    tag = tags[0]
                    return {
                        "tag_name": tag.get("name", ""),
                        "name": tag.get("name", ""),
                        "html_url": f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/tag/{tag.get('name', '')}",
                        "body": "",
                        "published_at": "",
                        "source": "tag",
                    }
                return {
                    "tag_name": "",
                    "name": "",
                    "html_url": f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases",
                    "body": "",
                    "published_at": "",
                    "source": "none",
                }
        logger.debug(f"Version check: GitHub returned HTTP {resp.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        logger.debug(f"Version check failed: {e}")
        return None


def check_for_update(force: bool = False, timeout: float = 5.0) -> dict:
    """Return current/latest version info plus an update_available flag.

    Cached for CACHE_TTL_SECONDS unless `force=True`. On network failure the
    last known cached entry is returned (so an offline UI still shows status);
    if no cache exists yet, returns `update_available=False` with `error`.
    """
    now = time.time()
    with _cache_lock:
        if not force and _cache["data"] and (now - _cache["timestamp"]) < CACHE_TTL_SECONDS:
            return _cache["data"]

    current = get_current_version()
    remote_info = _fetch_latest_from_github(timeout=timeout)

    if not remote_info:
        with _cache_lock:
            if _cache["data"]:
                stale = dict(_cache["data"])
                stale["from_cache"] = True
                return stale
        return {
            "current": current,
            "latest": None,
            "update_available": False,
            "release_url": f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases",
            "release_notes": "",
            "error": "Could not reach GitHub. Check your internet connection.",
        }

    latest_tag = remote_info["tag_name"] or remote_info["name"]
    update_available = bool(latest_tag) and _is_newer(latest_tag, current)

    result = {
        "current": current,
        "latest": latest_tag.lstrip("vV"),
        "update_available": update_available,
        "release_url": remote_info["html_url"] or f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases",
        "release_notes": remote_info["body"][:2000],
        "published_at": remote_info["published_at"],
        "source": remote_info["source"],
    }

    with _cache_lock:
        _cache["timestamp"] = now
        _cache["data"] = result

    return result


def invalidate_cache() -> None:
    """Force the next check_for_update() call to hit GitHub."""
    with _cache_lock:
        _cache["timestamp"] = 0.0
        _cache["data"] = None
