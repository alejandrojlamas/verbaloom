"""Mobile and Tailscale diagnostics registered on the main config blueprint."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

from flask import Blueprint, jsonify, make_response, request

from src.utils.branding import REPOSITORY_URL, ROUTE_PREFIX, env_value


TAILSCALE_IP_RE = re.compile(r"^100\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
MOBILE_ACCESS_LOG_LIMIT = 60
MOBILE_ACCESS_MAX_EVENTS = 2000
MOBILE_ACCESS_COMPACT_BYTES = 1024 * 1024


def configured_magicdns_url() -> str:
    """Return an optional user-provided MagicDNS URL without a device default."""
    return str(env_value("MAGICDNS_URL", "")).strip()


def local_tailnet_ip() -> str:
    """Return the local Tailscale IPv4 address for phone fallback URLs."""
    candidates: list[str] = []
    try:
        candidates.extend(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        candidates.extend(
            addr[4][0]
            for addr in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        )
    except OSError:
        pass
    for value in candidates:
        if TAILSCALE_IP_RE.match(value):
            return value
    configured = str(env_value("TAILSCALE_IP", "")).strip()
    return configured if TAILSCALE_IP_RE.match(configured) else "127.0.0.1"


def nipio_tailnet_host(tailnet_ip: str) -> str:
    """Return a DNS fallback hostname that still resolves inside Tailscale."""
    return f"{tailnet_ip}.nip.io"


def external_android_event(event: dict, local_tailnet_ip: str) -> bool:
    """Return whether an event proves that an external Android reached Flask."""
    if not event.get("is_android"):
        return False
    forwarded_for = str(event.get("forwarded_for") or "").split(",", 1)[0].strip()
    remote_addr = str(event.get("remote_addr") or "").strip()
    local_addresses = {"", "127.0.0.1", "::1", local_tailnet_ip}
    return any(
        address not in local_addresses
        for address in (forwarded_for, remote_addr)
        if address
    )


def mobile_request_context() -> dict:
    """Return safe request details useful for phone-side diagnostics."""
    user_agent = request.headers.get("User-Agent", "")
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    remote_addr = request.remote_addr or ""
    lower_ua = user_agent.lower()
    is_android = "android" in lower_ua
    is_mobile = is_android or any(
        token in lower_ua
        for token in ("mobile", "iphone", "ipad", "ipod", "samsungbrowser")
    )
    return {
        "user_agent": user_agent,
        "remote_addr": remote_addr,
        "forwarded_for": forwarded_for,
        "request_host": request.host,
        "request_path": request.path,
        "request_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "is_android": is_android,
        "is_mobile": is_mobile,
    }


class MobileAccessEventStore:
    """Persist the bounded, non-sensitive request metadata used by diagnostics."""

    def __init__(
        self,
        config_path_provider: Callable[[], str],
        logger: logging.Logger | None = None,
        *,
        max_events: int = MOBILE_ACCESS_MAX_EVENTS,
        compact_after_bytes: int = MOBILE_ACCESS_COMPACT_BYTES,
    ) -> None:
        self._config_path_provider = config_path_provider
        self._logger = logger or logging.getLogger(__name__)
        self._max_events = max(1, int(max_events))
        self._compact_after_bytes = max(1, int(compact_after_bytes))
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return Path(self._config_path_provider()) / "data" / "mobile_access.jsonl"

    def append(self, kind: str, context: dict) -> None:
        """Append only request metadata needed to confirm phone connectivity."""
        try:
            event = {
                "timestamp": time.time(),
                "kind": kind,
                "request_time": context.get("request_time", ""),
                "request_host": context.get("request_host", ""),
                "request_path": context.get("request_path", ""),
                "remote_addr": context.get("remote_addr", ""),
                "forwarded_for": context.get("forwarded_for", ""),
                "is_android": bool(context.get("is_android")),
                "is_mobile": bool(context.get("is_mobile")),
                "user_agent": str(context.get("user_agent") or "")[:240],
            }
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                if self.path.stat().st_size > self._compact_after_bytes:
                    self._compact_locked()
        except Exception as exc:
            self._logger.debug("Could not write mobile access event: %s", exc)

    def _compact_locked(self) -> None:
        """Keep only the latest valid events; caller holds ``self._lock``."""
        recent_lines: deque[str] = deque(maxlen=self._max_events)
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue
                recent_lines.append(line.rstrip("\n"))

        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            for line in recent_lines:
                handle.write(line + "\n")
        os.replace(temporary_path, self.path)

    def read(self, limit: int = MOBILE_ACCESS_LOG_LIMIT) -> list[dict]:
        """Return the latest valid diagnostic events, newest first."""
        bounded_limit = max(1, int(limit))
        try:
            with self._lock:
                if not self.path.exists():
                    return []
                events: deque[dict] = deque(maxlen=bounded_limit)
                with self.path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            self._logger.debug("Could not read mobile access events: %s", exc)
            return []
        return list(reversed(events))


def _disable_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def register_mobile_access_routes(
    bp: Blueprint,
    *,
    startup_time: int,
    server_version: str,
    config_path_provider: Callable[[], str],
    tailnet_ip_provider: Callable[[], str] = local_tailnet_ip,
    logger: logging.Logger | None = None,
) -> None:
    """Register mobile diagnostics while preserving the existing public routes."""
    event_store = MobileAccessEventStore(config_path_provider, logger)

    @bp.before_app_request
    def log_mobile_access_probe():
        """Record mobile requests that do not target a diagnostic endpoint."""
        try:
            context = mobile_request_context()
        except RuntimeError:
            return
        if not (context.get("is_android") or context.get("is_mobile")):
            return
        if context.get("request_path") in {
            "/mobile",
            "/android",
            f"{ROUTE_PREFIX}/mobile",
            f"{ROUTE_PREFIX}/android",
            "/api/mobile-access",
            "/api/mobile-access/events",
        }:
            return
        event_store.append("auto", context)

    @bp.route("/api/mobile-access", methods=["GET"])
    def mobile_access_status():
        """Return phone-friendly access hints for Tailscale debugging."""
        tailnet_ip = tailnet_ip_provider()
        app_port = int(os.getenv("PORT", "5000"))
        nipio_host = nipio_tailnet_host(tailnet_ip)
        host = request.host.split(":", 1)[0]
        scheme = request.headers.get("X-Forwarded-Proto") or request.scheme or "http"
        request_context = mobile_request_context()
        event_store.append("api", request_context)
        recent_events = event_store.read(limit=10)
        recent_external_android = next(
            (
                event
                for event in recent_events
                if external_android_event(event, tailnet_ip)
            ),
            None,
        )
        return jsonify(
            {
                "status": "ok",
                "current_url": f"{scheme}://{request.host}/",
                "recommended_url": f"http://{tailnet_ip}{ROUTE_PREFIX}",
                "dns_fallback_url": f"http://{nipio_host}{ROUTE_PREFIX}",
                "fallback_url": f"http://{tailnet_ip}:{app_port}{ROUTE_PREFIX}",
                "second_fallback_url": "",
                "magicdns_url": configured_magicdns_url(),
                "tailnet_ip": tailnet_ip,
                "request_host": host,
                "server_version": server_version,
                "session_id": startup_time,
                "request": request_context,
                "external_android_seen": recent_external_android is not None,
                "latest_external_android_event": recent_external_android,
                "recent_events": recent_events,
                "notes": [
                    "Use recommended_url from the Android device when MagicDNS fails.",
                    "Use dns_fallback_url if Android rejects the bare IP or does not apply Tailscale DNS.",
                    "The recommended URL avoids DNS and proxies directly through Tailscale Serve.",
                    "If the phone still fails, open /api/mobile-access/events on the Mac to confirm whether Android reached Flask.",
                ],
            }
        )

    @bp.route("/api/mobile-access/events", methods=["GET"])
    def mobile_access_events():
        """Return recent mobile diagnostic hits, newest first."""
        request_context = mobile_request_context()
        event_store.append("events", request_context)
        return jsonify(
            {
                "status": "ok",
                "tailnet_ip": tailnet_ip_provider(),
                "server_version": server_version,
                "session_id": startup_time,
                "events": event_store.read(limit=40),
            }
        )

    @bp.route("/mobile", methods=["GET"])
    @bp.route("/android", methods=["GET"])
    @bp.route(f"{ROUTE_PREFIX}/mobile", methods=["GET"])
    @bp.route(f"{ROUTE_PREFIX}/android", methods=["GET"])
    def mobile_access_page():
        """Render a bundle-independent page to prove phone-to-Mac connectivity."""
        tailnet_ip = tailnet_ip_provider()
        app_port = int(os.getenv("PORT", "5000"))
        nipio_host = nipio_tailnet_host(tailnet_ip)
        recommended_url = f"http://{tailnet_ip}{ROUTE_PREFIX}"
        diagnostic_url = f"http://{tailnet_ip}{ROUTE_PREFIX}/android"
        dns_fallback_url = f"http://{nipio_host}{ROUTE_PREFIX}"
        dns_fallback_diagnostic_url = f"http://{nipio_host}{ROUTE_PREFIX}/android"
        direct_backend_url = f"http://{tailnet_ip}:{app_port}{ROUTE_PREFIX}"
        direct_backend_diagnostic_url = f"http://{tailnet_ip}:{app_port}{ROUTE_PREFIX}/android"
        request_context = mobile_request_context()
        event_store.append("page", request_context)
        user_agent = html.escape(request_context["user_agent"] or "Unavailable")
        remote_addr = html.escape(
            request_context["forwarded_for"]
            or request_context["remote_addr"]
            or "Unavailable"
        )
        request_host = html.escape(request_context["request_host"] or "Unavailable")
        request_time = html.escape(request_context["request_time"])
        recent_android_hit = next(
            (
                event
                for event in event_store.read(limit=20)
                if external_android_event(event, tailnet_ip)
            ),
            None,
        )
        recent_android_text = (
            f"{html.escape(recent_android_hit.get('request_time', ''))} · "
            f"{html.escape(recent_android_hit.get('request_host', ''))}"
            if recent_android_hit
            else "No Android requests have been recorded in the current local log yet."
        )
        device_label = (
            "Android detected" if request_context["is_android"] else "Non-Android browser"
        )
        response = make_response(
            f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VerbaLoom mobile access</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #07111d;
      color: #eef4ff;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(92vw, 34rem);
      padding: 1.25rem;
      border: 1px solid rgba(125, 162, 255, .35);
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(21, 35, 54, .95), rgba(9, 17, 29, .95));
      box-shadow: 0 18px 60px rgba(0, 0, 0, .35);
    }}
    h1 {{ margin: 0 0 .35rem; font-size: 1.45rem; }}
    p {{ color: #aebbd0; line-height: 1.45; }}
    a {{
      display: block;
      margin: .85rem 0;
      padding: .9rem 1rem;
      border-radius: 12px;
      background: #caff2d;
      color: #07111d;
      font-weight: 800;
      text-align: center;
      text-decoration: none;
      overflow-wrap: anywhere;
    }}
    a.secondary {{ background: rgba(202, 255, 45, .12); color: #dfffa0; border: 1px solid rgba(202, 255, 45, .35); }}
    a.warning {{ background: rgba(255, 212, 121, .14); color: #ffd479; border: 1px solid rgba(255, 212, 121, .4); }}
    code {{
      display: block;
      padding: .8rem;
      border-radius: 12px;
      background: rgba(255, 255, 255, .06);
      color: #d8e5ff;
      overflow-wrap: anywhere;
    }}
    dl {{
      display: grid;
      gap: .55rem;
      margin: 1rem 0 0;
      padding: .85rem;
      border-radius: 12px;
      background: rgba(255, 255, 255, .05);
    }}
    dt {{ color: #8f9db4; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .ok {{ color: #7ef2a3; font-weight: 800; }}
    .warn {{ color: #ffd479; font-weight: 800; }}
    .bad {{ color: #ff9aa7; font-weight: 800; }}
    .muted {{ font-size: .9rem; color: #8f9db4; }}
  </style>
</head>
<body>
  <main>
    <h1>VerbaLoom mobile access</h1>
    <p><span class="ok">Server reached.</span> If you can see this page from Android, Tailscale is reaching VerbaLoom.</p>
    <p><span class="{'ok' if request_context['is_android'] else 'warn'}">{device_label}.</span> This page shows the request received by the app.</p>
    <p><span class="bad">Do not use HTTPS with the IP address.</span> On Android, enter the complete URL beginning with <strong>http://</strong>.</p>
    <p>Start with this short diagnostic URL on Android:</p>
    <a class="secondary" href="{diagnostic_url}">{diagnostic_url}</a>
    <p>If this page works on Android, open the full app here:</p>
    <a href="{recommended_url}">{recommended_url}</a>
    <p class="muted">If Android rejects the bare IP or does not apply MagicDNS, use this hostname, which resolves to the same Tailscale IP:</p>
    <a class="secondary" href="{dns_fallback_diagnostic_url}">{dns_fallback_diagnostic_url}</a>
    <a class="secondary" href="{dns_fallback_url}">{dns_fallback_url}</a>
    <p class="muted">If the Tailscale Serve proxy fails, connect directly to the app port:</p>
    <a class="warning" href="{direct_backend_diagnostic_url}">{direct_backend_diagnostic_url}</a>
    <a class="secondary" href="{direct_backend_url}">{direct_backend_url}</a>
    <dl>
      <dt>Request host</dt>
      <dd>{request_host}</dd>
      <dt>Remote IP seen by Flask</dt>
      <dd>{remote_addr}</dd>
      <dt>User-Agent</dt>
      <dd>{user_agent}</dd>
      <dt>Server time</dt>
      <dd>{request_time}</dd>
      <dt>Latest recorded Android request</dt>
      <dd>{recent_android_text}</dd>
    </dl>
    <p class="muted">Version {server_version} · session {startup_time} · <a class="secondary" href="{REPOSITORY_URL}">GitHub repository</a></p>
  </main>
</body>
</html>"""
        )
        return _disable_cache(response)
