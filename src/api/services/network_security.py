"""Secure defaults for browser origins and server bind addresses."""

from __future__ import annotations

import ipaddress

from src.utils.provider_security import endpoint_origin
from src.utils.branding import env_value


def allowed_browser_origins(raw: str | None = None) -> list[str]:
    """Return explicit HTTP(S) origins; wildcards are intentionally ignored."""
    value = env_value("ALLOWED_ORIGINS", "") if raw is None else raw
    origins: list[str] = []
    for item in str(value or "").split(","):
        candidate = item.strip()
        if not candidate or candidate == "*":
            continue
        origin = endpoint_origin(candidate)
        if origin and origin not in origins:
            origins.append(origin)
    return origins


def is_loopback_bind(host: str | None) -> bool:
    """Return whether a bind target is limited to the local machine."""
    value = str(host or "").strip().casefold()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def network_bind_is_explicitly_allowed(host: str | None) -> bool:
    """Require a second opt-in before binding this unauthenticated lab app publicly."""
    if is_loopback_bind(host):
        return True
    return str(env_value("ALLOW_NETWORK_BIND", "false")).strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def browser_origin_is_allowed(
    origin: str | None,
    request_origin: str | None,
    extra_origins: list[str] | tuple[str, ...] = (),
) -> bool:
    """Validate a browser Origin against same-origin and explicit UI origins."""
    if not origin:
        return True
    normalized = endpoint_origin(origin)
    if not normalized:
        return False
    allowed = {endpoint_origin(request_origin)}
    allowed.update(endpoint_origin(item) for item in extra_origins)
    allowed.discard("")
    return normalized in allowed
