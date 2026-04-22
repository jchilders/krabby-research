"""Optional shared-secret gate for HTTP pages, WebSockets, JSON config, and portal relay."""

from __future__ import annotations

from typing import Optional

from aiohttp import web

from teleop.portal.app_keys import PORTAL_AUTH_SETTINGS_APP_KEY
from teleop.portal import settings as portal_settings
from teleop.portal.settings import PortalAuthSettings


def _normalize_optional_token(t: object | None) -> Optional[str]:
    if t is None:
        return None
    s = str(t).strip()
    return s or None


def expected_token(request: web.Request) -> Optional[str]:
    """Return required token string, or ``None`` if auth is disabled."""
    ps = request.app.get(PORTAL_AUTH_SETTINGS_APP_KEY)
    if ps is not None and not isinstance(ps, PortalAuthSettings):
        raise web.HTTPInternalServerError(
            text="teleop: app auth settings must be PortalAuthSettings or None"
        )
    if isinstance(ps, PortalAuthSettings):
        # App settings only override module defaults when explicitly set.
        # This keeps test/runtime behavior where mutating settings.HTTP_TOKEN
        # still takes effect if app-level auth config is None.
        app_token = _normalize_optional_token(ps.http_token)
        if app_token is not None:
            return app_token
    t = portal_settings.HTTP_TOKEN
    return _normalize_optional_token(t)


def token_from_request(request: web.Request) -> Optional[str]:
    """Extract bearer token from query, ``X-Krabby-Teleop-Token``, or ``Authorization: Bearer``."""
    q = (request.query.get("token") or "").strip()
    if q:
        return q
    h = (request.headers.get("X-Krabby-Teleop-Token") or "").strip()
    if h:
        return h
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def require_token(request: web.Request) -> None:
    """Raise ``HTTPForbidden`` when a token is configured but the request does not match."""
    exp = expected_token(request)
    if exp is None:
        return
    got = token_from_request(request)
    if got != exp:
        raise web.HTTPForbidden(
            text="teleop: missing or invalid token (use ?token=, X-Krabby-Teleop-Token, or Authorization: Bearer)"
        )


@web.middleware
async def teleop_auth_middleware(request: web.Request, handler):
    """Skip auth for ``/static/**`` so browser assets load without query strings."""
    if request.path.startswith("/static/"):
        return await handler(request)
    if expected_token(request) is not None:
        require_token(request)
    return await handler(request)
