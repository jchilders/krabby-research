"""Portal HTTP auth — optional shared secret for ``/``, ``/ws*``, ``GET /api/teleop-config``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortalAuthSettings:
    """HTTP bearer / query token gate; default is built from module :data:`HTTP_TOKEN`."""

    http_token: str | None = None


# If set, required on ``/``, ``/ws*``, and ``GET /api/teleop-config``. ``/static/`` stays open.
HTTP_TOKEN: str | None = None


def build_portal_auth_settings() -> PortalAuthSettings:
    t = HTTP_TOKEN
    if t is None:
        return PortalAuthSettings(http_token=None)
    s = str(t).strip()
    return PortalAuthSettings(http_token=s or None)
