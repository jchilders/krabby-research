"""JSON the browser fetches before creating ``RTCPeerConnection`` (STUN/TURN servers)."""

from __future__ import annotations

from aiohttp import web

from teleop.portal.app_keys import BROWSER_ICE_APP_KEY
from teleop.portal.ice_config import BrowserIceConfig


async def teleop_client_config_handler(request: web.Request) -> web.Response:
    """``GET /api/teleop-config`` — ``{"version":1,"iceServers":[...]}`` (``iceServers`` is the WebRTC field name)."""
    raw = request.app.get(BROWSER_ICE_APP_KEY)
    if not isinstance(raw, BrowserIceConfig):
        raise web.HTTPInternalServerError(
            text="teleop: app must set browser_ice to BrowserIceConfig (see teleop.portal.relay.create_portal_app)"
        )
    return web.json_response({"version": 1, "iceServers": raw.stun_turn_servers})
