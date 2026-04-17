#!/usr/bin/env python3
"""Dev-only teleop connectivity checks (no HAL).

Subcommands:

- ``http`` — minimal aiohttp listener: ``/`` and ``/api/teleop-config`` (portal-like surface).
- ``signaling`` — outbound WebSocket client to a portal ``/ws/robot`` URL (signaling only, no video).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Default for local/manual signaling smoke when portal is on localhost:9000.
DEFAULT_SIGNALING_WS_URL = "ws://portal:9000/ws/robot"

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8765


def _cmd_http(args: argparse.Namespace) -> None:
    from aiohttp import web

    from teleop.portal.auth import teleop_auth_middleware
    from teleop.portal.client_config import teleop_client_config_handler
    from teleop.portal.ice_config import build_browser_ice_config
    from teleop.portal.settings import build_portal_auth_settings

    async def index(_request: web.Request) -> web.Response:
        return web.Response(text="teleop smoke http\n")

    app = web.Application(middlewares=[teleop_auth_middleware])
    app["browser_ice"] = build_browser_ice_config()
    app["portal_auth_settings"] = build_portal_auth_settings()
    app.router.add_get("/", index)
    app.router.add_get("/api/teleop-config", teleop_client_config_handler)
    web.run_app(app, host=args.host, port=args.port, print=None)


def _cmd_signaling(args: argparse.Namespace) -> None:
    from teleop.edge.portal_client import portal_client_loop
    from teleop.edge.robot_settings import build_teleop_edge_settings

    url = (args.url or "").strip() or DEFAULT_SIGNALING_WS_URL.strip()
    if not url:
        print("Pass --url (WebSocket to the teleop portal ``/ws/robot``).", file=sys.stderr)
        raise SystemExit(2)

    settings = build_teleop_edge_settings()
    try:
        asyncio.run(portal_client_loop(url, teleop_edge_settings=settings, video_track_factory=None))
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_http = sub.add_parser(
        "http",
        help="Serve minimal HTTP routes for portal API smoke checks.",
    )
    p_http.add_argument("--host", default=DEFAULT_HTTP_HOST)
    p_http.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT)
    p_http.set_defaults(_run=_cmd_http)

    p_sig = sub.add_parser(
        "signaling",
        help="Dial outbound WebSocket to portal /ws/robot (uses teleop.edge.robot_settings).",
    )
    p_sig.add_argument(
        "--url",
        default="",
        help=(
            "WebSocket URL (default: %s)"
            % (DEFAULT_SIGNALING_WS_URL,)
        ),
    )
    p_sig.set_defaults(_run=_cmd_signaling)

    args = parser.parse_args()
    args._run(args)


if __name__ == "__main__":
    main()
