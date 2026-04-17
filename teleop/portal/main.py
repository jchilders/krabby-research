"""CLI: run the minimal signaling portal (pair browser + robot relays)."""

from __future__ import annotations

import argparse
import logging

from aiohttp import web

from teleop.portal.ice_config import build_browser_ice_config
from teleop.portal.relay import create_portal_app
from teleop.portal.settings import build_portal_auth_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Krabby teleop portal — pair browser/edge signaling relay")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app = create_portal_app(
        browser_ice=build_browser_ice_config(),
        portal_auth_settings=build_portal_auth_settings(),
    )
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
