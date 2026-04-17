"""Portal HTTP serves ``/api/teleop-config`` for browser WebRTC bootstrap."""

from __future__ import annotations

import asyncio
import json

import aiohttp
from aiohttp import web

from teleop.portal.relay import create_portal_app


def test_portal_api_teleop_config(default_portal_kwargs) -> None:
    async def _run() -> None:
        app = create_portal_app(**default_portal_kwargs)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/api/teleop-config") as resp:
                    assert resp.status == 200
                    data = json.loads(await resp.text())
                    assert data.get("version") == 1
                    assert isinstance(data.get("iceServers"), list)
                    assert data["iceServers"]
        finally:
            await runner.cleanup()

    asyncio.run(_run())
