"""Portal relay forwards JSON text browser ↔ robot (FIFO pair)."""

from __future__ import annotations

import asyncio
import json

import aiohttp
from aiohttp import web

from teleop.portal.relay import create_portal_app


def test_portal_relays_hello_browser_to_robot(default_portal_kwargs) -> None:
    async def _run() -> None:
        app = create_portal_app(**default_portal_kwargs)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"

        hello = json.dumps({"type": "hello", "role": "browser", "version": 1})

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"{base}/ws/robot") as rw:
                    async with session.ws_connect(f"{base}/ws/browser") as bw:
                        await bw.send_str(hello)
                        msg = await asyncio.wait_for(rw.receive(), timeout=5.0)
                        assert msg.type == aiohttp.WSMsgType.TEXT
                        got = json.loads(msg.data)
        finally:
            await runner.cleanup()

        assert got["type"] == "hello"
        assert got.get("version") == 1

    asyncio.run(_run())
