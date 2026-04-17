"""teleop portal optional HTTP token gate (remote teleop server)."""

from __future__ import annotations

import asyncio

import aiohttp
from aiohttp import web

from teleop.portal import settings as portal_settings
from teleop.portal.relay import create_portal_app


def test_portal_index_forbidden_without_token(default_portal_kwargs) -> None:
    portal_settings.HTTP_TOKEN = "secret"

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
                async with session.get(f"http://127.0.0.1:{port}/") as resp:
                    assert resp.status == 403
        finally:
            await runner.cleanup()

    asyncio.run(_run())


def test_portal_teleop_config_forbidden_without_token(default_portal_kwargs) -> None:
    portal_settings.HTTP_TOKEN = "secret"

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
                    assert resp.status == 403
        finally:
            await runner.cleanup()

    asyncio.run(_run())


def test_portal_index_ok_with_query_token(default_portal_kwargs) -> None:
    portal_settings.HTTP_TOKEN = "secret"

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
                async with session.get(f"http://127.0.0.1:{port}/?token=secret") as resp:
                    assert resp.status == 200
                async with session.get(
                    f"http://127.0.0.1:{port}/api/teleop-config?token=secret"
                ) as resp2:
                    assert resp2.status == 200
        finally:
            await runner.cleanup()

    asyncio.run(_run())
