"""WebSocket telemetry broadcaster for Jetson HAL server."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    import websockets
except ModuleNotFoundError:  # pragma: no cover - exercised only in missing-dependency envs
    websockets = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

INVALID_PATH_CLOSE_CODE = 4404


@dataclass
class TelemetryWebSocketConfig:
    """Configuration for telemetry websocket endpoint."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8787
    path: str = "/krabby/telemetry"
    publish_hz: float = 10.0
    fake_data: bool = False

    def __post_init__(self) -> None:
        if self.port <= 0 or self.port > 65535:
            raise ValueError("telemetry websocket port must be in range 1-65535")
        if self.publish_hz <= 0.0:
            raise ValueError("telemetry websocket publish_hz must be > 0")
        if not self.path.startswith("/"):
            raise ValueError("telemetry websocket path must start with '/'")


class TelemetryWebSocketServer:
    """Background websocket server that broadcasts latest telemetry snapshots."""

    def __init__(
        self,
        config: TelemetryWebSocketConfig,
        snapshot_provider: Callable[[], dict[str, Any]],
    ):
        self._config = config
        self._snapshot_provider = snapshot_provider
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._startup_event = threading.Event()
        self._startup_error: Optional[Exception] = None
        self._server = None
        self._broadcast_task: Optional[asyncio.Task[Any]] = None
        self._clients: set[Any] = set()

    def _resolve_request_path(self, websocket: Any, path: Optional[str]) -> str:
        if path:
            return path
        ws_path = getattr(websocket, "path", None)
        if ws_path:
            return ws_path
        request = getattr(websocket, "request", None)
        if request is not None:
            req_path = getattr(request, "path", None)
            if req_path:
                return req_path
        return ""

    async def _handle_client(self, websocket: Any, path: Optional[str] = None) -> None:
        request_path = self._resolve_request_path(websocket, path)
        base_path, _, _ = request_path.partition("?")

        if base_path != self._config.path:
            await websocket.close(code=INVALID_PATH_CLOSE_CODE, reason="invalid path")
            return

        self._clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)

    async def _broadcast_loop(self) -> None:
        interval_s = 1.0 / self._config.publish_hz
        try:
            while True:
                payload = self._snapshot_provider()
                message = json.dumps(payload, separators=(",", ":"))
                clients = list(self._clients)
                if clients:
                    results = await asyncio.gather(
                        *(client.send(message) for client in clients),
                        return_exceptions=True,
                    )
                    for client, result in zip(clients, results):
                        if isinstance(result, Exception):
                            self._clients.discard(client)
                            try:
                                await client.close()
                            except Exception:
                                pass
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return

    async def _startup_async(self) -> None:
        if websockets is None:
            raise RuntimeError(
                "websockets package is required for telemetry websocket server. "
                "Install dependency `websockets>=12.0`."
            )
        self._server = await websockets.serve(
            self._handle_client,
            self._config.host,
            self._config.port,
            ping_interval=20.0,
            ping_timeout=20.0,
        )
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def _shutdown_async(self) -> None:
        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None

        clients = list(self._clients)
        self._clients.clear()
        for client in clients:
            try:
                await client.close(code=1001, reason="server stopping")
            except Exception:
                pass

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._startup_async())
        except Exception as exc:
            self._startup_error = exc
            self._startup_event.set()
            self._loop.close()
            self._loop = None
            return

        self._startup_event.set()
        self._loop.run_forever()

        try:
            self._loop.run_until_complete(self._shutdown_async())
        finally:
            self._loop.close()
            self._loop = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._startup_error = None
        self._startup_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="telemetry-ws", daemon=True)
        self._thread.start()

        if not self._startup_event.wait(timeout=5.0):
            raise RuntimeError("timed out while starting telemetry websocket server")
        if self._startup_error is not None:
            raise RuntimeError(f"failed to start telemetry websocket server: {self._startup_error}")

        logger.info(
            "Telemetry websocket server started on ws://%s:%d%s",
            self._config.host,
            self._config.port,
            self._config.path,
        )

    def stop(self) -> None:
        if not self._thread:
            return

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("failed to stop telemetry websocket server thread")

        self._thread = None
        logger.info("Telemetry websocket server stopped")
