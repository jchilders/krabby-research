"""Unit tests for Jetson telemetry websocket server."""

import asyncio
import inspect
import json
import socket
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[5]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

websockets = pytest.importorskip("websockets")
from websockets.exceptions import ConnectionClosed

from hal.server.jetson.telemetry_websocket import (
    TelemetryWebSocketConfig,
    TelemetryWebSocketServer,
    UNAUTHORIZED_CLOSE_CODE,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _connect_kwargs(headers: dict[str, str] | None = None) -> dict:
    if headers is None:
        return {}
    params = inspect.signature(websockets.connect).parameters
    if "additional_headers" in params:
        return {"additional_headers": headers}
    return {"extra_headers": headers}


def _run(coro):
    return asyncio.run(coro)


def test_accepts_valid_header_token():
    port = _find_free_port()
    payload = {
        "type": "joint_telemetry",
        "timestamp_ns": 100,
        "status": "disconnected",
        "source": "mcu",
        "joints": {},
    }
    server = TelemetryWebSocketServer(
        TelemetryWebSocketConfig(
            enabled=True,
            host="127.0.0.1",
            port=port,
            path="/krabby/telemetry",
            publish_hz=20.0,
            token="secret-token",
        ),
        snapshot_provider=lambda: payload,
    )
    server.start()

    async def _case():
        uri = f"ws://127.0.0.1:{port}/krabby/telemetry"
        headers = {"Authorization": "Bearer secret-token"}
        async with websockets.connect(uri, **_connect_kwargs(headers)) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            msg = json.loads(raw)
            assert msg["type"] == "joint_telemetry"

    try:
        _run(_case())
    finally:
        server.stop()


def test_accepts_valid_query_token():
    port = _find_free_port()
    payload = {
        "type": "joint_telemetry",
        "timestamp_ns": 100,
        "status": "disconnected",
        "source": "mcu",
        "joints": {},
    }
    server = TelemetryWebSocketServer(
        TelemetryWebSocketConfig(
            enabled=True,
            host="127.0.0.1",
            port=port,
            path="/krabby/telemetry",
            publish_hz=20.0,
            token="secret-token",
        ),
        snapshot_provider=lambda: payload,
    )
    server.start()

    async def _case():
        uri = f"ws://127.0.0.1:{port}/krabby/telemetry?token=secret-token"
        async with websockets.connect(uri) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            msg = json.loads(raw)
            assert msg["type"] == "joint_telemetry"

    try:
        _run(_case())
    finally:
        server.stop()


def test_rejects_missing_or_invalid_token():
    port = _find_free_port()
    payload = {
        "type": "joint_telemetry",
        "timestamp_ns": 100,
        "status": "disconnected",
        "source": "mcu",
        "joints": {},
    }
    server = TelemetryWebSocketServer(
        TelemetryWebSocketConfig(
            enabled=True,
            host="127.0.0.1",
            port=port,
            path="/krabby/telemetry",
            publish_hz=20.0,
            token="expected-token",
        ),
        snapshot_provider=lambda: payload,
    )
    server.start()

    async def _case():
        uri = f"ws://127.0.0.1:{port}/krabby/telemetry?token=wrong-token"
        async with websockets.connect(uri) as ws:
            with pytest.raises(ConnectionClosed):
                await asyncio.wait_for(ws.recv(), timeout=1.0)
            assert ws.close_code == UNAUTHORIZED_CLOSE_CODE

    try:
        _run(_case())
    finally:
        server.stop()


def test_payload_shape_is_json_joint_map():
    port = _find_free_port()
    payload = {
        "type": "joint_telemetry",
        "timestamp_ns": 1741036800000000000,
        "status": "connected",
        "source": "mcu",
        "joints": {
            "FLHY": {
                "pos": 0.723,
                "pot": 740,
                "current": 694,
                "en": [1, 1],
                "pwm": [0, 0],
                "saf": 0,
            }
        },
    }
    server = TelemetryWebSocketServer(
        TelemetryWebSocketConfig(
            enabled=True,
            host="127.0.0.1",
            port=port,
            path="/krabby/telemetry",
            publish_hz=20.0,
            token="secret-token",
        ),
        snapshot_provider=lambda: payload,
    )
    server.start()

    async def _case():
        uri = f"ws://127.0.0.1:{port}/krabby/telemetry?token=secret-token"
        async with websockets.connect(uri) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            msg = json.loads(raw)
            assert set(msg.keys()) == {"type", "timestamp_ns", "status", "source", "joints"}
            assert msg["status"] == "connected"
            assert "FLHY" in msg["joints"]
            assert msg["joints"]["FLHY"]["en"] == [1, 1]

    try:
        _run(_case())
    finally:
        server.stop()


def test_disconnected_payload_still_streams():
    port = _find_free_port()
    payload = {
        "type": "joint_telemetry",
        "timestamp_ns": 1741036800000000000,
        "status": "disconnected",
        "source": "mcu",
        "joints": {},
    }
    server = TelemetryWebSocketServer(
        TelemetryWebSocketConfig(
            enabled=True,
            host="127.0.0.1",
            port=port,
            path="/krabby/telemetry",
            publish_hz=20.0,
            token="secret-token",
        ),
        snapshot_provider=lambda: payload,
    )
    server.start()

    async def _case():
        uri = f"ws://127.0.0.1:{port}/krabby/telemetry?token=secret-token"
        async with websockets.connect(uri) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            msg = json.loads(raw)
            assert msg["status"] == "disconnected"
            assert msg["joints"] == {}

    try:
        _run(_case())
    finally:
        server.stop()
