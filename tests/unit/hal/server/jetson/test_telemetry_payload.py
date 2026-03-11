"""Unit tests for Jetson telemetry payload generation."""

import sys
from pathlib import Path

from types import SimpleNamespace
from unittest.mock import Mock, patch

_root = Path(__file__).resolve().parents[5]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from hal.server.config import HalServerConfig
from hal.server.jetson.hal_server import FAKE_TELEMETRY_JOINTS, JetsonHalServer
from hal.server.jetson.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from hal.server.jetson.telemetry_websocket import TelemetryWebSocketConfig


def _make_server(telemetry_cfg: TelemetryWebSocketConfig) -> JetsonHalServer:
    obs_dims = SimpleNamespace(num_scan=0)
    return JetsonHalServer(
        HalServerConfig(
            observation_bind="inproc://test_payload_obs",
            command_bind="inproc://test_payload_cmd",
        ),
        observation_dimensions=obs_dims,
        action_dim=0,
        robot_definition=KRABBY_HEX_DEFINITION,
        mcu_auto_connect=False,
        telemetry_ws_config=telemetry_cfg,
    )


@patch("hal.server.jetson.hal_server.KrabbyMCUSDK", Mock())
def test_fake_payload_mode_generates_connected_joint_data():
    server = _make_server(TelemetryWebSocketConfig(enabled=False, fake_data=True))
    payload = server._build_telemetry_payload()
    server.close()

    assert payload["type"] == "joint_telemetry"
    assert payload["status"] == "connected"
    assert payload["source"] == "fake"
    assert isinstance(payload["timestamp_ns"], int)
    assert set(payload["joints"].keys()) == set(FAKE_TELEMETRY_JOINTS)

    sample = payload["joints"]["FLHY"]
    assert 0.0 <= sample["pos"] <= 1.0
    assert 0 <= sample["pot"] <= 1023
    assert sample["en"] == [1, 1]
    assert len(sample["pwm"]) == 2
    assert sample["saf"] == 0


@patch("hal.server.jetson.hal_server.KrabbyMCUSDK", Mock())
def test_default_payload_without_mcu_is_disconnected():
    server = _make_server(TelemetryWebSocketConfig(enabled=False, fake_data=False))
    server._mcusdk = None
    payload = server._build_telemetry_payload()
    server.close()

    assert payload["status"] == "disconnected"
    assert payload["source"] == "mcu"
    assert payload["joints"] == {}
