### Unit tests for KrabbyMCUSDK (hal/server/jetson/krabby_mcusdk.py).
### Run: pytest tests/unit/hal/server/jetson/test_krabby_mcusdk.py -v

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[5]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
from unittest.mock import Mock, patch

from hal.server.jetson.krabby_mcusdk import (
    JOINT_LIMIT_RAD,
    JOINT_NEUTRAL,
    KrabbyMCUSDK,
    _hal_to_firmware_name,
    _map_mcu_joints_to_normalized,
    _rad_to_pwm,
)
from hal.server.jetson.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


class TestHalToFirmwareName:
    def test_known_suffixes(self):
        assert _hal_to_firmware_name("FL_hip_yaw") == "FLHY"
        assert _hal_to_firmware_name("FL_hip_pitch") == "FLHL"
        assert _hal_to_firmware_name("FL_knee") == "FLKL"
        assert _hal_to_firmware_name("RR_hip_yaw") == "RRHY"

    def test_unknown_suffix_first_two_chars_upper(self):
        assert _hal_to_firmware_name("FL_ab") == "FLAB"

    def test_short_suffix_fallback(self):
        assert _hal_to_firmware_name("FL_x") == "FL??"


class TestRadToPwm:
    def test_zero_rad_gives_zero_pwm(self):
        assert _rad_to_pwm(0.0) == 0

    def test_positive_and_negative(self):
        assert _rad_to_pwm(0.1) == 51
        assert _rad_to_pwm(-0.1) == -51

    def test_clamp_at_limits(self):
        assert _rad_to_pwm(JOINT_LIMIT_RAD) == 255
        assert _rad_to_pwm(-JOINT_LIMIT_RAD) == -255
        assert _rad_to_pwm(1.0) == 255
        assert _rad_to_pwm(-1.0) == -255


class TestMapMcuJointsToNormalized:
    def test_firmware_keys_and_normalized_range(self):
        mcu_joints = ("FL_hip_yaw", "FL_hip_pitch")
        command = {"FL_hip_yaw": 0.0, "FL_hip_pitch": JOINT_LIMIT_RAD}
        out = _map_mcu_joints_to_normalized(command, mcu_joints)
        assert set(out.keys()) == {"FLHY", "FLHL"}
        assert out["FLHY"] == pytest.approx(JOINT_NEUTRAL)
        assert out["FLHL"] == pytest.approx(1.0)
        for v in out.values():
            assert 0.0 <= v <= 1.0

    def test_missing_joint_defaults_to_zero_rad(self):
        mcu_joints = ("FL_knee",)
        command = {}
        out = _map_mcu_joints_to_normalized(command, mcu_joints)
        assert out["FLKL"] == pytest.approx(JOINT_NEUTRAL)


class TestKrabbyMCUSDKInit:
    @patch("hal.server.jetson.krabby_mcusdk.FirmwareKrabbyMCUSDK", Mock())
    def test_init_raises_value_error_for_wrong_joint_count(self):
        with pytest.raises(ValueError, match="18 names.*got 17"):
            KrabbyMCUSDK(mcu_joints=("A",) * 17, auto_connect=False)
        with pytest.raises(ValueError, match="18 names.*got 19"):
            KrabbyMCUSDK(mcu_joints=("A",) * 19, auto_connect=False)
    def test_init_succeeds_with_18_joints(self):
        mcu_joints = KRABBY_HEX_DEFINITION.get_mcu_joints()
        assert len(mcu_joints) == 18
        sdk = KrabbyMCUSDK(mcu_joints=mcu_joints, auto_connect=False)
        assert sdk._mcu_joints == mcu_joints
