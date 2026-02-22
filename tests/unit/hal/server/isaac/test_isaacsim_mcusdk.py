"""Unit tests for IsaacSimMCUSDK.

The SDK accepts JointCommand and returns dict for array conversion.

How to Run the Tests:
====================

From the krabby-research directory, run:

    # Run all tests in this file
    python -m pytest tests/unit/hal/server/isaac/test_isaacsim_mcusdk.py -v

    # Run with coverage
    python -m pytest tests/unit/hal/server/isaac/test_isaacsim_mcusdk.py --cov=hal.server.isaac.isaacsim_mcusdk --cov-report=html

    # Run with verbose output and show print statements
    python -m pytest tests/unit/hal/server/isaac/test_isaacsim_mcusdk.py -v -s

Prerequisites:
    - pytest: pip install pytest
    - pytest-cov (optional, for coverage): pip install pytest-cov

Note: These tests do not require actual IsaacSim environment.

Test Coverage:
==============

These tests verify:
- apply_command with valid JointCommand
- Error handling (wrong type, None)
"""

import time

import numpy as np
import pytest

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.isaacsim_mcusdk import IsaacSimMCUSDK
from hal.server.isaac.robot_definition_krabby_quad import KRABBY_QUAD_DEFINITION
from hal.server.jetson.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


def _make_command(joint_names: tuple[str, ...], values: list[float]) -> JointCommand:
    """Build JointCommand from names and values."""
    return JointCommand(
        _joint_positions=np.array(values, dtype=np.float32),
        timestamp_ns=time.time_ns(),
        observation_timestamp_ns=time.time_ns(),
        joint_names=joint_names,
    )


class TestIsaacSimMCUSDKApplyCommand:
    """Test apply_command method (JointCommand in, dict out)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sdk = IsaacSimMCUSDK()

    def test_apply_command_basic(self):
        """Test basic apply_command with valid JointCommand."""
        names = KRABBY_HEX_DEFINITION.get_joint_names()
        values = [0.1, 0.2, 0.3] * 6  # 18 joints
        command = _make_command(names, values)

        result = self.sdk.apply_command(command)

        assert isinstance(result, dict)
        assert result == command.to_positions_dict()
        assert len(result) == 18
        for name, val in zip(names, values):
            assert result[name] == pytest.approx(val)


class TestIsaacSimMCUSDKErrorHandling:
    """Test error handling in apply_command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sdk = IsaacSimMCUSDK()

    def test_apply_command_wrong_type(self):
        """Test that apply_command fails for non-JointCommand (no runtime type check)."""
        with pytest.raises(AttributeError):
            self.sdk.apply_command({})  # type: ignore

    def test_apply_command_none(self):
        """Test that apply_command fails for None."""
        with pytest.raises(AttributeError):
            self.sdk.apply_command(None)  # type: ignore

    def test_apply_command_accepts_12_or_18_joints(self):
        """Test that apply_command accepts JointCommand with 12 (quad) or 18 (hex) joints."""
        names_12 = KRABBY_QUAD_DEFINITION.get_joint_names()
        cmd_12 = _make_command(names_12, [0.0] * 12)
        result_12 = self.sdk.apply_command(cmd_12, num_envs=1)
        assert len(result_12) == 12

        names_18 = KRABBY_HEX_DEFINITION.get_joint_names()
        cmd_18 = _make_command(names_18, [0.0] * 18)
        result_18 = self.sdk.apply_command(cmd_18, num_envs=1)
        assert len(result_18) == 18
