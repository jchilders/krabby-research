"""Unit tests for IsaacSimMCUSDK.

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
- Error handling (invalid types, wrong shapes)
"""

import time

import numpy as np
import pytest

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.isaacsim_mcusdk import IsaacSimMCUSDK


def create_dummy_joint_command_18(
    timestamp_ns: int = None,
    observation_timestamp_ns: int = None,
    joint_values: np.ndarray = None,
) -> JointCommand:
    """Create dummy joint command with 18 joints for hexapod testing.
    
    Args:
        timestamp_ns: Optional timestamp (defaults to current time)
        observation_timestamp_ns: Optional observation timestamp (defaults to timestamp_ns)
        joint_values: Optional joint position values (defaults to zeros)
    
    Returns:
        JointCommand with 18 joints
    """
    if timestamp_ns is None:
        timestamp_ns = time.time_ns()
    if observation_timestamp_ns is None:
        observation_timestamp_ns = timestamp_ns
    if joint_values is None:
        joint_values = np.zeros(18, dtype=np.float32)
    else:
        joint_values = np.asarray(joint_values, dtype=np.float32)
        if joint_values.shape != (18,):
            raise ValueError(f"joint_values must have shape (18,), got {joint_values.shape}")
    
    return JointCommand(
        joint_positions=joint_values,
        timestamp_ns=timestamp_ns,
        observation_timestamp_ns=observation_timestamp_ns,
    )


class TestIsaacSimMCUSDKApplyCommand:
    """Test apply_command method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sdk = IsaacSimMCUSDK()

    def test_apply_command_basic(self):
        """Test basic apply_command with valid JointCommand."""
        joint_values = np.array([0.1, 0.2, 0.3] * 6, dtype=np.float32)  # 18 joints
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        action = self.sdk.apply_command(command)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (18,)
        assert action.dtype == np.float32
        
        # Verify values match
        np.testing.assert_array_almost_equal(action, joint_values, decimal=5)


class TestIsaacSimMCUSDKErrorHandling:
    """Test error handling in apply_command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sdk = IsaacSimMCUSDK()

    def test_apply_command_invalid_type(self):
        """Test that apply_command raises AttributeError for invalid type."""
        with pytest.raises(AttributeError):
            self.sdk.apply_command("invalid")  # type: ignore

    def test_apply_command_none(self):
        """Test that apply_command raises AttributeError for None."""
        with pytest.raises(AttributeError):
            self.sdk.apply_command(None)  # type: ignore

    def test_apply_command_wrong_joint_count(self):
        """Test that apply_command raises ValueError for wrong joint count."""
        # JointCommand validation catches wrong joint count before apply_command
        # Try to create JointCommand with 12 joints instead of 18
        with pytest.raises(ValueError, match="joint_positions shape.*!=.*18"):
            JointCommand(
                joint_positions=np.zeros(12, dtype=np.float32),
                timestamp_ns=time.time_ns(),
                observation_timestamp_ns=time.time_ns(),
            )
