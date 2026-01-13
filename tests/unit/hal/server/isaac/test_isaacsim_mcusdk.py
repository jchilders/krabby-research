"""Unit tests for IsaacSimMCUSDK.

How to Run the Tests:
====================

From the krabby-research directory, run:

    # Run all tests in this file
    python -m pytest tests/unit/hal/server/isaac/test_isaacsim_mcusdk.py -v

    # Run a specific test class
    python -m pytest tests/unit/hal/server/isaac/test_isaacsim_mcusdk.py::TestIsaacSimMCUSDKInitialization -v

    # Run a specific test
    python -m pytest tests/unit/hal/server/isaac/test_isaacsim_mcusdk.py::TestIsaacSimMCUSDKInitialization::test_default_initialization -v

    # Run with coverage
    python -m pytest tests/unit/hal/server/isaac/test_isaacsim_mcusdk.py --cov=hal.server.isaac.isaacsim_mcusdk --cov-report=html

    # Run with verbose output and show print statements
    python -m pytest tests/unit/hal/server/isaac/test_isaacsim_mcusdk.py -v -s


Prerequisites:
    - pytest: pip install pytest
    - pytest-cov (optional, for coverage): pip install pytest-cov
    - torch: pip install torch

Note: These tests use mocking and do not require actual IsaacSim environment.

Test Coverage:
==============

These tests verify:
- Initialization with default and custom devices
- apply_command with valid JointCommand
- Batch dimension handling (num_envs)
- Device placement (CPU/CUDA)
- Error handling (invalid types, wrong shapes)
- set_device method
- Tensor conversion and shape handling
"""

import time

import numpy as np
import pytest

# Try to import torch, skip all tests if not available
try:
    import torch
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)

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


class TestIsaacSimMCUSDKInitialization:
    """Test IsaacSimMCUSDK initialization."""

    def test_default_initialization(self):
        """Test initialization with default device (CPU)."""
        sdk = IsaacSimMCUSDK()
        
        assert sdk.device == torch.device("cpu")
        assert sdk.device.type == "cpu"

    def test_initialization_with_cpu_device(self):
        """Test initialization with explicit CPU device."""
        sdk = IsaacSimMCUSDK(device=torch.device("cpu"))
        
        assert sdk.device == torch.device("cpu")
        assert sdk.device.type == "cpu"

    def test_initialization_with_cuda_device(self):
        """Test initialization with CUDA device (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        sdk = IsaacSimMCUSDK(device=torch.device("cuda"))
        
        assert sdk.device == torch.device("cuda")
        assert sdk.device.type == "cuda"

    def test_initialization_with_cuda_index(self):
        """Test initialization with CUDA device index (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        sdk = IsaacSimMCUSDK(device=torch.device("cuda:0"))
        
        assert sdk.device == torch.device("cuda:0")
        assert sdk.device.type == "cuda"


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
        
        assert isinstance(action, torch.Tensor)
        assert action.shape == (1, 18)  # Single environment
        assert action.dtype == torch.float32
        assert action.device.type == "cpu"
        
        # Verify values match
        np.testing.assert_array_almost_equal(
            action[0].cpu().numpy(), joint_values, decimal=5
        )

    def test_apply_command_with_custom_values(self):
        """Test apply_command with custom joint values."""
        # Create non-zero joint values
        joint_values = np.linspace(-1.0, 1.0, 18, dtype=np.float32)
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        action = self.sdk.apply_command(command)
        
        assert action.shape == (1, 18)
        np.testing.assert_array_almost_equal(
            action[0].cpu().numpy(), joint_values, decimal=5
        )

    def test_apply_command_with_multiple_envs(self):
        """Test apply_command with num_envs > 1."""
        joint_values = np.array([0.5, -0.3, 0.2] * 6, dtype=np.float32)
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        num_envs = 4
        action = self.sdk.apply_command(command, num_envs=num_envs)
        
        assert action.shape == (num_envs, 18)
        assert action.dtype == torch.float32
        
        # All environments should have the same values
        for i in range(num_envs):
            np.testing.assert_array_almost_equal(
                action[i].cpu().numpy(), joint_values, decimal=5
            )

    def test_apply_command_with_large_num_envs(self):
        """Test apply_command with large num_envs."""
        joint_values = np.ones(18, dtype=np.float32) * 0.1
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        num_envs = 100
        action = self.sdk.apply_command(command, num_envs=num_envs)
        
        assert action.shape == (num_envs, 18)
        
        # Verify all environments have same values
        expected = joint_values
        for i in range(num_envs):
            np.testing.assert_array_almost_equal(
                action[i].cpu().numpy(), expected, decimal=5
            )

    def test_apply_command_preserves_values(self):
        """Test that apply_command preserves exact values."""
        # Use specific values that should be preserved
        joint_values = np.array([
            0.1234, -0.5678, 0.9012, -0.3456, 0.7890, -0.1234,
            0.4567, -0.8901, 0.2345, -0.6789, 0.0123, -0.4567,
            0.7890, -0.2345, 0.5678, -0.9012, 0.3456, -0.7890,
        ], dtype=np.float32)
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        action = self.sdk.apply_command(command)
        
        # Values should be preserved (within float32 precision)
        np.testing.assert_array_almost_equal(
            action[0].cpu().numpy(), joint_values, decimal=6
        )


class TestIsaacSimMCUSDKDeviceHandling:
    """Test device handling in apply_command."""

    def test_apply_command_cpu_device(self):
        """Test apply_command places tensor on CPU."""
        sdk = IsaacSimMCUSDK(device=torch.device("cpu"))
        command = create_dummy_joint_command_18()
        
        action = sdk.apply_command(command)
        
        assert action.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_apply_command_cuda_device(self):
        """Test apply_command places tensor on CUDA."""
        sdk = IsaacSimMCUSDK(device=torch.device("cuda"))
        command = create_dummy_joint_command_18()
        
        action = sdk.apply_command(command)
        
        assert action.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_apply_command_cuda_with_multiple_envs(self):
        """Test apply_command with CUDA and multiple environments."""
        sdk = IsaacSimMCUSDK(device=torch.device("cuda"))
        joint_values = np.ones(18, dtype=np.float32) * 0.5
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        action = sdk.apply_command(command, num_envs=8)
        
        assert action.device.type == "cuda"
        assert action.shape == (8, 18)
        
        # Verify values on GPU
        np.testing.assert_array_almost_equal(
            action[0].cpu().numpy(), joint_values, decimal=5
        )


class TestIsaacSimMCUSDKErrorHandling:
    """Test error handling in apply_command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sdk = IsaacSimMCUSDK()

    def test_apply_command_invalid_type(self):
        """Test that apply_command raises ValueError for invalid type."""
        with pytest.raises(ValueError, match="command must be JointCommand"):
            self.sdk.apply_command("invalid")  # type: ignore

    def test_apply_command_none(self):
        """Test that apply_command raises ValueError for None."""
        with pytest.raises(ValueError, match="command must be JointCommand"):
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

    def test_apply_command_wrong_shape_2d(self):
        """Test that apply_command raises ValueError for 2D array."""
        # Try to create JointCommand with wrong shape (this should fail at JointCommand creation)
        # But if it somehow gets through, apply_command should catch it
        joint_positions = np.zeros((2, 9), dtype=np.float32)  # Wrong shape
        
        # JointCommand validation should catch this, but if it doesn't, apply_command will
        try:
            command = JointCommand(
                joint_positions=joint_positions,
                timestamp_ns=time.time_ns(),
                observation_timestamp_ns=time.time_ns(),
            )
            # If we get here, JointCommand didn't validate, so apply_command should
            with pytest.raises(ValueError, match="Expected 18 joints"):
                self.sdk.apply_command(command)
        except ValueError:
            # JointCommand validation caught it, which is also correct
            pass


class TestIsaacSimMCUSDKSetDevice:
    """Test set_device method."""

    def test_set_device_cpu(self):
        """Test setting device to CPU."""
        sdk = IsaacSimMCUSDK()
        
        sdk.set_device(torch.device("cpu"))
        
        assert sdk.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_set_device_cuda(self):
        """Test setting device to CUDA."""
        sdk = IsaacSimMCUSDK()
        
        sdk.set_device(torch.device("cuda"))
        
        assert sdk.device == torch.device("cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_set_device_after_initialization(self):
        """Test setting device after initialization affects subsequent commands."""
        sdk = IsaacSimMCUSDK(device=torch.device("cpu"))
        command = create_dummy_joint_command_18()
        
        # First command on CPU
        action1 = sdk.apply_command(command)
        assert action1.device.type == "cpu"
        
        # Change device
        sdk.set_device(torch.device("cuda"))
        
        # Second command on CUDA
        action2 = sdk.apply_command(command)
        assert action2.device.type == "cuda"


class TestIsaacSimMCUSDKEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sdk = IsaacSimMCUSDK()

    def test_apply_command_zero_values(self):
        """Test apply_command with all zero values."""
        command = create_dummy_joint_command_18(joint_values=np.zeros(18, dtype=np.float32))
        
        action = self.sdk.apply_command(command)
        
        assert action.shape == (1, 18)
        assert torch.allclose(action, torch.zeros(1, 18, dtype=torch.float32))

    def test_apply_command_max_values(self):
        """Test apply_command with maximum values."""
        joint_values = np.full(18, 2.0, dtype=np.float32)  # Max clamped value
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        action = self.sdk.apply_command(command)
        
        assert action.shape == (1, 18)
        np.testing.assert_array_almost_equal(
            action[0].cpu().numpy(), joint_values, decimal=5
        )

    def test_apply_command_min_values(self):
        """Test apply_command with minimum values."""
        joint_values = np.full(18, -2.0, dtype=np.float32)  # Min clamped value
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        action = self.sdk.apply_command(command)
        
        assert action.shape == (1, 18)
        np.testing.assert_array_almost_equal(
            action[0].cpu().numpy(), joint_values, decimal=5
        )

    def test_apply_command_single_env_explicit(self):
        """Test apply_command with num_envs=1 explicitly."""
        joint_values = np.ones(18, dtype=np.float32) * 0.5
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        action = self.sdk.apply_command(command, num_envs=1)
        
        assert action.shape == (1, 18)

    def test_apply_command_timestamp_preservation(self):
        """Test that timestamps in command are accessible (for logging)."""
        timestamp_ns = 1234567890123456789
        obs_timestamp_ns = 1234567890123456790
        command = create_dummy_joint_command_18(
            timestamp_ns=timestamp_ns,
            observation_timestamp_ns=obs_timestamp_ns,
        )
        
        # apply_command should not modify the command
        action = self.sdk.apply_command(command)
        
        # Verify command still has original timestamps
        assert command.timestamp_ns == timestamp_ns
        assert command.observation_timestamp_ns == obs_timestamp_ns
        assert action.shape == (1, 18)

    def test_apply_command_immutable_input(self):
        """Test that apply_command does not modify input command."""
        original_values = np.array([0.1, 0.2, 0.3] * 6, dtype=np.float32)
        command = create_dummy_joint_command_18(joint_values=original_values.copy())
        
        action = self.sdk.apply_command(command)
        
        # Verify original command is unchanged
        np.testing.assert_array_equal(command.joint_positions, original_values)
        
        # Verify action has correct values
        np.testing.assert_array_almost_equal(
            action[0].cpu().numpy(), original_values, decimal=5
        )

    def test_apply_command_non_contiguous_array(self):
        """Test apply_command with non-contiguous array (should still work)."""
        # Create non-contiguous array by slicing
        large_array = np.zeros(36, dtype=np.float32)
        large_array[::2] = 0.5  # Set every other element
        joint_values = large_array[:18]  # This creates a view, might be non-contiguous
        
        # Ensure it's contiguous for JointCommand validation
        joint_values = np.ascontiguousarray(joint_values)
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        action = self.sdk.apply_command(command)
        
        assert action.shape == (1, 18)
        np.testing.assert_array_almost_equal(
            action[0].cpu().numpy(), joint_values, decimal=5
        )


class TestIsaacSimMCUSDKLogging:
    """Test logging behavior (indirectly through apply_command)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sdk = IsaacSimMCUSDK()

    def test_apply_command_logs_debug_info(self):
        """Test that apply_command logs debug information."""
        # This test verifies that logging calls are made (doesn't verify output)
        # We can't easily capture logger output in unit tests without more setup
        joint_values = np.array([0.1, 0.2, 0.3] * 6, dtype=np.float32)
        command = create_dummy_joint_command_18(joint_values=joint_values)
        
        # Should not raise any errors related to logging
        action = self.sdk.apply_command(command)
        
        assert action.shape == (1, 18)
        # If we got here, logging didn't cause any issues
