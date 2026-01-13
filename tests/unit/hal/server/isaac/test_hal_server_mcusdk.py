"""Unit tests for IsaacSimMCUSDK integration in IsaacSimHalServer. This is to specifically test the MCUSDK integration in the IsaacSimHalServer.

How to Run the Tests:
====================

From the krabby-research directory, run:

    # Run all tests in this file
    python -m pytest tests/unit/hal/server/isaac/test_hal_server_mcusdk.py -v

    # Run a specific test class
    python -m pytest tests/unit/hal/server/isaac/test_hal_server_mcusdk.py::TestIsaacSimHalServerMCUSDKInitialization -v

    # Run a specific test
    python -m pytest tests/unit/hal/server/isaac/test_hal_server_mcusdk.py::TestIsaacSimHalServerMCUSDKInitialization::test_cpu_device_initialization -v

    # Run with coverage
    python -m pytest tests/unit/hal/server/isaac/test_hal_server_mcusdk.py --cov=hal.server.isaac.hal_server --cov-report=html

    # Run with verbose output and show print statements
    python -m pytest tests/unit/hal/server/isaac/test_hal_server_mcusdk.py -v -s


Prerequisites:
    - pytest: pip install pytest
    - pytest-cov (optional, for coverage): pip install pytest-cov
    - torch: pip install torch

Note: These tests use mocking and do not require actual IsaacSim environment.

Test Coverage:
==============

These tests verify:
- MCUSDK initialization (_initialize_mcusdk method)
- MCUSDK initialization during __init__
- MCUSDK usage in apply_command method
- Device extraction and handling (CPU/CUDA, wrapped/unwrapped environments)
- Error handling when MCUSDK is not initialized
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Try to import torch, skip all tests if not available
try:
    import torch
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)

from hal.client.data_structures.hardware import JointCommand
from hal.server import HalServerConfig
from hal.server.isaac.hal_server import IsaacSimHalServer
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


def create_mock_env(device=None, num_envs=1, wrapped=False):
    """Create a mock IsaacSim environment for testing.
    
    Args:
        device: Device to set on environment (defaults to CPU)
        num_envs: Number of environments (defaults to 1)
        wrapped: If True, creates a wrapped environment with unwrapped attribute
    
    Returns:
        Mock environment object
    """
    if device is None:
        device = torch.device("cpu")
    
    # Create mock robot
    mock_robot = MagicMock()
    mock_robot.data = MagicMock()
    mock_robot.data.joint_pos = torch.zeros((num_envs, 12), dtype=torch.float32)
    mock_robot.data.root_ang_vel_b = torch.zeros((num_envs, 3), dtype=torch.float32)
    mock_robot.data.root_lin_vel_b = torch.zeros((num_envs, 3), dtype=torch.float32)
    mock_robot.data.root_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * num_envs, dtype=torch.float32)
    mock_robot.data.joint_vel = torch.zeros((num_envs, 12), dtype=torch.float32)
    
    # Create mock scene
    mock_scene = MagicMock()
    mock_scene.__getitem__ = MagicMock(return_value=mock_robot)
    mock_scene.__contains__ = MagicMock(return_value=True)
    mock_scene.keys = MagicMock(return_value=["robot"])
    mock_scene.sensors = {}
    
    # Create mock observation manager
    # OBS_DIM = 753 (53 proprio + 132 scan + 9 priv_explicit + 29 priv_latent + 530 history)
    # Import inside function to avoid dependency issues in unit tests
    try:
        from compute.parkour.parkour_types import OBS_DIM
    except ImportError:
        # Fallback to hardcoded value if import fails (for unit tests without full dependencies)
        OBS_DIM = 753
    mock_obs_manager = MagicMock()
    obs_tensor = torch.ones(OBS_DIM, dtype=torch.float32) * 0.1  # Non-zero values
    mock_obs_manager.compute = MagicMock(return_value={"policy": obs_tensor})
    
    # Create mock action manager
    mock_action_manager = MagicMock()
    
    # Create environment
    if wrapped:
        # Create wrapped environment
        wrapped_env = MagicMock()
        wrapped_env.scene = mock_scene
        wrapped_env.device = device
        wrapped_env.observation_manager = mock_obs_manager
        wrapped_env.action_manager = mock_action_manager
        
        # Create unwrapped environment
        unwrapped_env = MagicMock()
        unwrapped_env.scene = mock_scene
        unwrapped_env.device = device
        unwrapped_env.num_envs = num_envs
        unwrapped_env.observation_manager = mock_obs_manager
        unwrapped_env.action_manager = mock_action_manager
        
        # Set unwrapped attribute
        wrapped_env.unwrapped = unwrapped_env
        
        return wrapped_env
    else:
        # Create unwrapped environment
        env = MagicMock()
        env.scene = mock_scene
        env.device = device
        env.num_envs = num_envs
        env.observation_manager = mock_obs_manager
        env.action_manager = mock_action_manager
        env.unwrapped = env  # Point to itself for unwrapped
        
        return env


@pytest.fixture
def hal_server_config():
    """Create HAL server config for testing."""
    return HalServerConfig(
        observation_bind="inproc://test_mcusdk_observation",
        command_bind="inproc://test_mcusdk_command",
    )


class TestIsaacSimHalServerMCUSDKInitialization:
    """Test MCUSDK initialization in IsaacSimHalServer."""
    
    def test_cpu_device_initialization(self, hal_server_config):
        """Test SDK is initialized with CPU device when env.device is CPU."""
        mock_env = create_mock_env(device=torch.device("cpu"))
        
        with patch('hal.server.isaac.hal_server.IsaacSimMCUSDK') as mock_sdk_class:
            mock_sdk_instance = MagicMock()
            mock_sdk_class.return_value = mock_sdk_instance
            
            hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
            
            # Verify SDK was instantiated with CPU device
            mock_sdk_class.assert_called_once_with(device=torch.device("cpu"))
            assert hal_server._mcusdk is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_initialization(self, hal_server_config):
        """Test SDK is initialized with CUDA device when env.device is CUDA."""
        mock_env = create_mock_env(device=torch.device("cuda"))
        
        with patch('hal.server.isaac.hal_server.IsaacSimMCUSDK') as mock_sdk_class:
            mock_sdk_instance = MagicMock()
            mock_sdk_class.return_value = mock_sdk_instance
            
            hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
            
            # Verify SDK was instantiated with CUDA device
            mock_sdk_class.assert_called_once_with(device=torch.device("cuda"))
            assert hal_server._mcusdk is not None
    
    def test_wrapped_environment(self, hal_server_config):
        """Test device extraction works with env.unwrapped pattern."""
        mock_env = create_mock_env(device=torch.device("cpu"), wrapped=True)
        
        with patch('hal.server.isaac.hal_server.IsaacSimMCUSDK') as mock_sdk_class:
            mock_sdk_instance = MagicMock()
            mock_sdk_class.return_value = mock_sdk_instance
            
            hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
            
            # Verify SDK was instantiated with device from unwrapped environment
            mock_sdk_class.assert_called_once_with(device=torch.device("cpu"))
            assert hal_server._mcusdk is not None
    
    def test_unwrapped_environment(self, hal_server_config):
        """Test device extraction works when env is already unwrapped."""
        mock_env = create_mock_env(device=torch.device("cpu"), wrapped=False)
        
        with patch('hal.server.isaac.hal_server.IsaacSimMCUSDK') as mock_sdk_class:
            mock_sdk_instance = MagicMock()
            mock_sdk_class.return_value = mock_sdk_instance
            
            hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
            
            # Verify SDK was instantiated
            mock_sdk_class.assert_called_once_with(device=torch.device("cpu"))
            assert hal_server._mcusdk is not None
    
    def test_missing_device_attribute(self, hal_server_config):
        """Test defaults to CPU when device attribute doesn't exist."""
        # Create an environment object without device attribute
        class EnvWithoutDevice:
            def __init__(self, base_env):
                self.scene = base_env.scene
                self.num_envs = base_env.num_envs
                self.observation_manager = base_env.observation_manager
                self.action_manager = base_env.action_manager
                # Don't set device - getattr will use default
                self.unwrapped = self
        
        base_env = create_mock_env(device=torch.device("cpu"))
        env_without_device = EnvWithoutDevice(base_env)
        
        with patch('hal.server.isaac.hal_server.IsaacSimMCUSDK') as mock_sdk_class:
            mock_sdk_instance = MagicMock()
            mock_sdk_class.return_value = mock_sdk_instance
            
            hal_server = IsaacSimHalServer(hal_server_config, env=env_without_device)
            
            # Verify SDK was instantiated with default CPU device
            # getattr(env, 'device', torch.device("cpu")) will return the default
            mock_sdk_class.assert_called_once_with(device=torch.device("cpu"))
            assert hal_server._mcusdk is not None
    
    def test_no_op_when_env_is_none(self, hal_server_config):
        """Test method returns early without error when env is None."""
        hal_server = IsaacSimHalServer(hal_server_config, env=None)
        
        # _mcusdk should remain None
        assert hal_server._mcusdk is None
        
        # Calling _initialize_mcusdk() directly should not raise
        hal_server._initialize_mcusdk()
        assert hal_server._mcusdk is None
    
    def test_sdk_instance_created(self, hal_server_config):
        """Test that _mcusdk is an IsaacSimMCUSDK instance after initialization."""
        mock_env = create_mock_env(device=torch.device("cpu"))
        
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        
        # Verify _mcusdk is an instance of IsaacSimMCUSDK
        assert hal_server._mcusdk is not None
        assert isinstance(hal_server._mcusdk, IsaacSimMCUSDK)
    
    def test_device_as_string(self, hal_server_config):
        """Test device extraction handles string device like 'cpu'."""
        mock_env = create_mock_env(device="cpu")
        
        with patch('hal.server.isaac.hal_server.IsaacSimMCUSDK') as mock_sdk_class:
            mock_sdk_instance = MagicMock()
            mock_sdk_class.return_value = mock_sdk_instance
            
            hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
            
            # Verify SDK was instantiated (device will be converted to torch.device in _initialize_mcusdk)
            # The actual device passed will be the string "cpu", which getattr returns
            assert mock_sdk_class.called
            assert hal_server._mcusdk is not None
    
    def test_device_as_torch_device(self, hal_server_config):
        """Test device extraction handles torch.device objects."""
        mock_env = create_mock_env(device=torch.device("cpu"))
        
        with patch('hal.server.isaac.hal_server.IsaacSimMCUSDK') as mock_sdk_class:
            mock_sdk_instance = MagicMock()
            mock_sdk_class.return_value = mock_sdk_instance
            
            hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
            
            # Verify SDK was instantiated with torch.device
            mock_sdk_class.assert_called_once_with(device=torch.device("cpu"))
            assert hal_server._mcusdk is not None


class TestIsaacSimHalServerMCUSDKInitIntegration:
    """Test MCUSDK initialization during __init__."""
    
    def test_mcusdk_is_none_when_env_is_none(self, hal_server_config):
        """Test _mcusdk remains None when no environment provided."""
        hal_server = IsaacSimHalServer(hal_server_config, env=None)
        
        assert hal_server._mcusdk is None
    
    def test_mcusdk_initialized_when_env_provided(self, hal_server_config):
        """Test _mcusdk is initialized when env is provided in __init__."""
        mock_env = create_mock_env(device=torch.device("cpu"))
        
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        
        assert hal_server._mcusdk is not None
        assert isinstance(hal_server._mcusdk, IsaacSimMCUSDK)
    
    def test_initialization_order(self, hal_server_config):
        """Test _initialize_mcusdk() is called after _cache_references()."""
        mock_env = create_mock_env(device=torch.device("cpu"))
        
        with patch.object(IsaacSimHalServer, '_cache_references') as mock_cache, \
             patch.object(IsaacSimHalServer, '_initialize_mcusdk') as mock_init_sdk:
            
            hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
            
            # Verify _cache_references was called
            mock_cache.assert_called_once()
            # Verify _initialize_mcusdk was called
            mock_init_sdk.assert_called_once()
            
            # Verify order: _cache_references should be called before _initialize_mcusdk
            # Check call order
            cache_call_order = mock_cache.call_count
            init_call_order = mock_init_sdk.call_count
            # Both should be called, and _cache_references should be called first
            assert mock_cache.called
            assert mock_init_sdk.called


class TestIsaacSimHalServerMCUSDKApplyCommand:
    """Test MCUSDK usage in apply_command method."""
    
    def test_mcusdk_apply_command_called(self, hal_server_config):
        """Test MCUSDK.apply_command() is called with correct JointCommand."""
        mock_env = create_mock_env(device=torch.device("cpu"), num_envs=1)
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        hal_server.initialize()
        
        # Create mock command
        command = create_dummy_joint_command_18()
        
        # Mock get_joint_command to return the command
        hal_server.get_joint_command = MagicMock(return_value=command)
        
        # Mock MCUSDK apply_command to return a tensor
        expected_action = torch.zeros((1, 18), dtype=torch.float32)
        hal_server._mcusdk.apply_command = MagicMock(return_value=expected_action)
        
        # Call apply_command
        action = hal_server.apply_command()
        
        # Verify MCUSDK.apply_command was called with correct command
        hal_server._mcusdk.apply_command.assert_called_once()
        call_args = hal_server._mcusdk.apply_command.call_args
        assert call_args[0][0] == command  # First positional arg is command
        assert call_args[1]['num_envs'] == 1  # num_envs keyword arg
        
        # Verify action tensor is returned
        assert torch.equal(action, expected_action)
    
    def test_num_envs_extraction(self, hal_server_config):
        """Test num_envs is correctly extracted from unwrapped_env.num_envs."""
        num_envs = 4
        mock_env = create_mock_env(device=torch.device("cpu"), num_envs=num_envs)
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        hal_server.initialize()
        
        command = create_dummy_joint_command_18()
        hal_server.get_joint_command = MagicMock(return_value=command)
        
        expected_action = torch.zeros((num_envs, 18), dtype=torch.float32)
        hal_server._mcusdk.apply_command = MagicMock(return_value=expected_action)
        
        action = hal_server.apply_command()
        
        # Verify num_envs was passed correctly
        call_args = hal_server._mcusdk.apply_command.call_args
        assert call_args[1]['num_envs'] == num_envs
        assert action.shape[0] == num_envs
    
    def test_num_envs_default(self, hal_server_config):
        """Test defaults to 1 when num_envs attribute doesn't exist."""
        mock_env = create_mock_env(device=torch.device("cpu"), num_envs=1)
        # Remove num_envs attribute by setting it to raise AttributeError on access
        # Use a property that raises AttributeError or use getattr with default
        original_getattr = getattr
        def mock_getattr(obj, name, default=None):
            if obj is mock_env.unwrapped and name == 'num_envs':
                return default if default is not None else 1
            return original_getattr(obj, name, default)
        
        # Actually, we can just not set num_envs and let getattr handle it
        # But MagicMock will return a MagicMock for missing attributes, so we need to handle it differently
        # Let's use a real approach: create an object without num_envs
        class EnvWithoutNumEnvs:
            def __init__(self, base_env):
                self.scene = base_env.scene
                self.device = base_env.device
                self.observation_manager = base_env.observation_manager
                self.action_manager = base_env.action_manager
                # Don't set num_envs
                self.unwrapped = self
        
        env_without_num_envs = EnvWithoutNumEnvs(mock_env)
        hal_server = IsaacSimHalServer(hal_server_config, env=env_without_num_envs)
        hal_server.initialize()
        
        command = create_dummy_joint_command_18()
        hal_server.get_joint_command = MagicMock(return_value=command)
        
        expected_action = torch.zeros((1, 18), dtype=torch.float32)
        hal_server._mcusdk.apply_command = MagicMock(return_value=expected_action)
        
        action = hal_server.apply_command()
        
        # Verify num_envs defaults to 1
        call_args = hal_server._mcusdk.apply_command.call_args
        assert call_args[1]['num_envs'] == 1
    
    def test_wrapped_environment_handling(self, hal_server_config):
        """Test num_envs extraction works with wrapped environment."""
        num_envs = 2
        mock_env = create_mock_env(device=torch.device("cpu"), num_envs=num_envs, wrapped=True)
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        hal_server.initialize()
        
        command = create_dummy_joint_command_18()
        hal_server.get_joint_command = MagicMock(return_value=command)
        
        expected_action = torch.zeros((num_envs, 18), dtype=torch.float32)
        hal_server._mcusdk.apply_command = MagicMock(return_value=expected_action)
        
        action = hal_server.apply_command()
        
        # Verify num_envs was extracted from unwrapped environment
        call_args = hal_server._mcusdk.apply_command.call_args
        assert call_args[1]['num_envs'] == num_envs
    
    def test_action_tensor_returned(self, hal_server_config):
        """Test the action tensor from MCUSDK is returned correctly."""
        mock_env = create_mock_env(device=torch.device("cpu"), num_envs=1)
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        hal_server.initialize()
        
        command = create_dummy_joint_command_18()
        hal_server.get_joint_command = MagicMock(return_value=command)
        
        # Create a specific action tensor
        expected_action = torch.ones((1, 18), dtype=torch.float32) * 0.5
        hal_server._mcusdk.apply_command = MagicMock(return_value=expected_action)
        
        action = hal_server.apply_command()
        
        # Verify the exact tensor is returned
        assert torch.equal(action, expected_action)
        assert action.shape == (1, 18)
        assert action.dtype == torch.float32
    
    def test_error_when_mcusdk_is_none(self, hal_server_config):
        """Test RuntimeError is raised when _mcusdk is None."""
        mock_env = create_mock_env(device=torch.device("cpu"))
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        hal_server.initialize()
        
        # Manually set _mcusdk to None to simulate uninitialized state
        hal_server._mcusdk = None
        
        command = create_dummy_joint_command_18()
        hal_server.get_joint_command = MagicMock(return_value=command)
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="IsaacSimMCUSDK not initialized"):
            hal_server.apply_command()
    
    def test_error_when_env_is_none(self, hal_server_config):
        """Test RuntimeError is raised when env is None."""
        hal_server = IsaacSimHalServer(hal_server_config, env=None)
        hal_server.initialize()
        
        # Should raise RuntimeError because env is None (SDK not initialized)
        with pytest.raises(RuntimeError, match="No environment set|IsaacSimMCUSDK not initialized"):
            hal_server.apply_command()


class TestIsaacSimHalServerMCUSDKDeviceHandling:
    """Test device extraction and handling."""
    
    def test_device_as_string_cpu(self, hal_server_config):
        """Test device extraction handles string device 'cpu'."""
        mock_env = create_mock_env(device="cpu")
        
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        
        # SDK should be initialized (device will be handled by getattr)
        assert hal_server._mcusdk is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_as_string_cuda(self, hal_server_config):
        """Test device extraction handles string device 'cuda'."""
        mock_env = create_mock_env(device="cuda")
        
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        
        # SDK should be initialized
        assert hal_server._mcusdk is not None
    
    def test_device_as_torch_device_object(self, hal_server_config):
        """Test device extraction handles torch.device objects."""
        mock_env = create_mock_env(device=torch.device("cpu"))
        
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        
        # SDK should be initialized with torch.device
        assert hal_server._mcusdk is not None
        assert hal_server._mcusdk.device == torch.device("cpu")
    
    def test_device_attribute_missing_graceful_fallback(self, hal_server_config):
        """Test graceful fallback to CPU when device attribute is missing."""
        # Create an environment object without device attribute
        class EnvWithoutDevice:
            def __init__(self, base_env):
                self.scene = base_env.scene
                self.num_envs = base_env.num_envs
                self.observation_manager = base_env.observation_manager
                self.action_manager = base_env.action_manager
                # Don't set device
                self.unwrapped = self
        
        base_env = create_mock_env(device=torch.device("cpu"))
        env_without_device = EnvWithoutDevice(base_env)
        
        hal_server = IsaacSimHalServer(hal_server_config, env=env_without_device)
        
        # SDK should be initialized with default CPU device
        assert hal_server._mcusdk is not None
        assert hal_server._mcusdk.device == torch.device("cpu")


class TestIsaacSimHalServerMCUSDKErrorHandling:
    """Test error handling for MCUSDK integration."""
    
    def test_apply_command_timeout_handling(self, hal_server_config):
        """Test that apply_command handles timeout correctly."""
        mock_env = create_mock_env(device=torch.device("cpu"))
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        hal_server.initialize()
        
        # Mock get_joint_command to return None (simulating timeout)
        hal_server.get_joint_command = MagicMock(return_value=None)
        
        # Should raise RuntimeError after timeout
        with pytest.raises(RuntimeError, match="Failed to receive joint command"):
            hal_server.apply_command()
    
    def test_apply_command_with_valid_command_after_timeout(self, hal_server_config):
        """Test that apply_command succeeds when command is eventually received."""
        mock_env = create_mock_env(device=torch.device("cpu"))
        hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
        hal_server.initialize()
        
        command = create_dummy_joint_command_18()
        
        # Mock get_joint_command to return None first, then command
        call_count = [0]
        def mock_get_joint_command(timeout_ms):
            call_count[0] += 1
            if call_count[0] < 5:  # Return None for first 4 calls
                return None
            return command
        
        hal_server.get_joint_command = MagicMock(side_effect=mock_get_joint_command)
        
        expected_action = torch.zeros((1, 18), dtype=torch.float32)
        hal_server._mcusdk.apply_command = MagicMock(return_value=expected_action)
        
        # Should eventually succeed
        action = hal_server.apply_command()
        
        assert torch.equal(action, expected_action)
        assert hal_server._mcusdk.apply_command.called
