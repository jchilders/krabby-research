"""Integration tests for version compatibility."""

import time

import numpy as np
import pytest

from hal.client.client import HalClient
from hal.server import HalServerBase
from hal.client.config import HalClientConfig, HalServerConfig
from hal.client.observation.types import NavigationCommand
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from compute.parkour.parkour_types import ParkourObservation
from hal.client.data_structures.hardware import HardwareObservations
from hal.server.jetson.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from tests.helpers import create_dummy_hw_obs


def test_observation_receive():
    """Test receiving observation via HAL (server set_observation, client poll)."""
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_old",
        command_bind="inproc://test_command_old",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_old",
        command_endpoint="inproc://test_command_old",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    hw_obs.joint_positions[0] = 1.0
    server.set_observation(hw_obs)
    time.sleep(0.01)

    received_hw_obs = client.poll(timeout_ms=1000)

    # Should work with current schema
    assert received_hw_obs is not None

    client.close()
    server.close()


def test_forward_compatibility_unknown_fields():
    """Test forward compatibility (unknown fields ignored)."""
    # This tests that dataclasses with optional fields can handle
    # additional fields in future versions
    from hal.client.observation.types import NavigationCommand

    # Create command with current schema
    nav_cmd = NavigationCommand.create_now(vx=1.0, vy=0.0, yaw_rate=0.5)

    assert nav_cmd.timestamp_ns > 0
    assert nav_cmd.vx == 1.0
    assert nav_cmd.vy == 0.0
    assert nav_cmd.yaw_rate == 0.5

    # In future, if we add optional fields, they should have defaults
    # and existing code should continue to work


def test_action_dim_mismatch_detection():
    """Test action_dim mismatch detection."""
    import torch
    from compute.parkour.parkour_types import InferenceResponse

    # Create inference response with wrong action_dim
    action_wrong = torch.tensor([0.0] * 10, dtype=torch.float32)  # 10 instead of 12
    response_wrong = InferenceResponse.create_success(
        action=action_wrong,
        timing_breakdown=[],
    )

    # Validate with expected action_dim=12
    with pytest.raises(ValueError, match="action_dim"):
        response_wrong.validate_action_dim(12)

    # Should work with correct action_dim
    action_correct = torch.tensor([0.0] * 12, dtype=torch.float32)
    response_correct = InferenceResponse.create_success(
        action=action_correct,
        timing_breakdown=[],
    )
    response_correct.validate_action_dim(12)  # Should not raise


def test_model_io_complete():
    """Test ParkourModelIO is_complete() with valid observation and nav_cmd."""
    from compute.parkour.parkour_types import ParkourModelIO
    from hal.client.observation.types import NavigationCommand

    obs_dims = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
        KRABBY_HEX_DEFINITION
    )
    nav_cmd = NavigationCommand.create_now()
    observation = ParkourObservation.from_array(
        obs_dims,
        np.zeros(obs_dims.obs_dim, dtype=np.float32),
        timestamp_ns=time.time_ns(),
    )
    model_io = ParkourModelIO(
        timestamp_ns=time.time_ns(),
        nav_cmd=nav_cmd,
        observation=observation,
    )
    assert model_io.is_complete()

    import zmq
    
    shared_context2 = zmq.Context()
    
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_schema_check",
        command_bind="inproc://test_command_schema_check",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_schema_check",
        command_endpoint="inproc://test_command_schema_check",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    # Publish a dummy observation first to establish connection
    hw_obs_init = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    server.set_observation(hw_obs_init)
    client.poll(timeout_ms=1000)
    # Connection is now established

    # Publish observation
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    hw_obs.joint_positions[0] = 1.0
    server.set_observation(hw_obs)

    received_hw_obs = client.poll(timeout_ms=1000)
    assert received_hw_obs is not None

    client.close()
    server.close()
    shared_context2.term()

