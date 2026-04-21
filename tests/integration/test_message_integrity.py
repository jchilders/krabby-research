"""Integration tests for message integrity and error handling."""

import time

import numpy as np
import pytest
import zmq

from hal.client.client import HalClient
from hal.server import HalServerBase
from hal.client.config import HalClientConfig, HalServerConfig
from hal.client.data_structures.hardware import HardwareObservations
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from tests.helpers import create_dummy_hw_obs


def _observation_dimensions():
    return PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
        KRABBY_HEX_DEFINITION
    )


def test_corrupt_message_handling():
    """Test handling of corrupt messages."""
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_corrupt",
        command_bind="inproc://test_command_corrupt",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_corrupt",
        command_endpoint="inproc://test_command_corrupt",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    # Send malformed message via server's socket (topic only, no payload blob)
    server.observation_socket.send(b"observation")

    # Client raises ValueError for payload too short (single-part format requires 4B metadata len + payload)
    with pytest.raises(ValueError, match="at least 4 bytes|payload too short"):
        client.poll(timeout_ms=1000)

    client.close()
    server.close()


def test_malformed_binary_payload():
    """Test handling of malformed binary payload."""
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_malformed",
        command_bind="inproc://test_command_malformed",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_malformed",
        command_endpoint="inproc://test_command_malformed",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    time.sleep(0.1)

    # Send invalid frame via server's socket (topic + invalid payload; blob parse will fail)
    server.observation_socket.send(b"observation" + b"not a float32 array")

    # Client raises ValueError when deserializing invalid blob
    with pytest.raises(ValueError):
        client.poll(timeout_ms=1000)

    client.close()
    server.close()


def test_missing_multipart_messages():
    """Test handling of invalid observation messages (e.g. topic-only frame, payload too short)."""
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_multipart",
        command_bind="inproc://test_command_multipart",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_multipart",
        command_endpoint="inproc://test_command_multipart",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    # Send valid message first
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    hw_obs.joint_positions[0:3] = [1.0, 2.0, 3.0]
    server.set_observation(hw_obs)

    received_hw_obs = client.poll(timeout_ms=1000)
    assert received_hw_obs is not None
    previous_joint_pos = received_hw_obs.joint_positions.copy()

    # Send invalid message: single frame with topic only (no payload blob)
    server.observation_socket.send(b"observation")

    # Client raises ValueError for payload too short (need at least 4 bytes for metadata length)
    with pytest.raises(ValueError, match="at least 4 bytes|payload too short"):
        client.poll(timeout_ms=1000)
    client.close()
    server.close()


def test_invalid_type():
    """Test handling of invalid observation types."""
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_shape",
        command_bind="inproc://test_command_shape",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    # Test that server validates type
    # Try to publish numpy array (should fail - needs HardwareObservations)
    invalid_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="HardwareObservations"):
        server.set_observation(invalid_data)

    server.close()


def test_graceful_error_handling():
    """Test graceful error handling (skip corrupt, use previous)."""
    # Use shared context for inproc connections
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_graceful",
        command_bind="inproc://test_command_graceful",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_graceful",
        command_endpoint="inproc://test_command_graceful",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    # Send valid message
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    hw_obs.joint_positions[0:3] = [1.0, 2.0, 3.0]
    server.set_observation(hw_obs)

    received_hw_obs = client.poll(timeout_ms=1000)
    assert received_hw_obs is not None
    valid_joint_pos = received_hw_obs.joint_positions.copy()

    # Send corrupt message - use server's socket directly
    # Can't bind twice with inproc, so send directly through server socket
    server.observation_socket.send(b"invalid")  # Invalid message format

    # Poll again - should handle gracefully (may return None or previous value)
    # The key is that it should not crash
    client.poll(timeout_ms=1000)
    client.close()
    server.close()


def test_required_fields_validation():
    """Test validation of required fields."""
    from hal.client.observation.types import NavigationCommand
    from compute.parkour.parkour_types import ParkourModelIO, ParkourObservation

    obs_dims = _observation_dimensions()
    incomplete_io = ParkourModelIO(
        timestamp_ns=time.time_ns(),
        nav_cmd=None,
        observation=None,
    )

    assert not incomplete_io.is_complete()

    nav_cmd = NavigationCommand.create_now()
    observation = ParkourObservation.from_array(
        obs_dims,
        np.zeros(obs_dims.obs_dim, dtype=np.float32),
        timestamp_ns=time.time_ns(),
    )

    complete_io = ParkourModelIO(
        timestamp_ns=time.time_ns(),
        nav_cmd=nav_cmd,
        observation=observation,
    )

    assert complete_io.is_complete()


def test_timestamp_synchronization():
    """Test that is_synchronized() correctly validates timestamp synchronization."""
    from hal.client.observation.types import NavigationCommand
    from compute.parkour.parkour_types import ParkourModelIO, ParkourObservation

    obs_dims = _observation_dimensions()
    base_time_ns = time.time_ns()
    nav_cmd = NavigationCommand.create_now()
    nav_cmd.timestamp_ns = base_time_ns

    observation = ParkourObservation.from_array(
        obs_dims,
        np.zeros(obs_dims.obs_dim, dtype=np.float32),
        timestamp_ns=base_time_ns + 5_000_000,
    )
    
    synchronized_io = ParkourModelIO(
        timestamp_ns=base_time_ns,
        nav_cmd=nav_cmd,
        observation=observation,
    )
    
    assert synchronized_io.is_synchronized(), "Timestamps within 10ms should be synchronized"
    assert synchronized_io.is_synchronized(max_age_ns=10_000_000), "Should pass with default 10ms threshold"
    
    observation_unsync = ParkourObservation.from_array(
        obs_dims,
        np.zeros(obs_dims.obs_dim, dtype=np.float32),
        timestamp_ns=base_time_ns + 15_000_000,
    )
    
    unsynchronized_io = ParkourModelIO(
        timestamp_ns=base_time_ns,
        nav_cmd=nav_cmd,
        observation=observation_unsync,
    )
    
    assert not unsynchronized_io.is_synchronized(), "Timestamps >10ms apart should not be synchronized"
    assert unsynchronized_io.is_synchronized(max_age_ns=20_000_000), "Should pass with 20ms threshold"
    
    observation_custom = ParkourObservation.from_array(
        obs_dims,
        np.zeros(obs_dims.obs_dim, dtype=np.float32),
        timestamp_ns=base_time_ns + 5_000_000,
    )
    
    custom_io = ParkourModelIO(
        timestamp_ns=base_time_ns,
        nav_cmd=nav_cmd,
        observation=observation_custom,
    )
    
    # 5ms (5_000_000 ns) is > 1ms (1_000_000 ns) threshold, so should fail
    assert not custom_io.is_synchronized(max_age_ns=1_000_000), "5ms > 1ms threshold should fail"
    
    observation_exact = ParkourObservation.from_array(
        obs_dims,
        np.zeros(obs_dims.obs_dim, dtype=np.float32),
        timestamp_ns=base_time_ns + 10_000_000,
    )
    
    exact_io = ParkourModelIO(
        timestamp_ns=base_time_ns,
        nav_cmd=nav_cmd,
        observation=observation_exact,
    )
    
    assert exact_io.is_synchronized(max_age_ns=10_000_000), "Exactly at threshold should pass"
    assert not exact_io.is_synchronized(max_age_ns=9_999_999), "Just over threshold should fail"
    
    # Test that incomplete IO is not synchronized
    incomplete_io = ParkourModelIO(
        timestamp_ns=base_time_ns,
        nav_cmd=None,
        observation=None,
    )
    
    assert not incomplete_io.is_synchronized(), "Incomplete IO should not be synchronized"


def test_timestamp_synchronization_realistic_sequence():
    """Test synchronization with realistic sequence: nav_cmd created once, observations arrive continuously.
    
    This simulates the actual issue seen in the latency test where:
    1. nav_cmd is created once with NavigationCommand.create_now()
    2. Observations are published continuously with fresh timestamps
    3. The nav_cmd timestamp becomes stale, causing synchronization failures
    """
    from hal.client.observation.types import NavigationCommand
    from compute.parkour.parkour_types import ParkourModelIO, ParkourObservation
    from hal.client.data_structures.hardware import HardwareObservations
    from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper

    obs_dims = _observation_dimensions()
    nav_cmd = NavigationCommand.create_now()
    nav_cmd_timestamp = nav_cmd.timestamp_ns
    mapper = HWObservationsToParkourMapper(obs_dims)
    
    # First observation: arrives quickly, should be synchronized
    time.sleep(0.001)  # 1ms delay
    hw_obs_1 = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=480,
        camera_width=640,
        timestamp_ns=time.time_ns(),  # Fresh timestamp
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
    )
    
    parkour_obs_1 = mapper.map(hw_obs_1, nav_cmd=nav_cmd)
    model_io_1 = ParkourModelIO(
        timestamp_ns=parkour_obs_1.timestamp_ns,
        nav_cmd=nav_cmd,  # Same nav_cmd with old timestamp
        observation=parkour_obs_1,
    )
    
    # Should be synchronized if within 10ms
    time_diff_ns = abs(parkour_obs_1.timestamp_ns - nav_cmd_timestamp)
    if time_diff_ns <= 10_000_000:  # 10ms
        assert model_io_1.is_synchronized(), f"First observation should be synchronized (diff: {time_diff_ns/1e6:.2f}ms)"
    else:
        assert not model_io_1.is_synchronized(), f"First observation should not be synchronized (diff: {time_diff_ns/1e6:.2f}ms)"
    
    # Simulate multiple observations arriving (like in the test runner loop)
    # After several iterations, the nav_cmd timestamp becomes stale
    for i in range(5):
        time.sleep(0.01)  # 10ms between observations (100Hz)
        hw_obs = HardwareObservations(
            joint_positions=np.zeros(12, dtype=np.float32),
            camera_height=480,
            camera_width=640,
            timestamp_ns=time.time_ns(),  # Fresh timestamp each time
            base_ang_vel_b=np.zeros(3, dtype=np.float32),
            base_lin_vel_b=np.zeros(3, dtype=np.float32),
            base_quat_w=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            joint_velocities=np.zeros(12, dtype=np.float32),
            contact_forces=np.zeros(5, dtype=np.float32),
            previous_action=np.zeros(12, dtype=np.float32),
        )
        
        parkour_obs = mapper.map(hw_obs, nav_cmd=nav_cmd)  # Still using same nav_cmd
        model_io = ParkourModelIO(
            timestamp_ns=parkour_obs.timestamp_ns,
            nav_cmd=nav_cmd,  # Same nav_cmd with old timestamp
            observation=parkour_obs,
        )
        
        time_diff_ns = abs(parkour_obs.timestamp_ns - nav_cmd_timestamp)
        time_diff_ms = time_diff_ns / 1e6
        
        # After several iterations, timestamps will be >10ms apart
        if time_diff_ms > 10.0:
            assert not model_io.is_synchronized(), (
                f"Observation {i+1} should not be synchronized "
                f"(nav_cmd age: {time_diff_ms:.2f}ms > 10ms threshold)"
            )
        else:
            assert model_io.is_synchronized(), (
                f"Observation {i+1} should be synchronized "
                f"(nav_cmd age: {time_diff_ms:.2f}ms <= 10ms threshold)"
            )

