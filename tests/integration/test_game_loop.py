"""Integration tests for inference logic (game loop)."""

import threading
import time

import numpy as np
import pytest
import torch
import zmq

from hal.client.client import HalClient
from hal.server import HalServerBase
from hal.client.config import HalClientConfig, HalServerConfig
from hal.client.observation.types import NavigationCommand
from hal.client.data_structures.hardware import HardwareObservations
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from compute.testing.inference_test_runner import InferenceTestRunner
from tests.helpers import create_dummy_hw_obs


class ProtoHalServer(HalServerBase):
    """Proto HAL server for testing - publishes synthetic observations in training format.

    This is a minimal HAL server that publishes observations matching the training format
    exactly: [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
    """

    def __init__(self, config):
        """Initialize proto HAL server.
        
        Args:
            config: HAL server configuration
        """
        super().__init__(config)
        self.tick_count = 0
        self._running = False
        self._publish_thread = None
        self._command_thread = None

    def start_publishing(self, rate_hz: float = 100.0):
        """Start publishing observations at specified rate.

        Args:
            rate_hz: Publication rate in Hz
        """
        if self._running:
            return

        self._running = True
        period = 1.0 / rate_hz

        def publish_loop():
            """Background loop that publishes synthetic observations at a fixed rate."""
            while self._running:
                # Publish synthetic observation in training format
                self.publish_observation()
                self.tick_count += 1
                time.sleep(period)

        def command_loop():
            """Handle incoming commands in a loop."""
            while self._running:
                try:
                    # Poll for commands (non-blocking with short timeout)
                    command = self.get_joint_command(timeout_ms=10)
                    if command is not None:
                        # Command received and acknowledged by get_joint_command
                        pass
                except Exception as e:
                    if not self._running:
                        # Expected during shutdown - exit loop gracefully
                        break
                    # Unexpected exception while running - log and continue
                    import logging
                    logging.getLogger(__name__).debug(
                        f"Exception in command loop (continuing): {e}", exc_info=True
                    )

        self._publish_thread = threading.Thread(target=publish_loop, daemon=True)
        self._publish_thread.start()
        
        self._command_thread = threading.Thread(target=command_loop, daemon=True)
        self._command_thread.start()

    def stop_publishing(self):
        """Stop publishing observations."""
        self._running = False
        if self._publish_thread:
            self._publish_thread.join(timeout=1.0)
        if self._command_thread:
            self._command_thread.join(timeout=1.0)

    def publish_observation(self):
        """Set/publish synthetic hardware observation.

        Creates hardware observation with synthetic data for testing.
        """
        # Create synthetic hardware observation
        hw_obs = create_dummy_hw_obs(
            camera_height=480, camera_width=640
        )
        # Fill joint positions with test pattern
        hw_obs.joint_positions[:] = 0.1

        # Publish via base class
        super().set_observation(hw_obs)


class MockPolicyModel:
    """Mock policy model for testing."""

    def __init__(self, action_dim: int = 12, inference_time_ms: float = 5.0):
        """Initialize mock model.

        Args:
            action_dim: Action dimension
            inference_time_ms: Simulated inference time in milliseconds
        """
        self.action_dim = action_dim
        self.inference_time_ms = inference_time_ms
        self.inference_count = 0

    def inference(self, model_io):
        """Mock inference."""
        time.sleep(self.inference_time_ms / 1000.0)  # Simulate inference time
        self.inference_count += 1

        from compute.parkour.parkour_types import InferenceResponse

        # Return action tensor directly (matching inference output format)
        action = torch.zeros(self.action_dim, dtype=torch.float32)
        return InferenceResponse.create_success(
            action=action,
            timing_breakdown=[],
        )


@pytest.fixture
def hal_setup():
    """Setup HAL server and client with shared ZMQ context for testing."""
    # Use shared context for inproc connections
    server_config = HalServerConfig(
        observation_bind="inproc://test_obs",
        command_bind="inproc://test_command",
    )
    server = ProtoHalServer(server_config)
    server.initialize()
    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_obs",
        command_endpoint="inproc://test_command",
    )
    # Use shared ZMQ context from server for inproc connections
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    # Wait briefly for inproc connection to be established
    # Publish and poll to establish connection
    server.publish_observation()
    client.poll(timeout_ms=100)

    yield server, client

    # Cleanup
    server.stop_publishing()
    client.close()
    server.close()


def test_game_loop_basic_functionality(hal_setup):
    """Test basic inference logic (game loop) functionality with mock HAL server."""
    server, client = hal_setup

    # Setup mock model
    model = MockPolicyModel(action_dim=12, inference_time_ms=5.0)
    observation_dimensions = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
        KRABBY_HEX_DEFINITION
    )

    # Setup inference test runner (simulates game loop)
    test_runner = InferenceTestRunner(
        model,
        client,
        control_rate_hz=100.0,
        robot_definition=KRABBY_HEX_DEFINITION,
        observation_dimensions=observation_dimensions,
    )

    # Set navigation command on test runner (not client)
    nav_cmd = NavigationCommand.create_now()
    test_runner.set_navigation_command(nav_cmd)

    # Start publishing observations (handles threading internally)
    server.start_publishing(rate_hz=100.0)

    # Run inference test for a short time
    def stop_after_time():
        time.sleep(0.2)  # Run for 200ms
        test_runner.stop()

    stop_thread = threading.Thread(target=stop_after_time, daemon=True)
    stop_thread.start()

    test_runner.run()

    stop_thread.join(timeout=1.0)
    
    # Stop publishing before test ends to ensure threads are cleaned up
    server.stop_publishing()

    # Verify inference test ran
    assert test_runner.frame_count > 0


def test_game_loop_observation_tensor_correctness(hal_setup):
    """Test that observation tensor structure matches training format exactly.

    Verifies total obs_dim and component dimensions from definitions,
    view methods, and float32 dtype.
    """
    observation_dimensions = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
        KRABBY_HEX_DEFINITION
    )
    d = observation_dimensions

    server, client = hal_setup

    hw_obs = create_dummy_hw_obs(camera_height=480, camera_width=640)
    hw_obs.joint_positions[:] = 1.0

    server.set_observation(hw_obs)
    time.sleep(0.1)
    received_hw_obs = client.poll(timeout_ms=1000)

    assert received_hw_obs is not None, "Hardware observation should be received"
    assert received_hw_obs.joint_positions.shape == (12,), (
        f"Joint positions shape should be (12,), got {received_hw_obs.joint_positions.shape}"
    )
    assert received_hw_obs.joint_positions.dtype == np.float32, (
        f"Joint positions dtype should be float32, got {received_hw_obs.joint_positions.dtype}"
    )

    from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper

    mapper = HWObservationsToParkourMapper(observation_dimensions)
    parkour_obs = mapper.map(received_hw_obs)

    obs_arr = parkour_obs.to_array()
    assert obs_arr.shape == (d.obs_dim,), (
        f"ParkourObservation shape should be ({d.obs_dim},), got {obs_arr.shape}"
    )
    assert obs_arr.dtype == np.float32, (
        f"ParkourObservation dtype should be float32, got {obs_arr.dtype}"
    )

    prop = parkour_obs.get_proprioceptive()
    assert prop.shape == (d.num_prop,), (
        f"Proprioceptive shape should be ({d.num_prop},), got {prop.shape}"
    )
    scan = parkour_obs.get_scan()
    assert scan.shape == (d.num_scan,), (
        f"Scan shape should be ({d.num_scan},), got {scan.shape}"
    )
    vision_list = parkour_obs.get_vision()
    if d.num_vision > 0:
        assert len(vision_list) >= 1 and sum(a.size for a in vision_list) == d.num_vision, (
            f"Vision should be non-empty list with total size {d.num_vision}, got {vision_list}"
        )
    else:
        assert vision_list == [], "Vision should be empty list when num_vision is 0"
    priv_explicit = parkour_obs.get_priv_explicit()
    assert priv_explicit.shape == (d.num_priv_explicit,), (
        f"Privileged explicit shape should be ({d.num_priv_explicit},), got {priv_explicit.shape}"
    )
    priv_latent = parkour_obs.get_priv_latent()
    assert priv_latent.shape == (d.num_priv_latent,), (
        f"Privileged latent shape should be ({d.num_priv_latent},), got {priv_latent.shape}"
    )
    history = parkour_obs.get_history()
    assert history.shape == (d.history_dim,), (
        f"History shape should be ({d.history_dim},), got {history.shape}"
    )
    total_dim = (
        d.num_prop
        + d.num_scan
        + d.num_vision
        + d.num_priv_explicit
        + d.num_priv_latent
        + d.history_dim
    )
    assert total_dim == d.obs_dim, (
        f"Component dimensions sum to {total_dim}, but obs_dim is {d.obs_dim}"
    )


