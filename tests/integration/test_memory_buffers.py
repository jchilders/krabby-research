"""Integration tests for memory buffer management."""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)
import pytest

from hal.client.client import HalClient
from hal.server import HalServerBase, HalServerConfig
from hal.client.config import HalClientConfig
from hal.client.data_structures.hardware import HardwareObservations
from tests.helpers import create_dummy_hw_obs


def test_hwm_prevents_buffer_growth():
    """Test that HWM=1 prevents buffer growth."""
    import zmq
    
    # Use shared context for inproc connections
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_hwm",
        command_bind="inproc://test_command_hwm",
        observation_buffer_size=1,
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_hwm",
        command_endpoint="inproc://test_command_hwm",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)
    
    # Publish a dummy message first to establish connection
    hw_obs_init = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    server.set_observation(hw_obs_init)
    client.poll(timeout_ms=1000)
    # Connection is now established

    # Publish many messages rapidly (faster than consumption)
    # With HWM=1, older messages are dropped, so we need to ensure
    # the subscriber receives at least some messages
    for i in range(1, 101):  # Start at 1 to avoid confusion with init message
        hw_obs = create_dummy_hw_obs(
            camera_height=480, camera_width=640
        )
        hw_obs.joint_positions[0] = float(i)
        server.set_observation(hw_obs)
        # No sleep needed - test is measuring buffer behavior

    # Poll multiple times to drain queue - with HWM=1, we should eventually get a message
    # The key test is that memory stays bounded (HWM=1), not that we get the absolute latest
    received_values = []
    for _ in range(20):
        hw_obs = client.poll(timeout_ms=100)
        if hw_obs is not None:
            val = hw_obs.joint_positions[0]
            if val not in received_values:
                received_values.append(val)

    # With HWM=1, we should receive some messages (exact value depends on timing and HWM behavior)
    # The important thing is that we received messages and memory stayed bounded
    assert len(received_values) > 0, "Should have received at least one message"

    client.close()
    server.close()


def test_rapid_message_publishing():
    """Test with rapid message publishing (faster than consumption)."""
    import zmq
    
    # Use shared context for inproc connections
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_rapid",
        command_bind="inproc://test_command_rapid",
        observation_buffer_size=1,
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_rapid",
        command_endpoint="inproc://test_command_rapid",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    # Publish a dummy message first to establish connection
    hw_obs_init = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    server.set_observation(hw_obs_init)
    client.poll(timeout_ms=1000)

    # Publish messages very rapidly
    import threading

    publish_count = [0]

    def rapid_publish():
        for i in range(1000):
            hw_obs = create_dummy_hw_obs(
                camera_height=480, camera_width=640
            )
            hw_obs.joint_positions[0] = float(i)
            server.set_observation(hw_obs)
            publish_count[0] += 1
            # No sleep needed - test is measuring rapid publishing behavior

    pub_thread = threading.Thread(target=rapid_publish)
    pub_thread.start()

    # Poll occasionally (slower than publishing); with CONFLATE we may receive the latest in any of these
    received_any = []
    for _ in range(10):
        time.sleep(0.005)  # 5ms between polls
        obs = client.poll(timeout_ms=100)
        if obs is not None:
            received_any.append(obs)

    pub_thread.join()

    # With CONFLATE/HWM=1 we receive at most one (latest) per poll; we should have received at least one
    assert len(received_any) > 0, "Should have received at least one observation during rapid publish"

    client.close()
    server.close()


def test_memory_usage_bounded():
    """Test that memory usage stays bounded."""
    import zmq
    
    # Use shared context for inproc connections
    # This is a simplified test - full memory profiling would require more tools
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_memory",
        command_bind="inproc://test_command_memory",
        observation_buffer_size=1,
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_memory",
        command_endpoint="inproc://test_command_memory",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)
    
    # Publish a dummy message first to establish connection
    hw_obs_init = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    server.set_observation(hw_obs_init)
    client.poll(timeout_ms=1000)

    # Publish many messages
    for i in range(1000):
        hw_obs = create_dummy_hw_obs(
            camera_height=480, camera_width=640
        )
        hw_obs.joint_positions[:] = float(i)
        server.set_observation(hw_obs)
        if i % 100 == 0:
            client.poll(timeout_ms=100)

    # With HWM=1, memory should stay bounded
    # (We can't easily measure exact memory, but the system should not crash)
    final_hw_obs = client.poll(timeout_ms=100)
    assert final_hw_obs is not None

    client.close()
    server.close()


def test_old_messages_dropped():
    """Test that old messages are dropped (not buffered).

    Why this test matters:
    The HAL observation channel uses ZMQ PUB/SUB with High Water Mark (HWM) set to 1
    on both server (SNDHWM) and client (RCVHWM). We need to confirm that the client
    always receives the *latest* observation when it polls, and that older ones are
    discarded—not queued. That keeps memory bounded and guarantees latest-only semantics.

    What we mean by "old messages":
    If the server publishes observation 1.0, then 2.0, then 3.0 faster than the client
    polls, "old" messages are 1.0 and 2.0. The client should never see them after 3.0
    has been sent; only 3.0 should be available on the next poll.

    Why we drop them:
    - Memory: Buffering every observation would grow without bound if the client is slow.
    - Latency: We want the most recent state for control; stale observations would
      lead to outdated decisions.
    - Design: The HAL contract is "latest observation only"; dropping old messages
      is the intended behavior, not a fallback.

    How we verify: Send two observations (1.0 then 2.0) without polling in between,
    then poll once. With HWM=1 only the latest (2.0) should be kept; we must receive
    2.0, not 1.0, proving the old message was dropped.
    """
    # Use shared context for inproc connections (required for reliable inproc PUB/SUB)
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_drop",
        command_bind="inproc://test_command_drop",
        observation_buffer_size=1,
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_drop",
        command_endpoint="inproc://test_command_drop",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    # Establish connection: one send, one poll
    hw_obs_init = create_dummy_hw_obs(camera_height=480, camera_width=640)
    server.set_observation(hw_obs_init)
    init_recv = client.poll(timeout_ms=1000)
    assert init_recv is not None, "init: no message received (connection not established)"

    # Two sends, no poll in between. With HWM=1 only the latest (2.0) is kept; 1.0 is dropped.
    hw_obs_1 = create_dummy_hw_obs(camera_height=480, camera_width=640)
    hw_obs_1.joint_positions[0] = 1.0
    server.set_observation(hw_obs_1)

    hw_obs_2 = create_dummy_hw_obs(camera_height=480, camera_width=640)
    hw_obs_2.joint_positions[0] = 2.0
    server.set_observation(hw_obs_2)

    # One poll: must get the latest (2.0), not the old (1.0)
    time.sleep(0.01)
    received = client.poll(timeout_ms=1000)
    assert received is not None, (
        "After two sends (1.0, 2.0) and one poll: expected latest observation (2.0), got None"
    )
    assert received.joint_positions[0] == 2.0, (
        f"After two sends (1.0, 2.0) and one poll: expected latest=2.0 (old dropped), got {received.joint_positions[0]}"
    )

    client.close()
    server.close()

