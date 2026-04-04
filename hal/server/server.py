"""HAL server base class using ZMQ for communication."""

import logging
import time
from typing import Optional

import numpy as np
import zmq

from hal.client.data_structures.hardware import HardwareObservations, JointCommand
from hal.server.config import HalServerConfig  # Internal import - config is in same package
# OBS_DIM is not used in HAL server

logger = logging.getLogger(__name__)

# Schema version for messages

# Topics for PUB/SUB channels
TOPIC_OBSERVATION = b"observation"  # Complete observation in training format


class HalServerBase:
    """Base class for HAL server.
    
    Provides observation publishing and joint command receiving.
    Uses latest-only semantics (buffer size = 1).
    
    Note: This class uses ZMQ internally as an implementation detail.
    The ZMQ logic is black-boxed - users of this class don't need to know
    about ZMQ. If you need to switch to a different transport later,
    you can create a new implementation with the same interface.
    """

    def __init__(self, config: HalServerConfig):
        """Initialize HAL server.
        
        Server manages its own ZMQ context. For inproc connections,
        clients should use the same context (obtained via get_transport_context()).
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.context = zmq.Context()  # Server owns ZMQ context
        self.observation_socket: Optional[zmq.Socket] = None
        self.command_socket: Optional[zmq.Socket] = None
        self._initialized = False
        self._debug_enabled = False
        self._observation_send_count = 0  # for debug logging: which send we're on

    def get_transport_context(self):
        """Get transport context for inproc connections.
        
        Returns the ZMQ context that clients can use for inproc connections
        to ensure they're in the same process.
        
        Returns:
            ZMQ context for inproc connections
        """
        return self.context

    def initialize(self) -> None:
        """Initialize ZMQ context and sockets."""
        if self._initialized:
            return

        # Create PUB socket for observation (complete observation in training format)
        self.observation_socket = self.context.socket(zmq.PUB)
        self.observation_socket.setsockopt(zmq.SNDHWM, self.config.observation_buffer_size)
        self.observation_socket.bind(self.config.observation_bind)

        # Create PULL socket for commands (PUSH/PULL pattern with backpressure)
        self.command_socket = self.context.socket(zmq.PULL)
        self.command_socket.setsockopt(zmq.RCVHWM, 5)  # Default HWM of 5 for backpressure
        self.command_socket.bind(self.config.command_bind)

        self._initialized = True
        logger.info(f"HAL server initialized: observation={self.config.observation_bind}, "
                   f"command={self.config.command_bind}")

    def close(self) -> None:
        """Close all sockets and context."""
        if not self._initialized:
            return

        if self.observation_socket:
            self.observation_socket.close()
            self.observation_socket = None

        if self.command_socket:
            self.command_socket.close()
            self.command_socket = None

        if self.context:
            self.context.term()
            self.context = None

        self._initialized = False
        logger.info("HAL server closed")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug logging.

        When enabled, emits structured logs for all ZMQ messages (send/receive).
        When disabled, no debug logs are emitted to avoid overhead.

        Args:
            enabled: True to enable debug logging, False to disable
        """
        self._debug_enabled = enabled
        if enabled:
            logger.info("Debug logging enabled for HAL server")
        else:
            logger.info("Debug logging disabled for HAL server")

    def is_debug_enabled(self) -> bool:
        """Check if debug logging is enabled.

        Returns:
            True if debug logging is enabled, False otherwise
        """
        return self._debug_enabled

    def set_observation(self, hw_obs: "HardwareObservations") -> None:
        """Set/publish hardware observation to clients.

        Sends one ZMQ frame: topic prefix (b"observation") + serialized observation blob.
        Single-part allows CONFLATE on the subscriber for latest-only semantics.

        Args:
            hw_obs: HardwareObservations instance

        Raises:
            ValueError: If observation is invalid
            RuntimeError: If server not initialized
        """
        if not self._initialized:
            raise RuntimeError("Server not initialized. Call initialize() first.")

        # Runtime type validation: Validate input type
        if not isinstance(hw_obs, HardwareObservations):
            raise ValueError(f"hw_obs must be HardwareObservations, got {type(hw_obs)}")

        # Single-part message: topic prefix + payload blob (enables CONFLATE on subscriber)
        payload = hw_obs.to_bytes()
        frame = TOPIC_OBSERVATION + payload
        self._observation_send_count += 1
        send_id = self._observation_send_count

        if self._debug_enabled:
            logger.debug(
                f"[HAL server] send #{send_id}: timestamp_ns={hw_obs.timestamp_ns}, frame_len={len(frame)} (topic + blob), SNDHWM={self.observation_socket.getsockopt(zmq.SNDHWM)}",
            )
        try:
            self.observation_socket.send(frame, zmq.NOBLOCK)
        except zmq.ZMQError as e:
            if self._debug_enabled:
                logger.debug(
                    f"[HAL server] send #{send_id} raised ZMQError: errno={e.errno}, str={e}",
                )
            raise
        if self._debug_enabled:
            wall_ns = time.time_ns()
            logger.debug(
                f"[HAL server] send #{send_id} completed (message queued for delivery), timestamp_ns={hw_obs.timestamp_ns}, wall_ns={wall_ns}",
            )

    def get_joint_command(self, timeout_ms: int = 100) -> Optional["JointCommand"]:
        """Get latest joint command from clients.

        Uses non-blocking poll to check for commands. If command received,
        validates payload and returns full command instance with timestamp and metadata.

        Runtime validation includes:
        - Payload size validation (must be multiple of 4 bytes for float32)
        - Dtype validation (must be float32)
        - Shape validation (must be 1D array)
        - Value validation (no NaN or Inf)

        Args:
            timeout_ms: Poll timeout in milliseconds (default 100ms)

        Returns:
            JointCommand instance with joint positions, timestamp, and observation timestamp,
            or None if no command received or validation failed

        Raises:
            RuntimeError: If server not initialized
        """
        if not self._initialized:
            raise RuntimeError("Server not initialized. Call initialize() first.")

        # Poll for incoming command
        if self.command_socket.poll(timeout_ms, zmq.POLLIN):
            # Receive command as single-part message (blob)
            command_frame = self.command_socket.recv(zmq.NOBLOCK)

            if self._debug_enabled:
                logger.debug(
                    f"[ZMQ RECV] command: received {len(command_frame)} bytes (single blob)",
                )

            command = JointCommand.from_bytes(command_frame)

            # Debug logging after deserialization
            if self._debug_enabled:
                round_trip_latency_ns = command.timestamp_ns - command.observation_timestamp_ns
                round_trip_latency_ms = round_trip_latency_ns / 1e6
                d = command.to_positions_dict()
                vals = list(d.values())
                min_max = f"min={min(vals):.3f}, max={max(vals):.3f}, " if vals else ""
                logger.debug(
                    f"[ZMQ RECV] command: shape=({len(vals)},), dtype=float32, {min_max} "
                    f"timestamp_ns={command.timestamp_ns}, "
                    f"observation_timestamp_ns={command.observation_timestamp_ns}, "
                    f"round_trip_latency_ms={round_trip_latency_ms}"
                )

            return command

        return None

