"""Jetson HAL server implementation.

"""

import logging
import math
import time
from typing import Optional, Any

import numpy as np

from hal.server import HalServerBase, HalServerConfig
from hal.server.robot_definition import RobotDefinition
from hal.client.data_structures.hardware import HardwareObservations, JointCommand
from hal.server.jetson.rgb_depth_camera import RgbDepthCamera
from hal.server.jetson.camera import ZedCamera, create_zed_camera
from hal.server.jetson.krabby_mcusdk import KrabbyMCUSDK
from hal.server.jetson.telemetry_websocket import TelemetryWebSocketConfig, TelemetryWebSocketServer

from compute.parkour.model_definition import ObservationDimensions

logger = logging.getLogger(__name__)
TELEMETRY_STALE_TIMEOUT_S = 1.0
FAKE_TELEMETRY_JOINTS = (
    "FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL",
    "RLHY", "RLHL", "RLKL", "MLHY", "MLHL", "MLKL",
    "RRHY", "RRHL", "RRKL", "MRHY", "MRHL", "MRKL",
)


class JetsonHalServer(HalServerBase):
    """HAL server for Jetson robot deployment.

    Integrates with a front camera (e.g. ZED) and real sensors to publish observations.
    Applies joint commands to real actuators.
    """

    def __init__(
        self,
        config: HalServerConfig,
        observation_dimensions: ObservationDimensions,
        action_dim: int,
        robot_definition: RobotDefinition,
        camera_resolution: tuple[int, int] = (640, 480),
        camera_fps: int = 30,
        mcu_port: Optional[str] = None,
        mcu_baud: int = 115200,
        mcu_auto_connect: bool = True,
        telemetry_ws_config: Optional[TelemetryWebSocketConfig] = None,
    ):
        """Initialize Jetson HAL server.

        Args:
            config: HAL server configuration
            observation_dimensions: Layout from model_definition.get_observation_dimensions(robot_definition)
            action_dim: Action dimension (model output joint count)
            robot_definition: Robot definition for joint names and order (adapter uses get_joint_names()).
            camera_resolution: Front camera resolution (width, height)
            camera_fps: Front camera FPS
            mcu_port: Serial port for MCU connection. If None, uses default from MCU SDK.
            mcu_baud: Baud rate for MCU serial communication.
            mcu_auto_connect: If True, automatically connect to MCU on initialization.
            telemetry_ws_config: Optional websocket telemetry configuration.
        """
        super().__init__(config)
        self.observation_dimensions = observation_dimensions
        self.action_dim = action_dim
        self.robot_definition = robot_definition
        self.camera_resolution = camera_resolution
        self.camera_fps = camera_fps
        self.front_camera: Optional[RgbDepthCamera] = None
        self.state_source = None  # IMU/encoders (placeholder, real implementation in future)
        self.actuator_sink = None  # Motors (placeholder, real implementation in future)
        self._last_joint_positions: Optional[dict[str, float]] = None  # from command.to_positions_dict()
        self._telemetry_ws_config = telemetry_ws_config
        self._telemetry_ws_server: Optional[TelemetryWebSocketServer] = None
        self._fake_telemetry_started_s = time.time()
        
        # Initialize Krabby MCU SDK when robot definition has MCU joints
        self._mcusdk: Optional[KrabbyMCUSDK] = None
        mcu_joints = self.robot_definition.get_mcu_joints()
        if mcu_joints:
            try:
                self._mcusdk = KrabbyMCUSDK(
                    mcu_joints=mcu_joints,
                    port=mcu_port,
                    baud=mcu_baud,
                    auto_connect=mcu_auto_connect,
                )
                logger.info("KrabbyMCUSDK initialized for JetsonHalServer")
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(
                    f"KrabbyMCUSDK not available: {e}. "
                    "MCU commands will not be sent. Install firmware package and use a robot definition with mcu_joints."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize KrabbyMCUSDK: {e}. "
                    "MCU commands will not be sent. Check MCU connection and configuration.",
                    exc_info=True,
                )

    def initialize(self) -> None:
        """Initialize transport sockets and optional telemetry websocket server."""
        super().initialize()
        self.start_telemetry_ws()

    def _build_telemetry_payload(self) -> dict[str, Any]:
        if self._telemetry_ws_config is not None and self._telemetry_ws_config.fake_data:
            return self._build_fake_telemetry_payload()

        now_s = time.time()
        now_ns = time.time_ns()
        telemetry: dict[str, Any] = {
            "type": "joint_telemetry",
            "timestamp_ns": now_ns,
            "status": "disconnected",
            "source": "mcu",
            "joints": {},
        }

        if self._mcusdk is None:
            return telemetry

        snapshot = self._mcusdk.get_joint_telemetry_snapshot()
        last_feedback_ts = snapshot.get("last_feedback_ts")

        is_fresh = (
            isinstance(last_feedback_ts, (float, int))
            and (now_s - float(last_feedback_ts)) <= TELEMETRY_STALE_TIMEOUT_S
        )
        is_connected = bool(snapshot.get("connected")) and is_fresh

        if is_connected and isinstance(last_feedback_ts, (float, int)):
            telemetry["timestamp_ns"] = int(float(last_feedback_ts) * 1e9)
            joints = snapshot.get("joints", {})
            telemetry["joints"] = {
                name: joints[name] for name in sorted(joints.keys())
            }
            telemetry["status"] = "connected"

        return telemetry

    def _build_fake_telemetry_payload(self) -> dict[str, Any]:
        now_s = time.time()
        elapsed_s = now_s - self._fake_telemetry_started_s
        joints: dict[str, dict[str, Any]] = {}

        for idx, joint_name in enumerate(FAKE_TELEMETRY_JOINTS):
            phase = elapsed_s * 1.1 + idx * 0.31
            pos = 0.5 + 0.25 * math.sin(phase)
            pot = int(round(512 + 240 * math.sin(phase + 0.7)))
            current = int(round(580 + 120 * abs(math.sin(phase * 1.3 + 0.2))))
            drive = int(round(42 * math.sin(phase + 0.4)))
            pwm = [max(0, drive), max(0, -drive)]

            joints[joint_name] = {
                "pos": round(max(0.0, min(1.0, pos)), 3),
                "pot": max(0, min(1023, pot)),
                "current": max(0, current),
                "en": [1, 1],
                "pwm": pwm,
                "saf": 0,
            }

        return {
            "type": "joint_telemetry",
            "timestamp_ns": time.time_ns(),
            "status": "connected",
            "source": "fake",
            "joints": joints,
        }

    def start_telemetry_ws(self) -> None:
        """Start telemetry websocket server when enabled."""
        if self._telemetry_ws_config is None or not self._telemetry_ws_config.enabled:
            return
        if self._telemetry_ws_server is not None:
            return

        self._telemetry_ws_server = TelemetryWebSocketServer(
            self._telemetry_ws_config,
            snapshot_provider=self._build_telemetry_payload,
        )
        self._telemetry_ws_server.start()

    def stop_telemetry_ws(self) -> None:
        """Stop telemetry websocket server if running."""
        if self._telemetry_ws_server is None:
            return
        self._telemetry_ws_server.stop()
        self._telemetry_ws_server = None

    def initialize_camera(self) -> None:
        """Initialize front camera (RGB + depth + scan features).

        Uses the default camera implementation (e.g. ZED). Other cameras can be
        supported by providing a compatible interface and factory.
        """
        logger.info("Initializing front camera...")
        self.front_camera = create_zed_camera(
            resolution=self.camera_resolution,
            fps=self.camera_fps,
            depth_mode="PERFORMANCE",
            depth_feature_dim=self.observation_dimensions.num_scan,
        )

        if self.front_camera is None:
            logger.error("Failed to initialize front camera")
            raise RuntimeError("Front camera initialization failed")
        elif not self.front_camera.is_ready():
            logger.error("Front camera initialized but not ready")
            raise RuntimeError("Front camera is not ready")
        else:
            logger.info("Front camera initialized successfully")

    def initialize_sensors(self) -> None:
        """Initialize state sensors (IMU/encoders).

        **WARNING**: This is currently a placeholder implementation.
        Real sensor data (IMU, encoders) is required for production deployment.
        Placeholder data (zeros) will cause incorrect policy behavior and is unsafe.

        TODO: Implement real sensor initialization:
        - Initialize IMU driver
        - Initialize encoder drivers  
        - Configure sensor parameters
        - Set self.state_source to real sensor interface
        """
        # Placeholder - actual implementation needs sensor drivers
        logger.warning(
            "⚠️  PLACEHOLDER MODE: Sensors not initialized. "
            "Using placeholder data (zeros) for base pose, velocities, and joint velocities. "
            "This will cause INCORRECT POLICY BEHAVIOR and is UNSAFE for production. "
            "Implement real sensor drivers before deployment."
        )

    def initialize_actuators(self) -> None:
        """Initialize actuators (motors).

        This is a placeholder. Real implementation would:
        - Initialize motor controllers
        - Configure motor parameters
        - Enable motors
        """
        # Placeholder - actual implementation needs motor drivers
        logger.info("Actuator initialization (placeholder)")

    def _build_depth_features(self) -> np.ndarray:
        """Build depth features from front camera.

        Returns:
            Depth features as float32 array

        Raises:
            RuntimeError: If camera is not initialized or depth features cannot be obtained
        """
        if self.front_camera is None:
            raise RuntimeError(
                "Front camera not initialized. "
                "Camera must be initialized via initialize_camera() before building depth features."
            )

        # Get depth features (includes capture, validation, and feature extraction)
        depth_features = self.front_camera.get_depth_features()
        if depth_features is None:
            raise RuntimeError(
                "Failed to get depth features from front camera. "
                "Depth features are required for policy operation. "
                "Check camera connection and ensure frames are being captured."
            )

        # Return features array
        return depth_features

    def _build_state_vector(self) -> Optional[np.ndarray]:
        """Build state vector from sensors.

        **WARNING**: Currently uses placeholder data for base pose, velocities, and joint velocities.
        Real sensor data is required for production deployment.

        Format: base_pos(3), base_quat(4), base_lin_vel(3), base_ang_vel(3),
                joint_pos(ACTION_DIM), joint_vel(ACTION_DIM)
        
        Total length: 13 + 2*ACTION_DIM (13 base state + joint positions + joint velocities)

        Returns:
            State vector as float32 array

        TODO: When real sensors are implemented, extract:
        - Base pose from IMU/odometry (if state_source is available)
        - Base velocities from IMU
        - Joint positions from encoders
        - Joint velocities from encoder derivatives
        """
        # If we have real state source, use it
        if self.state_source is not None:
            # TODO: Implement real sensor data extraction from IMU/encoders
            # Real implementation would:
            # - Get base pose from IMU/odometry
            # - Get base velocities from IMU
            # - Get joint positions from encoders
            # - Get joint velocities from encoder derivatives
            # For now, fall through to placeholder
            logger.warning(
                "state_source is set but real sensor data extraction not implemented. "
                "Using placeholder data."
            )

        # Placeholder implementation: identity pose + echo joint positions
        # ⚠️ WARNING: This uses zeros for base pose, velocities, and joint velocities.
        # This will cause INCORRECT POLICY BEHAVIOR. Real sensors are required.
        
        # Base position: (0, 0, 0) - identity position (PLACEHOLDER - requires IMU/odometry)
        base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Base quaternion: (0, 0, 0, 1) - identity quaternion (PLACEHOLDER - requires IMU)
        base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Base velocities: (0, 0, 0) - zero velocities (PLACEHOLDER - requires IMU)
        base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        base_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Joint positions: echo last commanded targets (best available data)
        # This is the only "real" data we have (from commands we sent)
        if self._last_joint_positions is not None:
            names = self.robot_definition.get_joint_names()
            joint_pos = np.array(
                [self._last_joint_positions.get(n, 0.0) for n in names],
                dtype=np.float32,
            )
        else:
            joint_pos = np.zeros(self.action_dim, dtype=np.float32)

        # Joint velocities: (0, 0, ...) - zero velocities (PLACEHOLDER - requires encoders)
        joint_vel = np.zeros(self.action_dim, dtype=np.float32)

        # Concatenate into state vector
        state_vector = np.concatenate([
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            joint_pos,
            joint_vel,
        ]).astype(np.float32)

        return state_vector

    def set_observation(self) -> None:
        """Set observation from real sensors as hardware observations.
        
        Constructs HardwareObservations from raw sensor data.
        Extracts:
        - Joint positions from state vector (echoed from last command)
        - Depth maps from front camera (if available)
        - RGB images from front camera (if available)
        
        Base class will loop until observation is published or throw if client isn't consuming.
        
        Note: Camera initialization is optional for basic infrastructure testing.
        If camera is not initialized, placeholder values (zeros) will be used for
        depth and RGB images. WARNING: These placeholder values are NOT suitable
        for actual policy inference - the model requires real depth/scan features.
        For production deployment, camera MUST be initialized.
        """
        # Build state vector (required for publishing observations)
        state_vector = self._build_state_vector()
        if state_vector is None:
            raise RuntimeError(
                "Failed to build state vector. "
                "State vector is required for publishing observations to the inference client."
            )

        # Extract components from state vector
        # Format: base_pos(3), base_quat(4), base_lin_vel(3), base_ang_vel(3),
        #         joint_pos(ACTION_DIM), joint_vel(ACTION_DIM)
        # Total: 3 + 4 + 3 + 3 = 13 before joint positions
        base_pos = state_vector[0:3]
        base_quat = state_vector[3:7]  # (x, y, z, w) format
        base_lin_vel = state_vector[7:10]
        base_ang_vel = state_vector[10:13]
        joint_pos = state_vector[13:13+self.action_dim]
        joint_vel = state_vector[13+self.action_dim:13+2*self.action_dim]
        
        # Observation joint count from robot definition (e.g. 12 quad, 18 hex)
        obs_joint_count = self.robot_definition.get_total_joint_count()
        joint_positions = np.zeros(obs_joint_count, dtype=np.float32)
        num_joints = min(len(joint_pos), obs_joint_count)
        joint_positions[:num_joints] = joint_pos[:num_joints].astype(np.float32)
        
        joint_velocities = np.zeros(obs_joint_count, dtype=np.float32)
        num_joints_vel = min(len(joint_vel), obs_joint_count)
        joint_velocities[:num_joints_vel] = joint_vel[:num_joints_vel].astype(np.float32)
        
        # Extract base velocities (body frame) - for now use world frame values as placeholder
        # TODO: Transform to body frame when real IMU data is available
        base_ang_vel_b = base_ang_vel.astype(np.float32)
        base_lin_vel_b = base_lin_vel.astype(np.float32)
        
        # Base quaternion (world frame, x,y,z,w format)
        base_quat_w = base_quat.astype(np.float32)
        
        # Contact forces (placeholder - 5 values, normalized to [-0.5, 0.5])
        contact_forces = np.zeros(5, dtype=np.float32)
        
        # Previous action (from last command or zeros if none)
        previous_action = np.zeros(obs_joint_count, dtype=np.float32)
        if self._last_joint_positions is not None:
            names = self.robot_definition.get_joint_names()
            num_prev = min(len(names), obs_joint_count)
            for i in range(num_prev):
                previous_action[i] = self._last_joint_positions.get(names[i], 0.0)

        # Get camera data and scan features from ZED when available
        camera_height, camera_width = self.camera_resolution[1], self.camera_resolution[0]
        camera_rgb: Optional[np.ndarray] = None
        camera_depth: Optional[np.ndarray] = None
        scan_features: Optional[np.ndarray] = None

        if self.front_camera is not None:
            rgb_frame, depth_frame = self.front_camera.get_camera_frames()
            if rgb_frame is None or depth_frame is None:
                logger.warning(
                    f"Front camera get_camera_frames() returned None (rgb={rgb_frame is not None}, depth={depth_frame is not None}); publishing observation without camera_rgb/camera_depth",
                )
            if rgb_frame is not None and depth_frame is not None:
                expected_rgb_shape = (camera_height, camera_width, 3)
                expected_depth_shape = (camera_height, camera_width)
                if rgb_frame.shape != expected_rgb_shape:
                    raise RuntimeError(
                        f"ZED RGB frame shape {rgb_frame.shape} does not match configured resolution "
                        f"{expected_rgb_shape}. Set camera_resolution to (width, height) matching the ZED output."
                    )
                if depth_frame.shape != expected_depth_shape:
                    raise RuntimeError(
                        f"ZED depth frame shape {depth_frame.shape} does not match configured resolution "
                        f"{expected_depth_shape}. Set camera_resolution to (width, height) matching the ZED output."
                    )
                camera_rgb = np.asarray(rgb_frame, dtype=np.uint8)
                camera_depth = depth_frame.astype(np.float32)
            # Scan features from ZED (132-dim height-like features for policy; separate grab)
            scan_features = self.front_camera.get_depth_features()
            if scan_features is not None:
                scan_features = np.asarray(scan_features, dtype=np.float32)

        hw_obs = HardwareObservations(
            joint_positions=joint_positions,
            camera_height=camera_height,
            camera_width=camera_width,
            timestamp_ns=time.time_ns(),
            base_ang_vel_b=base_ang_vel_b,
            base_lin_vel_b=base_lin_vel_b,
            base_quat_w=base_quat_w,
            joint_velocities=joint_velocities,
            contact_forces=contact_forces,
            previous_action=previous_action,
            camera_rgb=camera_rgb,
            camera_depth=camera_depth,
            scan_features=scan_features,
        )

        # Publish hardware observation via base-class publisher
        super().set_observation(hw_obs)

    def apply_command(self, command: JointCommand) -> None:
        """Apply joint command to actuators.
        
        **Synchronous method** that applies commands **immediately** (no queuing).
        Applies the provided command directly to the robot actuators via MCU SDK.
        Does not perform any background work to keep the robot moving - the main loop
        must call this method regularly at the target control rate (typically 100 Hz).
        
        The robot continues moving based on the last applied command until the next
        command is received. If this method is not called regularly, the robot will
        stop moving after the last command's effect completes.
        
        Args:
            command: JointCommand instance to apply
        
        Raises:
            RuntimeError: If command is invalid (empty or None)
        """
        if command is None:
            raise RuntimeError("Command cannot be None")

        # Use command's dict view for validation and state echo
        cmd_dict = command.to_positions_dict()
        if len(cmd_dict) == 0:
            raise RuntimeError("Received empty joint command")

        joint_names = self.robot_definition.get_joint_names()
        if len(cmd_dict) != len(joint_names):
            raise RuntimeError(
                f"Joint command length {len(cmd_dict)} does not match robot definition joint count {len(joint_names)}"
            )

        # Store command for state echo (echo joint state from last commanded targets).
        # This allows set_observation() to echo back the commanded positions as current state.
        # TODO: In _build_state_vector()/set_observation(), read joint positions from encoders
        # instead of _last_joint_positions; echoing command means observed state can lead actual
        # pose and policy never sees tracking error.
        self._last_joint_positions = cmd_dict

        # Apply command via MCU SDK if available (SDK uses command.to_positions_dict())
        if self._mcusdk is not None:
            try:
                self._mcusdk.apply_command(command)
            except Exception as e:
                logger.error(
                    f"Error applying command via MCU SDK: {e}. "
                    "Command logged but not sent to MCU.",
                    exc_info=True
                )
                raise e
        else:
            # MCU SDK not available - log command for debugging
            logger.debug(
                f"[JOINT COMMAND] MCU SDK not available. Command not sent. "
                f"timestamp={command.timestamp_ns}, joint_pos={list(cmd_dict.values())}"
            )

    def close(self) -> None:
        """Close camera, MCU connection, and all server resources."""
        self.stop_telemetry_ws()

        # Close MCU SDK first
        if self._mcusdk is not None:
            self._mcusdk.close()
            self._mcusdk = None
        
        # Close camera
        if self.front_camera is not None:
            self.front_camera.close()
            self.front_camera = None

        # Close server resources (sockets, context)
        super().close()
