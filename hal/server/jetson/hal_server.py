"""Jetson HAL server implementation.

"""

import logging
import time
from typing import Optional

import numpy as np
from scipy.ndimage import zoom

from hal.server import HalServerBase, HalServerConfig
from hal.client.data_structures.hardware import HardwareObservations, JointCommand
from hal.server.jetson.camera import ZedCamera, create_zed_camera
from hal.server.jetson.krabby_mcusdk import KrabbyMCUSDK

from compute.parkour.model_definition import ObservationDimensions

logger = logging.getLogger(__name__)


class JetsonHalServer(HalServerBase):
    """HAL server for Jetson robot deployment.

    Integrates with ZED camera and real sensors to publish observations.
    Applies joint commands to real actuators.
    """

    def __init__(
        self,
        config: HalServerConfig,
        observation_dimensions: ObservationDimensions,
        action_dim: int,
        camera_resolution: tuple[int, int] = (640, 480),
        camera_fps: int = 30,
        mcu_port: Optional[str] = None,
        mcu_baud: int = 115200,
        mcu_auto_connect: bool = True,
    ):
        """Initialize Jetson HAL server.

        Args:
            config: HAL server configuration
            observation_dimensions: Layout from model_definition.get_observation_dimensions(robot_definition)
            action_dim: Action dimension (model output joint count)
            camera_resolution: ZED camera resolution (width, height)
            camera_fps: ZED camera FPS
            mcu_port: Serial port for MCU connection. If None, uses default from MCU SDK.
            mcu_baud: Baud rate for MCU serial communication.
            mcu_auto_connect: If True, automatically connect to MCU on initialization.
        """
        super().__init__(config)
        self.observation_dimensions = observation_dimensions
        self.action_dim = action_dim
        self.camera_resolution = camera_resolution
        self.camera_fps = camera_fps
        self.zed_camera: Optional[ZedCamera] = None
        self.state_source = None  # IMU/encoders (placeholder, real implementation in future)
        self.actuator_sink = None  # Motors (placeholder, real implementation in future)
        self._last_joint_positions: Optional[np.ndarray] = None
        
        # Initialize Krabby MCU SDK for standardized command application
        self._mcusdk: Optional[KrabbyMCUSDK] = None
        try:
            self._mcusdk = KrabbyMCUSDK(
                port=mcu_port,
                baud=mcu_baud,
                auto_connect=mcu_auto_connect,
            )
            logger.info("KrabbyMCUSDK initialized for JetsonHalServer")
        except (ImportError, RuntimeError) as e:
            logger.warning(
                f"KrabbyMCUSDK not available: {e}. "
                "MCU commands will not be sent. Install firmware package to enable MCU control."
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize KrabbyMCUSDK: {e}. "
                "MCU commands will not be sent. Check MCU connection and configuration.",
                exc_info=True
            )

    def initialize_camera(self) -> None:
        """Initialize ZED camera.

        Creates ZED camera wrapper with error handling.
        """
        logger.info("Initializing ZED camera...")
        self.zed_camera = create_zed_camera(
            resolution=self.camera_resolution,
            fps=self.camera_fps,
            depth_mode="PERFORMANCE",
            depth_feature_dim=self.observation_dimensions.num_scan,
        )

        if self.zed_camera is None:
            logger.error("Failed to initialize ZED camera")
            raise RuntimeError("ZED camera initialization failed")
        elif not self.zed_camera.is_ready():
            logger.error("ZED camera initialized but not ready")
            raise RuntimeError("ZED camera is not ready")
        else:
            logger.info("ZED camera initialized successfully")

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
        """Build depth features from ZED camera.

        Returns:
            Depth features as float32 array

        Raises:
            RuntimeError: If camera is not initialized or depth features cannot be obtained
        """
        if self.zed_camera is None:
            raise RuntimeError(
                "ZED camera not initialized. "
                "Camera must be initialized via initialize_camera() before building depth features."
            )

        # Get depth features (includes capture, validation, and feature extraction)
        depth_features = self.zed_camera.get_depth_features()
        if depth_features is None:
            raise RuntimeError(
                "Failed to get depth features from ZED camera. "
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
            joint_pos = self._last_joint_positions.astype(np.float32)
        else:
            # If no commands received yet, use zeros
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
        - Depth maps from ZED camera (if available)
        - RGB images if available from ZED camera (if available)
        
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
        
        # Pad or truncate joint positions to 12 DOF (Krabby has 12 joints)
        # HardwareObservations expects exactly 12 joint positions
        joint_positions = np.zeros(12, dtype=np.float32)
        num_joints = min(len(joint_pos), 12)
        joint_positions[:num_joints] = joint_pos[:num_joints].astype(np.float32)
        
        # Pad or truncate joint velocities to 12 DOF
        joint_velocities = np.zeros(12, dtype=np.float32)
        num_joints_vel = min(len(joint_vel), 12)
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
        previous_action = np.zeros(12, dtype=np.float32)
        if self._last_joint_positions is not None:
            num_prev = min(len(self._last_joint_positions), 12)
            previous_action[:num_prev] = self._last_joint_positions[:num_prev].astype(np.float32)

        # Get camera data from ZED camera
        # ZED camera provides depth features for model input, but we also need
        # raw depth maps and RGB images for HardwareObservations
        camera_height, camera_width = self.camera_resolution[1], self.camera_resolution[0]
        
        # Initialize with placeholder values
        rgb_camera_1 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        rgb_camera_2 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        depth_map = np.zeros((camera_height, camera_width), dtype=np.float32)
        confidence_map = np.ones((camera_height, camera_width), dtype=np.float32)
        
        if self.zed_camera is not None:
            # Try to get raw depth map if available
            if hasattr(self.zed_camera, 'get_depth_map'):
                depth_map_data = self.zed_camera.get_depth_map()
                if depth_map_data is not None:
                    # Ensure it's a numpy array
                    if not isinstance(depth_map_data, np.ndarray):
                        depth_map_data = np.array(depth_map_data)
                    
                    # Resize if needed to match camera resolution
                    if depth_map_data.shape != (camera_height, camera_width):
                        zoom_factors = (camera_height / depth_map_data.shape[0], 
                                      camera_width / depth_map_data.shape[1])
                        depth_map = zoom(depth_map_data, zoom_factors, order=1).astype(np.float32)
                    else:
                        depth_map = depth_map_data.astype(np.float32)
            
            # Try to get RGB images if available
            if hasattr(self.zed_camera, 'get_rgb_images'):
                rgb_images = self.zed_camera.get_rgb_images()
                if rgb_images is not None and len(rgb_images) >= 2:
                    # Ensure RGB images are uint8
                    rgb_camera_1 = np.array(rgb_images[0], dtype=np.uint8)
                    rgb_camera_2 = np.array(rgb_images[1], dtype=np.uint8)
                    
                    # Resize if needed
                    if rgb_camera_1.shape[:2] != (camera_height, camera_width):
                        zoom_factors = (camera_height / rgb_camera_1.shape[0], 
                                      camera_width / rgb_camera_1.shape[1], 1)
                        rgb_camera_1 = zoom(rgb_camera_1, zoom_factors, order=1).astype(np.uint8)
                    if rgb_camera_2.shape[:2] != (camera_height, camera_width):
                        zoom_factors = (camera_height / rgb_camera_2.shape[0], 
                                      camera_width / rgb_camera_2.shape[1], 1)
                        rgb_camera_2 = zoom(rgb_camera_2, zoom_factors, order=1).astype(np.uint8)

        # Create hardware observation
        hw_obs = HardwareObservations(
            joint_positions=joint_positions,
            rgb_camera_1=rgb_camera_1,
            rgb_camera_2=rgb_camera_2,
            depth_map=depth_map,
            confidence_map=confidence_map,
            camera_height=camera_height,
            camera_width=camera_width,
            timestamp_ns=time.time_ns(),
            base_ang_vel_b=base_ang_vel_b,
            base_lin_vel_b=base_lin_vel_b,
            base_quat_w=base_quat_w,
            joint_velocities=joint_velocities,
            contact_forces=contact_forces,
            previous_action=previous_action,
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

        # Extract joint positions array from command
        joint_positions = command.joint_positions
        
        # Validate command shape
        if len(joint_positions) == 0:
            raise RuntimeError("Received empty joint command")
        
        # Store command for state echo (echo joint state from last commanded targets)
        # This allows set_observation() to echo back the commanded positions as current state
        self._last_joint_positions = joint_positions.copy()

        # Apply command via MCU SDK if available
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
                f"timestamp={command.timestamp_ns}, "
                f"joint_pos={joint_positions.tolist()}, "
                f"shape={joint_positions.shape}, "
                f"dtype={joint_positions.dtype}"
            )

    def close(self) -> None:
        """Close camera, MCU connection, and all server resources."""
        # Close MCU SDK first
        if self._mcusdk is not None:
            self._mcusdk.close()
            self._mcusdk = None
        
        # Close camera
        if self.zed_camera is not None:
            self.zed_camera.close()
            self.zed_camera = None

        # Close server resources (sockets, context)
        super().close()

