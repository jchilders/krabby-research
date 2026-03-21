"""IsaacSim HAL server implementation."""

import logging
import time
from typing import Optional

from hal.server.robot_definition import RobotDefinition

import numpy as np
import torch
from scipy.ndimage import zoom

from hal.server import HalServerBase, HalServerConfig
from hal.client.data_structures.hardware import HardwareObservations, JointCommand
from hal.server.isaac.isaacsim_mcusdk import IsaacSimMCUSDK
from hal.server.isaac.sensor_backend_isaac import IsaacSensorInterface
from hal.server.sensor_interface import SensorInterface

logger = logging.getLogger(__name__)


class IsaacSimHalServer(HalServerBase):
    """HAL server for IsaacSim environment.
    
    Extracts observations from IsaacSim environment and publishes via HAL.
    Applies joint commands received via HAL to the environment.
    """

    def __init__(
        self,
        config: HalServerConfig,
        robot_definition: RobotDefinition,
        env=None,
        observation_dimensions=None,
        *,
        command_timeout_s: Optional[float] = 1.0,
    ):
        """Initialize IsaacSim HAL server.
        
        Args:
            config: HAL server configuration
            robot_definition: Robot definition (e.g. quad 12 joints). Used for command slice and observation sizes. Required.
            env: IsaacSim environment. If provided, environment component
                references will be cached via _cache_references().
            observation_dimensions: Observation dimensions from model definition. Used for history buffer calculations.
            command_timeout_s: Max seconds to wait for a joint command in apply_command(). If None, wait forever
                and log every 30s while waiting. Used in joystick mode so the server does not exit when no client is connected.
        
        Note:
            If env is provided, this will call _cache_references() to extract
            and cache references to scene, robot, sensors, and managers.
        """
        super().__init__(config)
        self.env = env
        self.robot_definition = robot_definition
        self.observation_dimensions = observation_dimensions
        self.scene = None
        self.robot = None
        self.observation_manager = None
        self.action_manager = None
        self.contact_sensor = None
        self.command_manager = None
        # Cache latest observation for logging (computed in set_observation)
        self._latest_obs_dict = None
        self._latest_obs_tensor = None
        # Track last applied action to use as previous_action in next observation
        action_dim = robot_definition.get_total_joint_count()
        self._last_applied_action = np.zeros(action_dim, dtype=np.float32)
        # Track last published observation to detect duplicates
        self._last_published_obs_vals = None
        
        # Initialize IsaacSim MCU SDK for standardized command application
        # Device will be set when environment is available
        self._mcusdk: Optional[IsaacSimMCUSDK] = None
        # GStreamer multi-sensor interface (synthetic sensors, same API as Jetson)
        self._sensor_interface: Optional[SensorInterface] = None

        self._command_timeout_s = command_timeout_s
        self._first_command_received = False
        self._waiting_log_last_t: Optional[float] = None

        if env is not None:
            self._cache_references()
            self._initialize_mcusdk()

    def get_sensor_interface(self) -> SensorInterface:
        """Return the GStreamer multi-sensor interface (list_sensors, get_gstreamer_handle, build_pipeline)."""
        if self._sensor_interface is None:
            self._sensor_interface = IsaacSensorInterface(
                scene_sensors=getattr(self, "camera_sensors", None) or {},
            )
        return self._sensor_interface

    def _cache_references(self) -> None:
        """Cache references to environment components for efficient access.
        
        This private method exists to keep __init__ clean and readable by extracting
        the complex initialization logic for environment component references into
        a separate method. The method handles:
        - Finding and caching scene, robot, and camera sensor references
        - Caching observation and action managers
        - Error handling and validation of required components
        
        This separation improves code maintainability since the caching logic is
        substantial (~60 lines) and includes error handling that would otherwise
        clutter __init__.
        
        **Important**: This method is only called from __init__ during server
        initialization. All initialization logic for environment references should
        remain here, not be mixed into __init__.
        
        Raises:
            RuntimeError: If robot entity cannot be found in scene
        """
        if self.env is None:
            return

        # Cache scene reference (use unwrapped to access underlying environment)
        # gym.make() wraps the environment, so we need to access the unwrapped version
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        self.scene = unwrapped_env.scene

        # Cache robot reference
        # The scene entity names come from the scene config class attributes.
        # Source of truth: parkour_tasks/parkour_tasks/default_cfg.py (ParkourDefaultSceneCfg)
        # Entity name "robot" is defined by the scene config attribute name.
        self.robot = self.scene["robot"]

        # Cache camera sensors if available (depth and RGB for ZED 2i-style front camera)
        # Cameras are accessed via scene.sensors dictionary, which supports .items() iteration
        self.camera_sensors = {}
        if hasattr(self.scene, 'sensors'):
            for sensor_name, sensor in self.scene.sensors.items():
                if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
                    out = sensor.data.output
                    has_depth = 'distance_to_camera' in out or 'distance_to_image_plane' in out
                    has_rgb = 'rgb' in out
                    if has_depth or has_rgb:
                        self.camera_sensors[sensor_name] = sensor

        # Cache managers (use unwrapped to access underlying environment)
        self.observation_manager = unwrapped_env.observation_manager
        self.action_manager = unwrapped_env.action_manager
        if hasattr(unwrapped_env, 'command_manager'):
            self.command_manager = unwrapped_env.command_manager
        
        # Cache contact sensor if available
        if hasattr(self.scene, 'sensors') and 'contact_forces' in self.scene.sensors:
            self.contacvt_sensor = self.scene.sensors['contact_forces']

        # Verify we have required references
        if self.robot is None:
            raise RuntimeError(
                "Robot not found in scene. "
                "IsaacSim HAL server requires a robot entity in the scene. "
                f"Available entities: {list(self.scene.keys()) if self.scene else 'None'}"
            )
        if self.observation_manager is None:
            raise RuntimeError(
                "Observation manager not available. "
                "IsaacSim HAL server requires observation_manager to extract observations."
            )
        if self.action_manager is None:
            raise RuntimeError(
                "Action manager not available. "
                "IsaacSim HAL server requires action_manager to apply actions."
            )

        logger.info("Cached environment references successfully")
    
    def _initialize_mcusdk(self) -> None:
        """Initialize IsaacSim MCU SDK.
        
        This method initializes the SDK after environment references are cached.
        """
        if self.env is None:
            return
        
        # Initialize SDK (takes JointCommand; order/timestamps from command)
        self._mcusdk = IsaacSimMCUSDK()
        logger.info("Initialized IsaacSimMCUSDK")

    def set_observation(self) -> None:
        """Set observation from IsaacSim environment as hardware observations.
        
        This method is called to publish observations. It will loop until a valid
        observation is found (not all-zero, not duplicate) or raise RuntimeError.
        
        Extracts raw sensor data from environment and constructs HardwareObservations.
        Extracts:
        - Joint positions from robot entity
        - Depth maps from camera sensors
        - RGB images if available from camera sensors or render products
        
        Rejects all-zero or duplicate observations by returning early (no exception).
        Base class will loop until observation is published or throw if client isn't consuming.
        """
        if self.env is None:
            raise RuntimeError("No environment set, cannot set observation")

        if self.robot is None:
            raise RuntimeError("Robot not available, cannot set observation")

        # Loop until we get a valid observation (not all-zero, not duplicate)
        max_attempts = 100  # Prevent infinite loop
        retry_delay_s = 0.01  # 10ms delay between attempts
        valid_observation_found = False
        attempt = 0
        
        while attempt < max_attempts and not valid_observation_found:
            # Use cached observation if available (from env.step())
            # Otherwise, extract from observation manager's computed observation (fallback for initial observation)
            if self._latest_obs_tensor is not None and self._latest_obs_dict is not None:
                # Use observation directly from env.step() to ensure we use the exact observation
                # that the environment computed with the correct previous_action from action_history_buf
                obs_tensor = self._latest_obs_tensor
                obs_dict = self._latest_obs_dict
            else:
                # Fallback: extract from observation manager (only used for initial observation before first step)
                # In Isaac Sim, observation_manager is always available
                obs_dict = self.observation_manager.compute()
                obs_tensor = obs_dict["policy"]
                
                # Cache observation for logging (avoid recomputing)
                self._latest_obs_dict = obs_dict
                self._latest_obs_tensor = obs_tensor
            
            # NOTE: We do NOT modify the observation tensor here.
            # We use what the environment returns directly. If there are differences in the history buffer,
            # we need to understand WHY the environment is returning different values, not patch them here.
            # The observation code zeros obs_buf[:, 6:8] AFTER building the observation,
            # so the returned observation includes the old history buffer. At timestep 0,
            # the history buffer should be zeros from reset. If it's not, there's a deeper issue.
            
            # Convert to numpy for extraction (after modifying tensor)
            if obs_tensor.ndim == 2:
                obs_vals = obs_tensor[0].cpu().numpy()  # Take first environment
            else:
                obs_vals = obs_tensor.cpu().numpy()
            
            # Check 1: Reject all-zero observations (before any modifications)
            # Skip this check for the very first observation (when _last_published_obs_vals is None)
            # as environments may return all-zero observations before the first step
            is_all_zero = np.allclose(obs_vals, 0.0, atol=1e-6)
            is_first_observation = self._last_published_obs_vals is None
            
            if not is_all_zero or is_first_observation:
                # Compute indices dynamically based on observation_joint_count
                # Proprioceptive observation structure:
                # [0:12] = fixed values (base_ang_vel, imu, delta_yaw, commands, terrain flags)
                # [12:12+num_joints] = joint positions
                # [12+num_joints:12+2*num_joints] = joint velocities
                # [12+2*num_joints:12+3*num_joints] = previous action
                num_joints = self.observation_dimensions.observation_joint_count
                fixed_values_size = 12  # [0:12] base_ang_vel, imu, delta_yaw, commands, terrain flags
                joint_positions_start = fixed_values_size
                joint_positions_end = joint_positions_start + num_joints
                joint_velocities_start = joint_positions_end
                joint_velocities_end = joint_velocities_start + num_joints
                previous_action_start = joint_velocities_end
                previous_action_end = previous_action_start + num_joints
                
                # Extract joint positions
                # The environment uses self.asset.data.joint_pos - self.asset.data.default_joint_pos
                joint_positions_from_obs = obs_vals[joint_positions_start:joint_positions_end].astype(np.float32)
                
                # Extract joint velocities
                # The environment uses self.asset.data.joint_vel * 0.05
                joint_velocities_from_obs = obs_vals[joint_velocities_start:joint_velocities_end].astype(np.float32)
                
                # Extract previous action
                # When using cached observation from env.step(), it already has the correct previous_action
                # from action_history_buf[:, -1] (updated during env.step() via action_manager.process_action()).
                # For the initial observation (fallback path), previous_action should be zeros (no action applied yet).
                previous_action_from_obs = obs_vals[previous_action_start:previous_action_end].astype(np.float32)
                
                # Extract contact forces (after previous_action, remaining values in proprioceptive observation)
                # Contact forces start after previous_action and fill the rest of num_prop
                contact_forces_start = previous_action_end
                num_prop = self.observation_dimensions.num_prop
                contact_forces_end = min(contact_forces_start + 5, num_prop)  # HardwareObservations expects max 5 values
                contact_forces_from_obs = obs_vals[contact_forces_start:contact_forces_end].astype(np.float32)
                # Pad to 5 values if needed (HardwareObservations expects 5)
                if len(contact_forces_from_obs) < 5:
                    contact_forces_padded = np.zeros(5, dtype=np.float32)
                    contact_forces_padded[:len(contact_forces_from_obs)] = contact_forces_from_obs
                    contact_forces_from_obs = contact_forces_padded
                
                # Use observation as-is for duplicate checking (no modifications)
                # The observation from env.step() is the exact observation computed by the environment
                obs_vals_modified = obs_vals.copy()
                
                # Zero out history buffer for first observation
                # History buffer starts at index 223 (53 proprio + 132 scan + 9 priv_explicit + 29 priv_latent = 223)
                # History buffer is 530 values (10 history entries * 53 values each)
                # At timestep 0, the history buffer should be zeros from reset
                if is_first_observation:
                    obs_vals_modified[223:] = 0.0
                
                # Check 2: Reject duplicate observations (identical to last published, after previous_action update)
                is_duplicate = False
                if self._last_published_obs_vals is not None:
                    is_duplicate = np.allclose(obs_vals_modified, self._last_published_obs_vals, atol=1e-6)
                
                if not is_duplicate:
                    # Valid observation found
                    valid_observation_found = True
                else:
                    # Duplicate observation, will retry
                    time.sleep(retry_delay_s)
            else:
                # All-zero observation, will retry
                time.sleep(retry_delay_s)
            
            attempt += 1
        
        if not valid_observation_found:
            # Loop exhausted without finding valid observation
            error_msg = f"Failed to get valid observation after {max_attempts} attempts (all-zero or duplicate observations)"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Store this modified observation as the last published one
        self._last_published_obs_vals = obs_vals_modified.copy()
        
        # Extract scan features (measured_heights) from full observation
        # The full observation structure: [0:53] proprioceptive, [53:185] scan, [185:194] priv_explicit, [194:223] priv_latent, [223:] history
        # Scan features are at indices 53-184 (132 values, slice [53:185] accesses indices 53-184)
        # Need len >= 185 to access index 184 (last index of scan section)
        scan_features = obs_vals[53:185].astype(np.float32) if len(obs_vals) >= 185 else np.zeros(132, dtype=np.float32)
        
        # Extract privileged latent features (available in simulation)
        # Privileged latent is at indices 194-222 (29 values, slice [194:223] accesses indices 194-222)
        # Need len >= 223 to access index 222 (last index of priv_latent section)
        if len(obs_vals) < 223:
            raise ValueError(
                f"Observation length {len(obs_vals)} < 223, cannot extract privileged_latent (need index 222). "
                "IsaacSim HAL server requires full observation space including privileged_latent."
            )
        privileged_latent = obs_vals[194:223].astype(np.float32)
        
        # Use joint positions extracted from observation manager (ensures exact match).
        # Pad to robot joint count (12 or 18, hardware format) with zeros if needed.
        n_joints = self.robot_definition.get_total_joint_count()
        joint_positions = np.zeros(n_joints, dtype=np.float32)
        joint_positions[:len(joint_positions_from_obs)] = joint_positions_from_obs

        # Extract camera data: depth from front_camera (RayCaster), RGB from front_rgb (Pinhole) to simulate ZED 2i
        camera_height, camera_width = 480, 640  # IsaacSim fixed resolution
        rgb_camera_1 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        depth_map = np.zeros((camera_height, camera_width), dtype=np.float32)

        # Depth from named sensor "front_camera" (RayCaster)
        depth_sensor = self.camera_sensors.get("front_camera") if self.camera_sensors else None
        if depth_sensor is not None and hasattr(depth_sensor, 'data') and hasattr(depth_sensor.data, 'output'):
            depth_data = None
            if 'distance_to_camera' in depth_sensor.data.output:
                depth_data = depth_sensor.data.output["distance_to_camera"]
            elif 'distance_to_image_plane' in depth_sensor.data.output:
                depth_data = depth_sensor.data.output["distance_to_image_plane"]
            if depth_data is not None:
                if isinstance(depth_data, torch.Tensor):
                    if depth_data.ndim > 2:
                        depth_data = depth_data[0]
                    if depth_data.ndim == 3 and depth_data.shape[-1] == 1:
                        depth_data = depth_data.squeeze(-1)
                    depth_np = depth_data.detach().cpu().numpy().astype(np.float32)
                    if depth_np.shape != (camera_height, camera_width):
                        zoom_factors = (camera_height / depth_np.shape[0], camera_width / depth_np.shape[1])
                        depth_map = zoom(depth_np, zoom_factors, order=1).astype(np.float32)
                    else:
                        depth_map = depth_np

        # RGB from named sensor "front_rgb" (PinholeCamera) when available
        rgb_sensor = self.camera_sensors.get("front_rgb") if self.camera_sensors else None
        if rgb_sensor is not None and hasattr(rgb_sensor, 'data') and hasattr(rgb_sensor.data, 'output') and 'rgb' in rgb_sensor.data.output:
            rgb_data = rgb_sensor.data.output["rgb"]
            if isinstance(rgb_data, torch.Tensor):
                if rgb_data.ndim > 3:
                    rgb_data = rgb_data[0]
                rgb_np = rgb_data.detach().cpu().numpy()
                if rgb_np.dtype != np.uint8:
                    rgb_np = (rgb_np * 255).astype(np.uint8)
                if rgb_np.shape[:2] != (camera_height, camera_width):
                    zoom_factors = (camera_height / rgb_np.shape[0], camera_width / rgb_np.shape[1], 1)
                    rgb_camera_1 = zoom(rgb_np, zoom_factors, order=1).astype(np.uint8)
                else:
                    rgb_camera_1 = rgb_np.copy()
        else:
            # Fallback: RGB from any camera in list
            camera_list = list(self.camera_sensors.values()) if self.camera_sensors else []
            for cam in camera_list:
                if hasattr(cam, 'data') and hasattr(cam.data, 'output') and 'rgb' in cam.data.output:
                    rgb_data = cam.data.output["rgb"]
                    if isinstance(rgb_data, torch.Tensor):
                        if rgb_data.ndim > 3:
                            rgb_data = rgb_data[0]
                        rgb_np = rgb_data.detach().cpu().numpy()
                        if rgb_np.dtype != np.uint8:
                            rgb_np = (rgb_np * 255).astype(np.uint8)
                        if rgb_np.shape[:2] != (camera_height, camera_width):
                            zoom_factors = (camera_height / rgb_np.shape[0], camera_width / rgb_np.shape[1], 1)
                            rgb_camera_1 = zoom(rgb_np, zoom_factors, order=1).astype(np.uint8)
                        else:
                            rgb_camera_1 = rgb_np.copy()
                    break
        # Fallback: viewport render when no sensor RGB
        if not np.any(rgb_camera_1) and hasattr(self.env, 'render') and self.env.render_mode == "rgb_array":
            rgb_data = self.env.render()
            if rgb_data is not None and rgb_data.size > 0:
                if rgb_data.shape[:2] != (camera_height, camera_width):
                    zoom_factors = (camera_height / rgb_data.shape[0], camera_width / rgb_data.shape[1], 1)
                    rgb_camera_1 = zoom(rgb_data, zoom_factors, order=1).astype(np.uint8)
                else:
                    rgb_camera_1 = rgb_data.astype(np.uint8)

        # Extract robot state data (always available in Isaac Sim as torch.Tensor)
        # Use inference_mode to disable autograd and improve performance for all GPU->CPU transfers
        with torch.inference_mode():
            # Base angular velocity (body frame)
            ang_vel = self.robot.data.root_ang_vel_b
            if ang_vel.ndim == 2:
                ang_vel = ang_vel[0]
            base_ang_vel_b = ang_vel.detach().cpu().numpy().astype(np.float32)
            
            # Base linear velocity (body frame)
            lin_vel = self.robot.data.root_lin_vel_b
            if lin_vel.ndim == 2:
                lin_vel = lin_vel[0]
            base_lin_vel_b = lin_vel.detach().cpu().numpy().astype(np.float32)
            
            # Base quaternion (world frame)
            quat = self.robot.data.root_quat_w
            if quat.ndim == 2:
                quat = quat[0]
            base_quat_w = quat.detach().cpu().numpy().astype(np.float32)
            
            # Use joint velocities extracted from observation manager (ensures exact match).
            # Observation manager already applies * 0.05 scaling. Pad to 12 or 18 joints (hardware format) with zeros if needed.
            joint_velocities = np.zeros(n_joints, dtype=np.float32)
            joint_velocities[:len(joint_velocities_from_obs)] = joint_velocities_from_obs
        
        # Use previous action and contact forces extracted from observation manager (ensures exact match)
        # These were extracted inside the if block above, so they're available here
        previous_action = previous_action_from_obs
        contact_forces = contact_forces_from_obs

        # Extract delta_yaw and terrain flags from observation (reuse obs_vals already computed)
        # [0:3] base_ang_vel, [3:5] roll/pitch, [5] zero, [6] delta_yaw, [7] delta_next_yaw,
        # [8] zero, [9] vx, [10] terrain_type, [11] flat_terrain
        delta_yaw = float(obs_vals[6])
        delta_next_yaw = float(obs_vals[7])
        terrain_type_flag = float(obs_vals[10])
        flat_terrain_flag = float(obs_vals[11])
        
        has_camera_data = np.any(rgb_camera_1) or np.any(depth_map)
        camera_rgb = rgb_camera_1 if has_camera_data else None
        camera_depth = depth_map if has_camera_data else None

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
            delta_yaw=delta_yaw,
            delta_next_yaw=delta_next_yaw,
            terrain_type_flag=terrain_type_flag,
            flat_terrain_flag=flat_terrain_flag,
            scan_features=scan_features,
            privileged_latent=privileged_latent,
            camera_rgb=camera_rgb,
            camera_depth=camera_depth,
        )

        # Publish hardware observation via base-class publisher
        super().set_observation(hw_obs)
    
    def get_latest_observation(self) -> tuple[Optional[dict], Optional[torch.Tensor]]:
        """Get the latest observation that was computed in set_observation().
        
        Returns:
            Tuple of (obs_dict, obs_tensor) from the last call to set_observation().
            Returns (None, None) if set_observation() hasn't been called yet.
        """
        return self._latest_obs_dict, self._latest_obs_tensor


    def apply_command(self) -> torch.Tensor:
        """Get joint command from transport layer and convert to action tensor.
        
        Loops until a command is received or timeout is reached (if command_timeout_s is set).
        If command_timeout_s is None, waits forever and logs every 30s while waiting.
        Gets the latest command from the transport layer and uses IsaacSimMCUSDK
        to get normalized PWM values as a numpy array, then converts it to a
        torch tensor for compatibility with env.step() and calling code.
        Does NOT apply the action - env.step() will handle that.
        
        Returns:
            Action tensor ready to be passed to env.step().
            
        Raises:
            RuntimeError: If environment not available, SDK not initialized, or no command received within timeout (when timeout is not None).
        """
        if self.env is None:
            raise RuntimeError("No environment set, cannot apply command")
        
        if self._mcusdk is None:
            raise RuntimeError("IsaacSimMCUSDK not initialized. Call _initialize_mcusdk() first.")

        timeout_s = self._command_timeout_s
        poll_delay_s = 0.01  # 10ms between poll attempts
        wait_log_interval_s = 30.0
        start_time = time.time()
        
        while True:
            # Check timeout (only when a timeout is configured)
            if timeout_s is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout_s:
                    error_msg = (
                        f"Failed to receive joint command after {timeout_s}s timeout. "
                        f"Inference client may not be responding or is too slow."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            
            # Log every 30s while waiting for command (when waiting forever)
            if timeout_s is None:
                now = time.time()
                if self._waiting_log_last_t is None or (now - self._waiting_log_last_t) >= wait_log_interval_s:
                    logger.info("Waiting for joint command (start krabby-uno-sim to connect)...")
                    self._waiting_log_last_t = now
            
            # Poll for incoming command (non-blocking, 10ms timeout)
            poll_delay_ms = int(poll_delay_s * 1000)  # Convert to milliseconds for get_joint_command
            command = self.get_joint_command(timeout_ms=poll_delay_ms)
            if command is not None:
                # Drain queue so we apply the latest command, not stale ones (client sends ~50 Hz, we step at control_rate)
                while True:
                    latest = self.get_joint_command(timeout_ms=0)
                    if latest is None:
                        break
                    command = latest
                if not self._first_command_received:
                    self._first_command_received = True
                    logger.info("Client connected (joint command received).")
                d = command.to_positions_dict()
                non_zero = [(k, v) for k, v in d.items() if v != 0.0]
                logger.debug(
                    "Joint command received: %d joints, range=[%.3f, %.3f]",
                    len(d), min(d.values()), max(d.values()),
                )
                if non_zero:
                    logger.debug(
                        "  non-zero positions (joint=rad): %s",
                        ", ".join(f"{k}={v:.3f}" for k, v in non_zero),
                    )
                else:
                    logger.debug("  all positions zero (no leg selected or sticks neutral)")
                break

            # No command available, sleep and retry
            time.sleep(poll_delay_s)

        # Get number of environments from unwrapped environment
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        num_envs = unwrapped_env.num_envs

        # SDK takes JointCommand and returns dict for array conversion
        joint_names = self.robot_definition.get_joint_names()
        action_dict = self._mcusdk.apply_command(command, num_envs=num_envs)

        # Adapter: dict -> ordered array using get_joint_names() so indices stay in sync
        action_dim = self.robot_definition.get_total_joint_count()
        action_list = [action_dict[name] for name in joint_names[:action_dim]]
        action_np = np.array(action_list, dtype=np.float32)

        # Convert to torch tensor for compatibility with env.step() and calling code
        device = unwrapped_env.device
        action = torch.from_numpy(action_np).to(device=device, dtype=torch.float32)
        
        # Add batch dimension if needed (env.step() expects (num_envs, action_dim))
        if action.ndim == 1:
            action = action.unsqueeze(0)  # Shape: (1, 12)
        
        # Expand to num_envs if needed
        if action.shape[0] == 1 and num_envs > 1:
            action = action.expand(num_envs, -1)  # Shape: (num_envs, 12)

        return action

