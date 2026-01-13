"""IsaacSim HAL server implementation."""

import logging
import time
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import zoom

from hal.server import HalServerBase, HalServerConfig
from hal.client.data_structures.hardware import HardwareObservations, JointCommand
from hal.server.isaac.isaacsim_mcusdk import IsaacSimMCUSDK

logger = logging.getLogger(__name__)


class IsaacSimHalServer(HalServerBase):
    """HAL server for IsaacSim environment.
    
    Extracts observations from IsaacSim environment and publishes via HAL.
    Applies joint commands received via HAL to the environment.
    """

    def __init__(self, config: HalServerConfig, env=None):
        """Initialize IsaacSim HAL server.
        
        Args:
            config: HAL server configuration
            env: IsaacSim environment. If provided, environment component
                references will be cached via _cache_references().
        
        Note:
            If env is provided, this will call _cache_references() to extract
            and cache references to scene, robot, sensors, and managers.
        """
        super().__init__(config)
        self.env = env
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
        self._last_applied_action = np.zeros(12, dtype=np.float32)
        # Track last published observation to detect duplicates
        self._last_published_obs_vals = None
        
        # Initialize IsaacSim MCU SDK for standardized command application
        # Device will be set when environment is available
        self._mcusdk: Optional[IsaacSimMCUSDK] = None

        if env is not None:
            self._cache_references()
            self._initialize_mcusdk()

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

        # Cache camera sensors if available
        # Cameras are accessed via scene.sensors dictionary, which supports .items() iteration
        self.camera_sensors = {}
        if hasattr(self.scene, 'sensors'):
            # Find camera sensors in the sensors dictionary
            for sensor_name, sensor in self.scene.sensors.items():
                # Check if it's a camera-like sensor (RayCasterCamera, Camera, etc.)
                if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
                    # Check for depth or RGB outputs
                    if 'distance_to_camera' in sensor.data.output or 'distance_to_image_plane' in sensor.data.output:
                        self.camera_sensors[sensor_name] = sensor

        # Cache managers (use unwrapped to access underlying environment)
        self.observation_manager = unwrapped_env.observation_manager
        self.action_manager = unwrapped_env.action_manager
        if hasattr(unwrapped_env, 'command_manager'):
            self.command_manager = unwrapped_env.command_manager
        
        # Cache contact sensor if available
        if hasattr(self.scene, 'sensors') and 'contact_forces' in self.scene.sensors:
            self.contact_sensor = self.scene.sensors['contact_forces']

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
        """Initialize IsaacSim MCU SDK with environment device.
        
        This method initializes the SDK after environment references are cached,
        so we can get the correct device (CPU/CUDA) from the environment.
        """
        if self.env is None:
            return
        
        # Get device from unwrapped environment (gym.make() wraps the environment)
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        device = getattr(unwrapped_env, 'device', torch.device("cpu"))
        
        # Initialize SDK with environment device
        self._mcusdk = IsaacSimMCUSDK(device=device)
        logger.info(f"Initialized IsaacSimMCUSDK with device: {device}")

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
            # Extract all observation components from observation manager's computed observation
            # This ensures we use the exact same values as the environment
            # In Isaac Sim, observation_manager is always available
            obs_dict = self.observation_manager.compute()
            obs_tensor = obs_dict["policy"]
            
            # Cache observation for logging (avoid recomputing)
            self._latest_obs_dict = obs_dict
            self._latest_obs_tensor = obs_tensor
            
            # Convert to numpy for extraction
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
                # Extract joint positions (indices 12-23 in proprioceptive observation)
                # The environment uses self.asset.data.joint_pos - self.asset.data.default_joint_pos
                joint_positions_from_obs = obs_vals[12:24].astype(np.float32)
                
                # Extract joint velocities (indices 24-35 in proprioceptive observation)
                # The environment uses self.asset.data.joint_vel * 0.05
                joint_velocities_from_obs = obs_vals[24:36].astype(np.float32)
                
                # Extract previous action (indices 36-47 in proprioceptive observation)
                # NOTE: The observation manager's action_history_buf is updated during env.step(),
                # but we call set_observation() BEFORE env.step(), so the observation manager's
                # previous action is from 2 steps ago, not 1 step ago. We need to use our tracked
                # last applied action instead.
                # previous_action_from_obs = obs_vals[36:48].astype(np.float32)  # This is stale
                previous_action_from_obs = self._last_applied_action.copy()  # Use tracked action
                
                # Create a modified observation array with updated previous_action for duplicate checking
                # This ensures we check for duplicates after updating previous_action, not before
                obs_vals_modified = obs_vals.copy()
                obs_vals_modified[36:48] = previous_action_from_obs
                
                # Zero out history buffer for first observation to match play script behavior
                # History buffer starts at index 223 (53 proprio + 132 scan + 9 priv_explicit + 29 priv_latent = 223)
                # History buffer is 530 values (10 history entries * 53 values each)
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
        
        # Use joint positions extracted from observation manager (ensures exact match)
        # Pad to 12 joints (hardware format) with zeros if needed
        joint_positions = np.zeros(12, dtype=np.float32)
        joint_positions[:len(joint_positions_from_obs)] = joint_positions_from_obs

        # Extract camera data from sensors
        camera_height, camera_width = 480, 640  # IsaacSim fixed resolution
        rgb_camera_1 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        rgb_camera_2 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        depth_map = np.zeros((camera_height, camera_width), dtype=np.float32)
        confidence_map = np.ones((camera_height, camera_width), dtype=np.float32)

        # Try to get depth data from camera sensors
        camera_list = list(self.camera_sensors.values()) if self.camera_sensors else []
        
        if len(camera_list) > 0:
            # Get depth from first camera
            camera_0 = camera_list[0]
            if hasattr(camera_0, 'data') and hasattr(camera_0.data, 'output'):
                # Try different depth output formats
                depth_data = None
                if 'distance_to_camera' in camera_0.data.output:
                    depth_data = camera_0.data.output["distance_to_camera"]
                elif 'distance_to_image_plane' in camera_0.data.output:
                    depth_data = camera_0.data.output["distance_to_image_plane"]
                
                if depth_data is not None:
                    # Convert to numpy
                    if isinstance(depth_data, torch.Tensor):
                        # Handle batched data - take first environment
                        if depth_data.ndim > 2:
                            depth_data = depth_data[0]
                        # Remove channel dimension if present
                        if depth_data.ndim == 3 and depth_data.shape[-1] == 1:
                            depth_data = depth_data.squeeze(-1)
                        depth_np = depth_data.detach().cpu().numpy().astype(np.float32)
                        
                        # Resize if needed
                        if depth_np.shape != (camera_height, camera_width):
                            zoom_factors = (camera_height / depth_np.shape[0], 
                                          camera_width / depth_np.shape[1])
                            depth_map = zoom(depth_np, zoom_factors, order=1).astype(np.float32)
                        else:
                            depth_map = depth_np
            
            # Try to get RGB from second camera or render product
            if len(camera_list) > 1:
                camera_1 = camera_list[1]
                # Try to get RGB if available
                if hasattr(camera_1, 'data') and hasattr(camera_1.data, 'output'):
                    if 'rgb' in camera_1.data.output:
                        rgb_data = camera_1.data.output["rgb"]
                        if isinstance(rgb_data, torch.Tensor):
                            if rgb_data.ndim > 3:
                                rgb_data = rgb_data[0]
                            rgb_np = rgb_data.detach().cpu().numpy()
                            # Convert to uint8 if needed
                            if rgb_np.dtype != np.uint8:
                                rgb_np = (rgb_np * 255).astype(np.uint8)
                            if rgb_np.shape[:2] != (camera_height, camera_width):
                                zoom_factors = (camera_height / rgb_np.shape[0], 
                                              camera_width / rgb_np.shape[1], 1)
                                rgb_camera_2 = zoom(rgb_np, zoom_factors, order=1).astype(np.uint8)
                            else:
                                rgb_camera_2 = rgb_np

        # Try to get RGB from render product if available (for first camera)
        if hasattr(self.env, 'render') and self.env.render_mode == "rgb_array":
            rgb_data = self.env.render()
            if rgb_data is not None and rgb_data.size > 0:
                # rgb_data is typically (H, W, 3) uint8
                if rgb_data.shape[:2] != (camera_height, camera_width):
                    zoom_factors = (camera_height / rgb_data.shape[0], 
                                  camera_width / rgb_data.shape[1], 1)
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
            
            # Use joint velocities extracted from observation manager (ensures exact match)
            # Note: observation manager already applies * 0.05 scaling, so we use them directly
            # Pad to 12 joints (hardware format) with zeros if needed
            joint_velocities = np.zeros(12, dtype=np.float32)
            joint_velocities[:len(joint_velocities_from_obs)] = joint_velocities_from_obs
        
        # Extract contact forces (indices 48-52 in proprioceptive observation)
        # The environment has 5 contact force values, not 4
        # Reuse obs_vals already computed above
        contact_forces = obs_vals[48:53].astype(np.float32)  # 5 values: indices 48-52
        
        # Use previous action extracted from observation manager (ensures exact match)
        previous_action = previous_action_from_obs

        # Extract delta_yaw and terrain flags from observation (reuse obs_vals already computed)
        # [0:3] base_ang_vel, [3:5] roll/pitch, [5] zero, [6] delta_yaw, [7] delta_next_yaw,
        # [8] zero, [9] vx, [10] terrain_type, [11] flat_terrain
        delta_yaw = float(obs_vals[6])
        delta_next_yaw = float(obs_vals[7])
        terrain_type_flag = float(obs_vals[10])
        flat_terrain_flag = float(obs_vals[11])
        
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
            delta_yaw=delta_yaw,
            delta_next_yaw=delta_next_yaw,
            terrain_type_flag=terrain_type_flag,
            flat_terrain_flag=flat_terrain_flag,
            scan_features=scan_features,
            privileged_latent=privileged_latent,
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
        
        Loops until a command is received or timeout is reached.
        Gets the latest command from the transport layer and uses IsaacSimMCUSDK
        to convert it to an action tensor that can be passed to env.step().
        Does NOT apply the action - env.step() will handle that.
        
        Returns:
            Action tensor ready to be passed to env.step().
            
        Raises:
            RuntimeError: If environment not available, SDK not initialized, or no command received within timeout.
        """
        if self.env is None:
            raise RuntimeError("No environment set, cannot apply command")
        
        if self._mcusdk is None:
            raise RuntimeError("IsaacSimMCUSDK not initialized. Call _initialize_mcusdk() first.")

        # Loop until command received or timeout
        timeout_s = 1.0  # 1s timeout
        poll_delay_s = 0.01  # 10ms between poll attempts
        start_time = time.time()
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_s:
                error_msg = (
                    f"Failed to receive joint command after {timeout_s}s timeout. "
                    f"Inference client may not be responding or is too slow."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Poll for incoming command (non-blocking, 10ms timeout)
            poll_delay_ms = int(poll_delay_s * 1000)  # Convert to milliseconds for get_joint_command
            command = self.get_joint_command(timeout_ms=poll_delay_ms)
            if command is not None:
                break
            
            # No command available, sleep and retry
            time.sleep(poll_delay_s)

        # Get number of environments from unwrapped environment
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        num_envs = getattr(unwrapped_env, 'num_envs', 1)
        
        # Use standardized SDK to convert command to action tensor
        # The SDK handles device placement, batch dimensions, and logging
        action = self._mcusdk.apply_command(command, num_envs=num_envs)

        return action

