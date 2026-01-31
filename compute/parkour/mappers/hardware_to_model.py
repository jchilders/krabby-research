"""Mapper from hardware observations to Parkour model format.

This mapper converts raw hardware sensor data into the format expected by
the Parkour policy model. It uses zero-copy operations where possible to
minimize data copying.
"""

import logging
from typing import Optional, Type

import numpy as np
import torch
from compute.parkour.utils.math import euler_xyz_from_quat, wrap_to_pi

from hal.client.data_structures.hardware import HardwareObservations
from hal.client.observation.types import NavigationCommand
from compute.parkour.parkour_types import (
    NUM_PROP,
    NUM_SCAN,
    NUM_PRIV_EXPLICIT,
    NUM_PRIV_LATENT,
    NUM_HIST,
    HISTORY_DIM,
    OBS_DIM,
    ParkourObservation,
    TeacherObservation,
)

logger = logging.getLogger(__name__)


class HWObservationsToParkourMapper:
    """Maps hardware observations to Parkour model format.
    
    Uses zero-copy operations where possible to minimize data copying.
    Only copies when structural transformation is required.
    
    Zero-copy guarantees:
    - Large arrays (RGB, depth) are processed but may require copies for
      feature extraction (depends on preprocessing pipeline)
    - Joint positions can be views if source is compatible
    - Final observation array is constructed from parts (may require copy)
    
    The mapper implements the full feature extraction pipeline matching the
    training environment's observation structure.
    """
    
    def __init__(self):
        """Initialize the mapper with history buffer."""
        # History buffer: stores last NUM_HIST proprioceptive observations
        # Shape: (NUM_HIST, NUM_PROP) = (10, 53) = 530 dims
        self._history_buffer = np.zeros((NUM_HIST, NUM_PROP), dtype=np.float32)
        self._episode_step = 0
        # Previous action: stores last joint command (12 dims)
        self._previous_action = np.zeros(12, dtype=np.float32)
    
    def set_previous_action(self, action: np.ndarray) -> None:
        """Set the previous action for history tracking.
        
        Args:
            action: Previous joint command (12 dims, will be truncated/padded if needed)
        """
        num_dims = min(12, len(action))
        self._previous_action[:num_dims] = action[:num_dims]
    
    def map(self, hw_obs: HardwareObservations, nav_cmd: Optional[NavigationCommand] = None, observation_type: Type[ParkourObservation] = TeacherObservation) -> ParkourObservation:
        """Map hardware observations to model format.
        
        Args:
            hw_obs: Hardware observation data
            nav_cmd: Optional navigation command (vx, vy, yaw_rate) to include in observation
            observation_type: Type of observation to create (default: TeacherObservation)
            
        Returns:
            ParkourObservation in model format
            
        Raises:
            ValueError: If hardware observation is invalid
        """
        # Extract features from hardware data
        proprioceptive = self._extract_proprioceptive(hw_obs, nav_cmd)
        scan = self._extract_scan_features(hw_obs)
        priv_explicit = self._extract_priv_explicit(hw_obs)
        priv_latent = self._extract_priv_latent(hw_obs)
        history = self._extract_history(proprioceptive)
        
        # Create observation from parts (use specified observation type)
        return observation_type.from_parts(
            proprioceptive=proprioceptive,
            scan=scan,
            priv_explicit=priv_explicit,
            priv_latent=priv_latent,
            history=history,
            timestamp_ns=hw_obs.timestamp_ns,
        )
    
    def _extract_proprioceptive(self, hw_obs: HardwareObservations, nav_cmd: Optional[NavigationCommand] = None) -> np.ndarray:
        """Extract proprioceptive features from hardware observations.
        
        Matches training format from ExtremeParkourObservations:
        - [0:3]   base angular velocity (body frame) * 0.25
        - [3:5]   IMU roll, pitch (wrapped to [-pi, pi])
        - [5]     zero (placeholder)
        - [6]     delta_yaw (target yaw - current yaw)
        - [7]     delta_next_yaw (next target yaw - current yaw)
        - [8]     zero (placeholder for vy command)
        - [9]     vx command (forward velocity)
        - [10]    terrain type flag (1 if not flat, 0 if flat)
        - [11]    flat terrain flag (1 if flat, 0 if not flat)
        - [12:24] joint positions (relative to default) - 12 joints
        - [24:36] joint velocities * 0.05 - 12 joints
        - [36:48] previous action (last joint command) - 12 joints
        - [48:52] contact forces (4 feet, normalized to [-0.5, 0.5])
        - [52]    (padding if needed)
        
        Args:
            hw_obs: Hardware observations
            nav_cmd: Optional navigation command to include in proprioceptive features
            
        Returns:
            Proprioceptive features array of shape (NUM_PROP,)
        """
        proprioceptive = np.zeros(NUM_PROP, dtype=np.float32)
        
        # [0:3] Base angular velocity (body frame) * 0.25
        proprioceptive[0:3] = hw_obs.base_ang_vel_b * 0.25
        
        # [3:5] IMU roll, pitch (derived from base quaternion)
        # Use same conversion as environment: euler_xyz_from_quat + wrap_to_pi
        # Quaternion format from HardwareObservations: (x, y, z, w)
        # Convert numpy quaternion to torch tensor (matching environment format)
        quat_torch = torch.from_numpy(hw_obs.base_quat_w).unsqueeze(0)  # Add batch dimension
        roll, pitch, _ = euler_xyz_from_quat(quat_torch)
        # Wrap to [-pi, pi] (matching environment)
        roll_wrapped = wrap_to_pi(roll)
        pitch_wrapped = wrap_to_pi(pitch)
        proprioceptive[3] = roll_wrapped.item()
        proprioceptive[4] = pitch_wrapped.item()
        
        # [5] Zero (placeholder)
        proprioceptive[5] = 0.0
        
        # [6] delta_yaw (target yaw - current yaw)
        # Use delta_yaw from HardwareObservations (extracted from environment)
        proprioceptive[6] = float(hw_obs.delta_yaw)
        
        # [7] delta_next_yaw (next target yaw - current yaw)
        # Use delta_next_yaw from HardwareObservations (extracted from environment)
        proprioceptive[7] = float(hw_obs.delta_next_yaw)
        
        # [8] Zero (placeholder for vy command)
        proprioceptive[8] = 0.0
        
        # [9] vx command (forward velocity)
        if nav_cmd is not None:
            proprioceptive[9] = nav_cmd.vx
        else:
            proprioceptive[9] = 0.0
        
        # [10] Terrain type flag (1 if not flat, 0 if flat)
        # Use terrain_type_flag from HardwareObservations (extracted from environment)
        proprioceptive[10] = float(hw_obs.terrain_type_flag)
        
        # [11] Flat terrain flag (1 if flat, 0 if not flat)
        # Use flat_terrain_flag from HardwareObservations (extracted from environment)
        proprioceptive[11] = float(hw_obs.flat_terrain_flag)
        
        # [12:24] Joint positions (relative to default) - 12 joints
        # NOTE: HardwareObservations has 12 joints, matching training
        # HAL server now extracts the exact 12 joints from observation manager
        # Use first 12 joints (already in correct format from observation manager)
        num_joints = min(12, len(hw_obs.joint_positions))
        proprioceptive[12:12+num_joints] = hw_obs.joint_positions[:num_joints]
        
        # [24:36] Joint velocities * 0.05 - 12 joints
        # HAL server now extracts joint velocities with * 0.05 already applied from observation manager
        # So we use them directly (no need to multiply by 0.05 again)
        num_joints = min(12, len(hw_obs.joint_velocities))
        proprioceptive[24:24+num_joints] = hw_obs.joint_velocities[:num_joints]
        
        # [36:48] Previous action (last joint command) - 12 joints
        # Use previous_action from hardware observations (always provided by HAL server)
        proprioceptive[36:48] = hw_obs.previous_action
        
        # [48:53] Contact forces (5 values from environment, normalized to [-0.5, 0.5])
        # The environment observation has 5 contact values (indices 48-52)
        num_contact = min(5, len(hw_obs.contact_forces))
        proprioceptive[48:48+num_contact] = hw_obs.contact_forces[:num_contact]
        
        return proprioceptive
    
    def _extract_scan_features(self, hw_obs: HardwareObservations) -> np.ndarray:
        """Extract scan/depth features from hardware observations.
        
        Matches training format: 132 height measurements from ray scanner.
        The HAL server now extracts scan features directly from the observation manager's
        measured_heights, which ensures exact matching with the environment.
        
        If scan_features is available in HardwareObservations (extracted from observation
        manager), use it directly. Otherwise, fall back to reconstructing from depth_map.
        
        Args:
            hw_obs: Hardware observations
            
        Returns:
            Scan features array of shape (NUM_SCAN,)
        """
        # Check if scan_features is available (extracted from observation manager)
        # This ensures exact matching with the environment's measured_heights
        if hasattr(hw_obs, 'scan_features') and hw_obs.scan_features is not None:
            scan_features = hw_obs.scan_features
            if len(scan_features) == NUM_SCAN:
                return scan_features.astype(np.float32)
            elif len(scan_features) > NUM_SCAN:
                return scan_features[:NUM_SCAN].astype(np.float32)
            else:
                # Pad with zeros if shorter
                features = np.zeros(NUM_SCAN, dtype=np.float32)
                features[:len(scan_features)] = scan_features.astype(np.float32)
                return features
        
        # Fallback: reconstruct from depth_map (for backward compatibility)
        # This should not be used in normal operation when HAL server provides scan_features
        height, width = hw_obs.depth_map.shape
        
        # Sample points from depth image to simulate ray-casting
        # Use a grid pattern that covers the image
        # Calculate grid dimensions that approximate NUM_SCAN
        grid_rows = int(np.sqrt(NUM_SCAN))
        grid_cols = (NUM_SCAN + grid_rows - 1) // grid_rows  # Ceiling division
        
        features = np.zeros(NUM_SCAN, dtype=np.float32)
        
        row_indices = np.linspace(0, height - 1, grid_rows, dtype=np.int32)
        col_indices = np.linspace(0, width - 1, grid_cols, dtype=np.int32)
        
        idx = 0
        for row in row_indices:
            for col in col_indices:
                if idx >= NUM_SCAN:
                    break
                # Get depth value at this point
                depth_value = hw_obs.depth_map[row, col]
                
                # Convert depth to height measurement (relative to camera)
                # Depth is distance from camera, height is relative to camera position
                # For simplicity, use depth directly as height measurement
                # Apply normalization: clip(height - 0.3, -1, 1)
                height_measurement = np.clip(depth_value - 0.3, -1.0, 1.0)
                features[idx] = height_measurement
                idx += 1
            if idx >= NUM_SCAN:
                break
        
        return features
    
    def _extract_priv_explicit(self, hw_obs: HardwareObservations) -> np.ndarray:
        """Extract privileged explicit features from hardware observations.
        
        Matches training format: 9 dims
        - [0:3] base_lin_vel * 2.0
        - [3:6] zero (placeholder)
        - [6:9] zero (placeholder)
        
        NOTE: These features are typically estimated by the estimator network
        during inference, so returning zeros is acceptable. The estimator will
        fill them in based on proprioceptive features.
        
        Args:
            hw_obs: Hardware observations
            
        Returns:
            Privileged explicit features array of shape (NUM_PRIV_EXPLICIT,)
        """
        # Privileged explicit features are estimated by the estimator network
        # during inference, so we return zeros here
        # The estimator will fill them in based on proprioceptive features
        return np.zeros(NUM_PRIV_EXPLICIT, dtype=np.float32)
    
    def _extract_priv_latent(self, hw_obs: HardwareObservations) -> np.ndarray:
        """Extract privileged latent features from hardware observations.
        
        Matches training format: 29 dims
        - Body mass and center of mass
        - Friction coefficients
        - Joint stiffness and damping (normalized)
        
        In simulation, privileged latent is available from environment.
        On real hardware, it's not available and will be zeros.
        
        Args:
            hw_obs: Hardware observations
            
        Returns:
            Privileged latent features array of shape (NUM_PRIV_LATENT,)
        """
        # If privileged latent is available (simulation), use it
        if hw_obs.privileged_latent is not None:
            if hw_obs.privileged_latent.shape != (NUM_PRIV_LATENT,):
                raise ValueError(f"privileged_latent shape {hw_obs.privileged_latent.shape} != ({NUM_PRIV_LATENT},)")
            return hw_obs.privileged_latent.astype(np.float32)
        
        # On real hardware, privileged latent is not available
        # Return zeros (policy encoder will infer from proprioceptive observations)
        return np.zeros(NUM_PRIV_LATENT, dtype=np.float32)
    
    def _extract_history(self, proprioceptive: np.ndarray) -> np.ndarray:
        """Extract history features from previous proprioceptive observations.
        
        Matches training format: 530 dims (NUM_HIST * NUM_PROP)
        - Stores last NUM_HIST (10) proprioceptive observations
        - On episode start (step <= 1), fills buffer with current observation
        - Otherwise, shifts buffer and appends current observation
        
        Args:
            proprioceptive: Current proprioceptive observation (NUM_PROP,)
            
        Returns:
            History features array of shape (HISTORY_DIM,)
        """
        # Update history buffer
        # On episode start (step <= 1), fill buffer with current observation
        # Otherwise, shift buffer and append current observation
        if self._episode_step <= 1:
            # Fill buffer with current observation
            for i in range(NUM_HIST):
                self._history_buffer[i] = proprioceptive.copy()
        else:
            # Shift buffer and append current observation
            self._history_buffer[:-1] = self._history_buffer[1:]
            self._history_buffer[-1] = proprioceptive.copy()
        
        self._episode_step += 1
        
        # Return flattened history buffer
        return self._history_buffer.flatten()
