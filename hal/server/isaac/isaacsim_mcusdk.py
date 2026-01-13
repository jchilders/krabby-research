"""IsaacSim MCU SDK interface for applying joint commands.

This module provides a standardized SDK interface for applying joint commands
to IsaacSim environments. It converts JointCommand structures into the format
expected by IsaacSim (torch.Tensor) and logs commands in Isaac's preferred joint format.
"""

import logging
from typing import Optional

import numpy as np
import torch

from hal.client.data_structures.hardware import JointCommand

logger = logging.getLogger(__name__)


class IsaacSimMCUSDK:
    """Standardized SDK interface for applying joint commands to IsaacSim.
    
    This class provides a clean interface for converting JointCommand structures
    into the format expected by IsaacSim (torch.Tensor) and applying them to the
    environment. It replaces ad hoc command application code with a standardized
    SDK wrapper.
    
    The SDK handles:
    - Converting JointCommand to torch.Tensor format
    - Logging commands in Isaac's preferred joint format
    - Device placement (CPU/CUDA)
    - Batch dimension handling
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize IsaacSim MCU SDK.
        
        Args:
            device: Target device for tensors (CPU or CUDA). If None, uses CPU.
        """
        self.device = device if device is not None else torch.device("cpu")
        logger.info(f"IsaacSimMCUSDK initialized with device: {self.device}")
    
    def apply_command(
        self,
        command: JointCommand,
        num_envs: int = 1,
    ) -> torch.Tensor:
        """Apply joint command to IsaacSim environment.
        
        Converts JointCommand to torch.Tensor format expected by IsaacSim.
        Logs the command in Isaac's preferred joint format for debugging.
        
        Args:
            command: JointCommand containing joint positions (18 joints for hexapod)
            num_envs: Number of environments (for batch dimension). Default: 1.
            
        Returns:
            Action tensor ready to be passed to env.step().
            Shape: (num_envs, 18) for hexapod with 18 joints.
            
        Raises:
            ValueError: If command is invalid or has wrong number of joints.
        """
        if not isinstance(command, JointCommand):
            raise ValueError(f"command must be JointCommand, got {type(command)}")
        
        # Extract joint positions array from command
        command_array = command.joint_positions
        
        # Validate joint positions shape (should be 18 for hexapod)
        if command_array.shape != (18,):
            raise ValueError(
                f"Expected 18 joints for hexapod, got {command_array.shape[0]} joints"
            )
        
        # Convert NumPy array to tensor (zero-copy when array is C-contiguous float32)
        # The joint_positions array from JointCommand is already
        # a zero-copy view of the bytes and is C-contiguous float32
        action = torch.from_numpy(command_array).to(device=self.device, dtype=torch.float32)
        
        # Add batch dimension if needed (env.step() expects (num_envs, action_dim))
        if action.ndim == 1:
            action = action.unsqueeze(0)  # Shape: (1, 18)
        
        # Expand to num_envs if needed
        if action.shape[0] == 1 and num_envs > 1:
            action = action.expand(num_envs, -1)  # Shape: (num_envs, 18)
        
        # Log command in Isaac's preferred joint format
        # Format: joint positions as comma-separated values with joint names
        # For hexapod: 6 legs × 3 DOF (hip_yaw, hip_pitch, knee)
        # Joint order: FL, FR, ML, MR, RL, RR (each leg has 3 joints)
        joint_names = [
            "FL_hip_yaw", "FL_hip_pitch", "FL_knee",
            "FR_hip_yaw", "FR_hip_pitch", "FR_knee",
            "ML_hip_yaw", "ML_hip_pitch", "ML_knee",
            "MR_hip_yaw", "MR_hip_pitch", "MR_knee",
            "RL_hip_yaw", "RL_hip_pitch", "RL_knee",
            "RR_hip_yaw", "RR_hip_pitch", "RR_knee",
        ]
        
        # Log joint positions in Isaac's preferred format
        joint_values_str = ", ".join(
            f"{name}={val:.4f}" for name, val in zip(joint_names, command_array)
        )
        logger.debug(
            f"IsaacSimMCUSDK: Applying joint command (timestamp_ns={command.timestamp_ns}, "
            f"observation_timestamp_ns={command.observation_timestamp_ns}): "
            f"{joint_values_str}"
        )
        
        # Log summary statistics
        logger.debug(
            f"IsaacSimMCUSDK: Joint command stats - "
            f"min={command_array.min():.4f}, max={command_array.max():.4f}, "
            f"mean={command_array.mean():.4f}, std={command_array.std():.4f}"
        )
        
        return action
    
    def set_device(self, device: torch.device) -> None:
        """Set target device for tensors.
        
        Args:
            device: Target device (CPU or CUDA).
        """
        self.device = device
        logger.info(f"IsaacSimMCUSDK device set to: {self.device}")
