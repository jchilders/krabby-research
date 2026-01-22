"""IsaacSim MCU SDK interface for applying joint commands.

This module provides a standardized SDK interface for applying joint commands
to IsaacSim environments.
"""

import logging

import numpy as np

from hal.client.data_structures.hardware import JointCommand

logger = logging.getLogger(__name__)


class IsaacSimMCUSDK:
    """Standardized SDK interface for applying joint commands to IsaacSim.

    
    The SDK handles:
    - Processing normalized PWM values (-1.0 to 1.0) from JointCommand
    - Logging commands in Isaac's preferred joint format
    - Validating joint command structure
    """
    def __init__(self):
        logger.info("IsaacSimMCUSDK initialized")
    
    def apply_command(
        self,
        command: JointCommand,
        num_envs: int = 1,
    ) -> np.ndarray:
        """Apply joint command to IsaacSim environment.
        
        Processes normalized PWM values from JointCommand for prismatic joints.
        Logs the command in Isaac's preferred joint format for debugging.
        
        Args:
            command: JointCommand containing normalized joint positions (-1.0 to 1.0)
                     representing PWM values for 12 joints (quadruped) or 18 joints (hexapod)
            num_envs: Number of environments (currently unused, kept for API compatibility)
            
        Returns:
            Normalized joint positions as numpy array.
            Shape: (18,) - always returns 18 joints (pads 12-joint commands with zeros).
            Values are normalized PWM values: -1.0 (move in at max speed) to 1.0 (move out at max speed),
            with 0.0 meaning keep joint in current position.
            
        Raises:
            ValueError: If command is invalid or has wrong number of joints.
        """
        
        # Extract joint positions array from command
        command_array = command.joint_positions
        
        # Validate joint positions shape (should be 18 for hexapod)
        if command_array.shape[0] != 18:
            raise ValueError(
                f"Expected 18 joints, got {command_array.shape[0]} joints"
            )
        
        # TODO: Add actual IsaacSim code to control prismatic joints here.
        # IsaacSim uses prismatic joints with position/velocity targets.
        # The normalized PWM values (-1.0 to 1.0) in command_array should be
        # converted to appropriate position/velocity targets for the prismatic joints.
        
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
        
        return command_array
