"""Krabby MCU SDK interface for applying joint commands.

This module provides a standardized SDK interface for applying joint commands
to the Krabby robot MCU (real hardware).
"""

import logging
from typing import Optional

import numpy as np

from hal.client.data_structures.hardware import JointCommand

# Import the MCU SDK from firmware package
# TODO: This import path may need adjustment based on package structure
try:
    from firmware.krabby_mcu import KrabbyMCUSDK as FirmwareKrabbyMCUSDK
except ImportError:
    # Fallback for development/testing without firmware package
    FirmwareKrabbyMCUSDK = None  # type: ignore

logger = logging.getLogger(__name__)

# Joint name mapping for hexapod (18 joints)
# Format: FL_hip_yaw, FL_hip_pitch, FL_knee, FR_hip_yaw, etc.
# Order: FL, FR, ML, MR, RL, RR (each leg has 3 joints: hip_yaw, hip_pitch, knee)
JOINT_NAMES = [
    "FL_hip_yaw", "FL_hip_pitch", "FL_knee",
    "FR_hip_yaw", "FR_hip_pitch", "FR_knee",
    "ML_hip_yaw", "ML_hip_pitch", "ML_knee",
    "MR_hip_yaw", "MR_hip_pitch", "MR_knee",
    "RL_hip_yaw", "RL_hip_pitch", "RL_knee",
    "RR_hip_yaw", "RR_hip_pitch", "RR_knee",
]

# Joint position normalization parameters
# MCU expects normalized values [0.0, 1.0] where:
# - 0.0 = minimum position (fully retracted/negative)
# - 0.5 = neutral position (zero radians)
# - 1.0 = maximum position (fully extended/positive)
# Joint positions from JointCommand are in radians, need to be normalized
# Assuming joint limits: ±0.5 radians 
# TODO: this will need to be adjusted as testing is done with the actuators.
JOINT_LIMIT_RAD = 0.5  # Maximum joint deflection in radians
JOINT_NEUTRAL = 0.5    # Normalized neutral position


class KrabbyMCUSDK:
    """Standardized SDK interface for applying joint commands to Krabby MCU.
    
    The SDK handles:
    - Converting JointCommand (18 joints in radians) to MCU format
    - Mapping joint names from hexapod format to MCU command format
    - Wrapping the firmware KrabbyMCUSDK for standardized interface
    """
    
    def __init__(self, port: Optional[str] = None, baud: int = 115200, auto_connect: bool = True):
        """Initialize Krabby MCU SDK.
        
        Args:
            port: Serial port for MCU connection (e.g., "/dev/ttyACM0").
                If None, uses default from firmware SDK.
            baud: Baud rate for serial communication. Default: 115200.
            auto_connect: If True, automatically connect to MCU on initialization.
                If False, connection must be done manually. Default: True.
        """
        if FirmwareKrabbyMCUSDK is None:
            raise RuntimeError(
                "KrabbyMCUSDK firmware module not available. "
                "Cannot initialize MCU SDK without firmware package."
            )
        
        self._mcu = FirmwareKrabbyMCUSDK(port=port, baud=baud)
        self._connected = False
        
        logger.info(f"KrabbyMCUSDK initialized (port={port}, baud={baud}, auto_connect={auto_connect})")
        
        if auto_connect:
            self.connect()
    
    def connect(self) -> bool:
        """Connect to MCU.
        
        Returns:
            True if connection successful, False otherwise.
        """
        if self._connected:
            logger.warning("MCU already connected")
            return True
        
        success = self._mcu.connect()
        if success:
            self._connected = True
            logger.info("MCU connected successfully")
        else:
            logger.error("Failed to connect to MCU")
        
        return success
    
    def is_connected(self) -> bool:
        """Check if MCU is connected.
        
        Returns:
            True if connected, False otherwise.
        """
        return self._connected and self._mcu.running
    
    def apply_command(
        self,
        command: JointCommand,
    ) -> None:
        """Apply joint command to MCU.
        
        Converts JointCommand (18 joints in radians) to MCU command format
        (normalized [0.0, 1.0] with joint names) and sends to MCU.
        
        Args:
            command: JointCommand containing 18 joint positions in radians.
            
        Raises:
            RuntimeError: If MCU is not connected.
            ValueError: If command is invalid or has wrong number of joints.
        """
        if not self.is_connected():
            raise RuntimeError("MCU is not connected. Call connect() first.")
        
        # Extract joint positions array from command
        command_array = command.joint_positions
        
        # Validate joint positions shape (should be 18 for hexapod)
        if command_array.shape[0] != 18:
            raise ValueError(
                f"Expected 18 joints, got {command_array.shape[0]} joints"
            )
        
        # Convert radians to normalized [0.0, 1.0] range
        # Formula: normalized = (radians / JOINT_LIMIT_RAD) * 0.5 + 0.5
        # This maps: -JOINT_LIMIT_RAD -> 0.0, 0.0 -> 0.5, +JOINT_LIMIT_RAD -> 1.0
        # TODO: this will be adjusted as it is tested with the actuators.
        normalized_positions = (command_array / JOINT_LIMIT_RAD) * 0.5 + JOINT_NEUTRAL
        
        # Clamp to [0.0, 1.0] range (this is a safety check to ensure the values are within the expected range)
        normalized_positions = np.clip(normalized_positions, 0.0, 1.0)
        
        # Convert to dictionary format expected by firmware SDK
        # Map joint names to normalized positions
        cmds_by_joint = {
            joint_name: float(normalized_positions[i])
            for i, joint_name in enumerate(JOINT_NAMES)
        }
        
        # Send command to MCU
        self._mcu.send_command_joints(cmds_by_joint)
        
        # Log command in MCU-preferred format for debugging
        if logger.isEnabledFor(logging.DEBUG):
            joint_values_str = ", ".join(
                f"{name}={val:.4f}" for name, val in zip(JOINT_NAMES, normalized_positions)
            )
            logger.debug(
                f"KrabbyMCUSDK: Applied joint command (timestamp_ns={command.timestamp_ns}, "
                f"observation_timestamp_ns={command.observation_timestamp_ns}): "
                f"{joint_values_str}"
            )
        
        # Log summary statistics
        logger.debug(
            f"KrabbyMCUSDK: Joint command stats - "
            f"radians: min={command_array.min():.4f}, max={command_array.max():.4f}, "
            f"normalized: min={normalized_positions.min():.4f}, max={normalized_positions.max():.4f}"
        )
    
    def close(self) -> None:
        """Close MCU connection."""
        if self._mcu is not None:
            self._mcu.close()
            self._connected = False
            logger.info("MCU connection closed")
