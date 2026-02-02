"""SDK interface for applying joint commands to the Krabby MCU."""

import logging
from typing import Optional
import numpy as np
from hal.client.data_structures.hardware import JointCommand
from firmware.krabby_mcu import KrabbyMCUSDK as FirmwareKrabbyMCUSDK

logger = logging.getLogger(__name__)

# 18 joints.
JOINT_NAMES = [
    "FL_hip_yaw", "FL_hip_pitch", "FL_knee",
    "FR_hip_yaw", "FR_hip_pitch", "FR_knee",
    "ML_hip_yaw", "ML_hip_pitch", "ML_knee",
    "MR_hip_yaw", "MR_hip_pitch", "MR_knee",
    "RL_hip_yaw", "RL_hip_pitch", "RL_knee",
    "RR_hip_yaw", "RR_hip_pitch", "RR_knee",
]

# Radians → normalized [0, 1]; 0.5 = neutral. JOINT_LIMIT_RAD used for scaling.
JOINT_LIMIT_RAD = 0.5
JOINT_NEUTRAL = 0.5

# 6 MCU joints (FL/FR legs).
MCU_JOINT_NAMES = ("LHY", "RHY", "LHL", "LKL", "RHL", "RKL")

NEUTRAL_RAD_THRESHOLD = 0.01

PWM_SCALE = 255 / JOINT_LIMIT_RAD


def _rad_to_pwm(rad: float) -> int:
    """Clamp radian-scaled value to PWM in [-255, 255]."""
    return max(-255, min(255, int(round(rad * PWM_SCALE))))


def _map_18_joints_to_6_mcu_joints(joint_positions_rad: np.ndarray) -> dict[str, float]:
    """Map 18-element HAL joint positions (radians) to 6 MCU joint names; returns dict of name → normalized [0, 1] float.
    Only the first 6 joints are used.
    TODO: Add support for the other 12 joints. These will be added in the future.
    """
    six_rad = np.array([
        joint_positions_rad[0], joint_positions_rad[3],
        joint_positions_rad[1], joint_positions_rad[2],
        joint_positions_rad[4], joint_positions_rad[5],
    ], dtype=np.float32)
    normalized = (six_rad / JOINT_LIMIT_RAD) * 0.5 + JOINT_NEUTRAL
    normalized = np.clip(normalized, 0.0, 1.0)
    return {name: float(normalized[i]) for i, name in enumerate(MCU_JOINT_NAMES)}


class KrabbyMCUSDK:
    """SDK for applying 18-joint HAL commands to the MCU."""

    def __init__(self, port: Optional[str] = None, baud: int = 115200, auto_connect: bool = True):
        """Initialize MCU SDK (port, baud, optional auto_connect)."""
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
        """Connect to MCU. Returns True if successful."""
        if self._connected:
            logger.warning("MCU already connected")
            return True
        
        success = self._mcu.connect()
        if success:
            self._connected = True
            logger.info("MCU connected successfully")
        else:
            logger.error("Failed to connect to MCU")
            raise RuntimeError("Failed to connect to MCU")
        
        return success
    
    def is_connected(self) -> bool:
        """Return whether MCU is connected."""
        return self._connected and self._mcu.running

    def _jog_all_joints(self, six_rad: np.ndarray) -> None:
        """Send one jog command per MCU joint: 0 if neutral, else PWM. Stops only released axes."""
        for i in range(6):
            name = MCU_JOINT_NAMES[i]
            pwm = 0 if abs(six_rad[i]) <= NEUTRAL_RAD_THRESHOLD else _rad_to_pwm(six_rad[i])
            self._mcu.send_command_jog(name, pwm)

    def apply_command(
        self,
        command: JointCommand,
    ) -> None:
        """Apply 18-joint command to MCU (radians → normalized, then send)."""
        if not self.is_connected():
            raise RuntimeError("MCU is not connected. Call connect() first.")

        command_array = command.joint_positions
        if command_array.shape[0] != 18:
            raise ValueError(
                f"Expected 18 joints, got {command_array.shape[0]} joints"
            )

        cmds_by_joint = _map_18_joints_to_6_mcu_joints(command_array)
        six_rad = np.array([
            command_array[0], command_array[3],
            command_array[1], command_array[2],
            command_array[4], command_array[5],
        ], dtype=np.float32)
        ### TODO: As a pot is not connected for all joints, we do not call self._mcu.send_command_joints(cmds_by_joint) for now.
        ### Instead, we call self._jog_all_joints(six_rad) to jog all joints (neutral → 0, non-neutral → PWM).
        self._jog_all_joints(six_rad)

        joint_values_str = ", ".join(
            f"{name}={val:.4f}" for name, val in cmds_by_joint.items()
        )
        logger.info(
            f"KrabbyMCUSDK: Applied joint command (from 18 joints, timestamp_ns={command.timestamp_ns}, "
            f"observation_timestamp_ns={command.observation_timestamp_ns}): "
            f"{joint_values_str}"
        )
        logger.debug(
            f"KrabbyMCUSDK: Joint command stats - "
            f"radians: min={command_array.min():.4f}, max={command_array.max():.4f}, "
            f"6-joint normalized: min={min(cmds_by_joint.values()):.4f}, max={max(cmds_by_joint.values()):.4f}"
        )
    
    def close(self) -> None:
        """Close MCU connection and release resources."""
        self._mcu.close()
        self._connected = False
        logger.info("MCU connection closed")