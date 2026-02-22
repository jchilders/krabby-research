"""SDK interface for applying joint commands to the Krabby MCU.

This module provides a standardized SDK interface for applying joint commands
to the Krabby MCU (real hardware on Jetson).
"""

import logging
from typing import Optional

from hal.client.data_structures.hardware import JointCommand
from firmware.krabby_mcu import KrabbyMCUSDK as FirmwareKrabbyMCUSDK

logger = logging.getLogger(__name__)

# Radians -> normalized [0, 1]; 0.5 = neutral. JOINT_LIMIT_RAD used for scaling.
JOINT_LIMIT_RAD = 0.5
JOINT_NEUTRAL = 0.5

# 6 MCU joints (FL/FR legs). Firmware protocol names: LHY, RHY, LHL, LKL, RHL, RKL.
MCU_JOINT_NAMES = ("LHY", "RHY", "LHL", "LKL", "RHL", "RKL")

NEUTRAL_RAD_THRESHOLD = 0.01

PWM_SCALE = 255 / JOINT_LIMIT_RAD


def _rad_to_pwm(rad: float) -> int:
    """Clamp radian-scaled value to PWM in [-255, 255]."""
    return max(-255, min(255, int(round(rad * PWM_SCALE))))


def _map_6_mcu_joints_to_normalized(command: dict[str, float], mcu_joints: tuple[str, ...]) -> dict[str, float]:
    """Map 6 MCU joint positions (radians) from command dict to MCU joint names; return dict name -> normalized [0, 1] float."""
    six_mcu_joint_rad = [command.get(name, 0.0) for name in mcu_joints]
    normalized = [
        max(0.0, min(1.0, (r / JOINT_LIMIT_RAD) * 0.5 + JOINT_NEUTRAL))
        for r in six_mcu_joint_rad
    ]
    return {name: normalized[i] for i, name in enumerate(MCU_JOINT_NAMES)}


def _six_mcu_joint_rad_from_command(command: dict[str, float], mcu_joints: tuple[str, ...]) -> list[float]:
    """Extract 6 MCU joint positions from command dict (radians), in same order as mcu_joints."""
    return [command.get(name, 0.0) for name in mcu_joints]


class KrabbyMCUSDK:
    """SDK for applying 18-joint HAL commands to the MCU.

    Accepts JointCommand; joint order is given by joint_names (e.g. robot_definition.get_joint_names()).
    mcu_joints comes from robot_definition.get_mcu_joints().
    """

    def __init__(
        self,
        mcu_joints: tuple[str, ...],
        port: Optional[str] = None,
        baud: int = 115200,
        auto_connect: bool = True,
    ):
        """Initialize MCU SDK. mcu_joints: joint names for the MCU (e.g. from robot_definition.get_mcu_joints())."""
        if FirmwareKrabbyMCUSDK is None:
            raise RuntimeError(
                "KrabbyMCUSDK firmware module not available. "
                "Cannot initialize MCU SDK without firmware package."
            )
        if len(mcu_joints) != len(MCU_JOINT_NAMES):
            raise ValueError(
                f"mcu_joints must have {len(MCU_JOINT_NAMES)} names (firmware expects {len(MCU_JOINT_NAMES)} joints), got {len(mcu_joints)}"
            )
        
        self._mcu_joints = mcu_joints
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
    
    def _jog_all_joints(self, six_mcu_joint_rad: list[float]) -> None:
        """Send one jog command per MCU joint: 0 if neutral, else PWM. Stops only released axes."""
        for i in range(6):
            name = MCU_JOINT_NAMES[i]
            pwm = 0 if abs(six_mcu_joint_rad[i]) <= NEUTRAL_RAD_THRESHOLD else _rad_to_pwm(six_mcu_joint_rad[i])
            self._mcu.send_command_jog(name, pwm)

    def apply_command(self, command: JointCommand) -> None:
        """Apply joint command to MCU (radians -> normalized, then send).

        Args:
            command: JointCommand with joint_positions, joint_names, and timestamps (order fixed on command).
        """
        if not self.is_connected():
            raise RuntimeError("MCU is not connected. Call connect() first.")

        cmd_dict = command.to_positions_dict()
        for name in self._mcu_joints:
            if name not in cmd_dict:
                raise ValueError(
                    f"command must contain key {name!r} (and all names in mcu_joints)"
                )

        cmds_by_mcu_joint = _map_6_mcu_joints_to_normalized(cmd_dict, self._mcu_joints)
        six_mcu_joint_rad = _six_mcu_joint_rad_from_command(cmd_dict, self._mcu_joints)
        ### TODO: As a pot is not connected for all joints, we do not call self._mcu.send_command_joints(cmds_by_mcu_joint) for now.
        ### Instead, we call self._jog_all_joints(six_mcu_joint_rad) to jog all joints (neutral → 0, non-neutral → PWM).
        self._jog_all_joints(six_mcu_joint_rad)

        # Log applied joint values (MCU joint names, normalized) and timestamps
        joint_values_str = ", ".join(
            f"{name}={val:.4f}" for name, val in cmds_by_mcu_joint.items()
        )
        logger.info(
            f"KrabbyMCUSDK: Applied joint command (from 18 joints, "
            f"timestamp_ns={command.timestamp_ns}, observation_timestamp_ns={command.observation_timestamp_ns}): "
            f"{joint_values_str}"
        )
        rad_vals = list(cmd_dict.values())
        if rad_vals:
            logger.debug(
                f"KrabbyMCUSDK: Joint command stats - "
                f"radians: min={min(rad_vals):.4f}, max={max(rad_vals):.4f}, "
                f"6 MCU joints normalized: min={min(cmds_by_mcu_joint.values()):.4f}, max={max(cmds_by_mcu_joint.values()):.4f}"
            )
    
    def close(self) -> None:
        """Close MCU connection and release resources."""
        self._mcu.close()
        self._connected = False
        logger.info("MCU connection closed")
