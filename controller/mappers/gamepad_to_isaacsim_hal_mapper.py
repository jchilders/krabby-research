"""Mapper from gamepad control data to IsaacSim HAL command format.

This mapper converts GamepadControlData (leg selections and axis values)
into JointCommand format compatible with IsaacSim HAL server.
"""

import logging
import time
from typing import Optional

import numpy as np

from controller.input.state import GamepadControlData, LegIdentifier
from hal.client.data_structures.hardware import JointCommand

logger = logging.getLogger(__name__)

# Joint ordering: 18 joints total (6 legs × 3 DOF per leg: hip_yaw, hip_pitch, knee)
# Order: FL, FR, ML, MR, RL, RR (each leg has 3 joints: hip_yaw, hip_pitch, knee)
# Joint indices per leg: hip_yaw, hip_pitch, knee
# FL: 0,1,2; FR: 3,4,5; ML: 6,7,8; MR: 9,10,11; RL: 12,13,14; RR: 15,16,17

# Leg to joint index mapping (3 joints per leg: hip_yaw, hip_pitch, knee)
LEG_TO_JOINT_INDICES = {
    LegIdentifier.FRONT_LEFT: (0, 1, 2),      # hip_yaw, hip_pitch, knee
    LegIdentifier.FRONT_RIGHT: (3, 4, 5),    # hip_yaw, hip_pitch, knee
    LegIdentifier.MIDDLE_LEFT: (6, 7, 8),     # hip_yaw, hip_pitch, knee
    LegIdentifier.MIDDLE_RIGHT: (9, 10, 11),  # hip_yaw, hip_pitch, knee
    LegIdentifier.REAR_LEFT: (12, 13, 14),    # hip_yaw, hip_pitch, knee
    LegIdentifier.REAR_RIGHT: (15, 16, 17),   # hip_yaw, hip_pitch, knee
}

# Default scaling factors (radians per unit of normalized input)
# These can be configured per mapper instance
DEFAULT_HIP_UP_DOWN_SCALE = 0.3  # radians per unit (max ~0.3 rad = ~17 degrees)
DEFAULT_KNEE_OUT_IN_SCALE = 0.3  # radians per unit
DEFAULT_HIP_YAW_SCALE = 0.2      # radians per unit (yaw is typically smaller range)


class GamepadToIsaacSimHALMapper:
    """Maps gamepad control data to IsaacSim HAL joint commands.
    
    Converts GamepadControlData (selected legs and axis values) into
    JointCommand format with normalized joint targets/speeds.
    
    The mapper applies axis scaling and maps leg selections to joint indices.
    Joint commands are relative to current positions (incremental control).
    
    Zero-copy guarantees:
    - Joint positions array is always newly created (not a view)
    - Timestamp is always copied (scalar)
    """
    
    def __init__(
        self,
        hip_up_down_scale: float = DEFAULT_HIP_UP_DOWN_SCALE,
        knee_out_in_scale: float = DEFAULT_KNEE_OUT_IN_SCALE,
        hip_yaw_scale: float = DEFAULT_HIP_YAW_SCALE,
    ):
        """Initialize the mapper.
        
        Args:
            hip_up_down_scale: Scaling factor for hip up/down axis (radians per unit).
                Default: 0.3 rad (~17 degrees max deflection).
            knee_out_in_scale: Scaling factor for knee out/in axis (radians per unit).
                Default: 0.3 rad (~17 degrees max deflection).
            hip_yaw_scale: Scaling factor for hip yaw axis (radians per unit).
                Default: 0.2 rad (~11 degrees max deflection).
        """
        self.hip_up_down_scale = hip_up_down_scale
        self.knee_out_in_scale = knee_out_in_scale
        self.hip_yaw_scale = hip_yaw_scale
        
        # Store last joint positions for incremental control
        # Initialize to zeros (neutral position) - 18 joints for hexapod
        # TODO: This is a placeholder for the actual joint positions. This may be replaced with the actual joint positions in the future.
        self._last_joint_positions = np.zeros(18, dtype=np.float32)
    
    def map(
        self,
        control_data: GamepadControlData,
        observation_timestamp_ns: Optional[int] = None,
    ) -> JointCommand:
        """Map gamepad control data to joint command.
        
        Args:
            control_data: GamepadControlData with selected legs and axis values.
            observation_timestamp_ns: Optional timestamp of the observation this
                command responds to. If None, uses current time.
                
        Returns:
            JointCommand with joint positions for selected legs.
            
        Raises:
            ValueError: If control data is invalid.
        """
        if not isinstance(control_data, GamepadControlData):
            raise ValueError(f"control_data must be GamepadControlData, got {type(control_data)}")
        
        # Start with last joint positions (incremental control)
        joint_positions = self._last_joint_positions.copy()
        
        # Apply control to selected legs
        if control_data.selected_legs:
            for leg in control_data.selected_legs:
                if leg in LEG_TO_JOINT_INDICES:
                    hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[leg]
                    
                    # Apply hip up/down (hip_pitch joint)
                    # Positive hip_up_down = up = positive joint angle
                    hip_delta = control_data.hip_up_down * self.hip_up_down_scale
                    joint_positions[hip_pitch_idx] += hip_delta
                    
                    # Apply knee out/in (knee joint)
                    # Positive knee_out_in = out = positive joint angle
                    knee_delta = control_data.knee_out_in * self.knee_out_in_scale
                    joint_positions[knee_idx] += knee_delta
                    
                    # Apply hip yaw (hip_yaw joint)
                    # Positive hip_yaw = forward/back rotation
                    hip_yaw_delta = control_data.hip_yaw * self.hip_yaw_scale
                    joint_positions[hip_yaw_idx] += hip_yaw_delta
                    
                    logger.debug(
                        f"Leg {leg.value}: hip_yaw_idx={hip_yaw_idx} delta={hip_yaw_delta:.3f}, "
                        f"hip_pitch_idx={hip_pitch_idx} delta={hip_delta:.3f}, "
                        f"knee_idx={knee_idx} delta={knee_delta:.3f}"
                    )
                else:
                    logger.warning(f"Unknown leg identifier: {leg}")
        else:
            # No legs selected, maintain current positions (no change)
            logger.debug("No legs selected, maintaining current joint positions")
        
        # Clamp joint positions to reasonable limits
        # Typical joint limits: hip_pitch [-1.0, 1.0] rad, knee [-2.0, 0.0] rad
        # Clamp to safe ranges
        # TODO: This may be replaced by limits per joint type in the future. E.g. knee limits are [-2.0, 0.0] rad, but hip_pitch limits are [-1.0, 1.0] rad.
        joint_positions = np.clip(joint_positions, -2.0, 2.0)
        
        # Update last joint positions for next iteration
        self._last_joint_positions = joint_positions.copy()
        
        # Create timestamps
        # JointCommand requires timestamps for:
        # 1. timestamp_ns: When this command was created (used for ordering and latency tracking)
        # 2. observation_timestamp_ns: Timestamp of the observation this command responds to
        #    (used for tracking command-observation relationships and measuring round-trip latency)
        # These timestamps enable the HAL system to:
        # - Match commands to their corresponding observations
        # - Measure end-to-end latency (observation → command → application)
        # - Debug timing issues and ensure proper synchronization
        current_timestamp_ns = time.time_ns()
        if observation_timestamp_ns is None:
            # If no observation timestamp provided, use current time
            # (In a full implementation, this would track the last received observation timestamp)
            observation_timestamp_ns = current_timestamp_ns
        
        # Create joint command
        joint_cmd = JointCommand(
            joint_positions=joint_positions,
            timestamp_ns=current_timestamp_ns,
            observation_timestamp_ns=observation_timestamp_ns,
        )
        
        logger.debug(
            f"Mapped gamepad control: {len(control_data.selected_legs)} legs selected, "
            f"joint_positions range=[{joint_positions.min():.3f}, {joint_positions.max():.3f}]"
        )
        
        return joint_cmd
    
    def reset(self) -> None:
        """Reset mapper state (reset last joint positions to zero)."""
        self._last_joint_positions = np.zeros(18, dtype=np.float32)
        logger.debug("GamepadToIsaacSimHALMapper reset")
