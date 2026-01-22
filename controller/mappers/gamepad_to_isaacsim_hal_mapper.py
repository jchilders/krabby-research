"""Mapper from gamepad controller state to HAL command format (for IsaacSim).

This mapper encapsulates the complete mapping from:
- Input: Gamepad controller state (ControllerState)
- Robot embodiment: Hexapod (6 legs, 3 DOF per leg = 18 joints)
- Input control type: Gamepad
- Output: HAL JointCommand format (used with IsaacSim HAL server)

The mapper handles leg selection, axis mapping, and joint position calculation,
outputting absolute joint positions in JointCommand format.
"""

import logging
import time
from typing import Optional, Set

import numpy as np

from controller.input.state import ControllerState, LegIdentifier
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
    """Maps gamepad controller state to IsaacSim HAL joint commands.
    
    This mapper encapsulates the complete transformation from raw gamepad input
    to joint commands for a hexapod robot in IsaacSim:
    
    - Robot embodiment: Hexapod (6 legs × 3 DOF = 18 joints)
    - Input control type: Gamepad
    - Output environment: IsaacSim
    
    The mapper handles:
    1. Leg selection based on button combinations
    2. Axis mapping (sticks → joint control axes)
    3. Joint position calculation (absolute positions)
    
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
    
    def _select_legs(self, state: ControllerState) -> Set[LegIdentifier]:
        """Select legs based on controller button state.
        
        Leg selection rules:
        - LT (without LB): Select Front Left (FL)
        - LB (without LT): Select Rear Left (RL)
        - LS: Select Left Middle (ML)
        - RS: Select Right Middle (MR)
        - RT (without RB): Select Front Right (FR)
        - RB (without RT): Select Rear Right (RR)
        - LT + LB: Select FL, RL, MR (tripod combo left)
        - RT + RB: Select FR, RR, ML (tripod combo right)
        
        Args:
            state: Controller state with button presses.
            
        Returns:
            Set of selected leg identifiers.
        """
        # Determine leg selections
        select_FL = state.LT and not state.LB
        select_RL = state.LB and not state.LT
        select_ML = state.LS
        select_MR = state.RS
        select_FR = state.RT and not state.RB
        select_RR = state.RB and not state.RT
        
        # Combo triggers
        combo_left = state.LT and state.LB  # FL/RL/MR
        combo_right = state.RT and state.RB  # FR/RR/ML
        
        # Build selected legs set
        legs = set()
        if combo_left:
            legs |= {LegIdentifier.FRONT_LEFT, LegIdentifier.REAR_LEFT, LegIdentifier.MIDDLE_RIGHT}
        if combo_right:
            legs |= {LegIdentifier.FRONT_RIGHT, LegIdentifier.REAR_RIGHT, LegIdentifier.MIDDLE_LEFT}
        if not combo_left and not combo_right:
            if select_FL:
                legs.add(LegIdentifier.FRONT_LEFT)
            if select_RL:
                legs.add(LegIdentifier.REAR_LEFT)
            if select_ML:
                legs.add(LegIdentifier.MIDDLE_LEFT)
            if select_MR:
                legs.add(LegIdentifier.MIDDLE_RIGHT)
            if select_FR:
                legs.add(LegIdentifier.FRONT_RIGHT)
            if select_RR:
                legs.add(LegIdentifier.REAR_RIGHT)
        
        return legs
    
    def _map_axes(self, state: ControllerState) -> tuple[float, float, float]:
        """Map controller stick axes to control axes.
        
        Axis mappings:
        - Left stick Y: Hip up/down (inverted: -LY, so up = positive)
        - Left stick X: Knee out/in (direct: LX)
        - Right stick Y: Hip yaw forward/back (direct: RY)
        
        Args:
            state: Controller state with stick values.
            
        Returns:
            Tuple of (hip_up_down, knee_out_in, hip_yaw) axis values.
        """
        hip_up_down = -state.LY  # Invert Y axis (up = positive)
        knee_out_in = state.LX
        hip_yaw = state.RY
        
        return hip_up_down, knee_out_in, hip_yaw
    
    def map(
        self,
        state: ControllerState,
        observation_timestamp_ns: Optional[int] = None,
    ) -> JointCommand:
        """Map gamepad controller state to joint command.
        
        This method encapsulates the complete mapping from raw gamepad input
        to absolute joint positions for a hexapod robot in IsaacSim.
        
        Args:
            state: ControllerState with button and stick values.
            observation_timestamp_ns: Optional timestamp of the observation this
                command responds to. If None, uses current time.
                
        Returns:
            JointCommand with absolute joint positions.
            
        Raises:
            ValueError: If state is invalid.
        """
        if not isinstance(state, ControllerState):
            raise ValueError(f"state must be ControllerState, got {type(state)}")
        
        # Select legs based on button state
        selected_legs = self._select_legs(state)
        
        # Map stick axes to control axes
        hip_up_down, knee_out_in, hip_yaw = self._map_axes(state)
        
        # Start from neutral position (all joints at 0.0)
        # TODO: In the future, starting joint positions will be available from the robot state/observations
        joint_positions = np.zeros(18, dtype=np.float32)
        
        # Apply control to selected legs (absolute positions)
        if selected_legs:
            for leg in selected_legs:
                if leg in LEG_TO_JOINT_INDICES:
                    hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[leg]
                    
                    # Apply hip up/down (hip_pitch joint)
                    # Positive hip_up_down = up = positive joint angle
                    joint_positions[hip_pitch_idx] = hip_up_down * self.hip_up_down_scale
                    
                    # Apply knee out/in (knee joint)
                    # Positive knee_out_in = out = positive joint angle
                    joint_positions[knee_idx] = knee_out_in * self.knee_out_in_scale
                    
                    # Apply hip yaw (hip_yaw joint)
                    # Positive hip_yaw = forward/back rotation
                    joint_positions[hip_yaw_idx] = hip_yaw * self.hip_yaw_scale
                    
                    logger.debug(
                        f"Leg {leg.value}: hip_yaw_idx={hip_yaw_idx} pos={joint_positions[hip_yaw_idx]:.3f}, "
                        f"hip_pitch_idx={hip_pitch_idx} pos={joint_positions[hip_pitch_idx]:.3f}, "
                        f"knee_idx={knee_idx} pos={joint_positions[knee_idx]:.3f}"
                    )
                else:
                    logger.warning(f"Unknown leg identifier: {leg}")
        else:
            # No legs selected, all joints remain at 0.0 (neutral position)
            logger.debug("No legs selected, all joints at neutral position")
        
        
        # Create timestamps
        # JointCommand requires timestamps for:
        # 1. timestamp_ns: When this command was created (used for ordering and latency tracking)
        # 2. observation_timestamp_ns: Timestamp of the observation this command responds to
        #    (used for tracking command-observation relationships and measuring round-trip latency)
        # These timestamps enable the HAL system to:
        # - Match commands to their corresponding observations
        # - Measure round-trip latency (observation → command)
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
            f"Mapped gamepad state: {len(selected_legs)} legs selected, "
            f"joint_positions range=[{joint_positions.min():.3f}, {joint_positions.max():.3f}]"
        )
        
        return joint_cmd
