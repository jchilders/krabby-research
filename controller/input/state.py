"""Controller state data structures."""
from dataclasses import dataclass
from enum import Enum


class LegIdentifier(str, Enum):
    """Leg identifier enum for hexapod robot legs.
    
    Using str, Enum allows these values to work as strings while providing type safety and IDE autocomplete.
    """
    FRONT_LEFT = "FL"
    FRONT_RIGHT = "FR"
    MIDDLE_LEFT = "ML"
    MIDDLE_RIGHT = "MR"
    REAR_LEFT = "RL"
    REAR_RIGHT = "RR"


@dataclass
class ControllerState:
    """Normalized controller state with buttons, sticks, and triggers.
    
    All axis values are normalized to [-1.0, 1.0] range.
    Button states are boolean (True when pressed).
    """
    # Buttons
    LT: bool = False  # Left Trigger 
    LB: bool = False  # Left Bumper
    LS: bool = False  # Left Stick button
    RS: bool = False  # Right Stick button
    RT: bool = False  # Right Trigger 
    RB: bool = False  # Right Bumper
    
    # Sticks (normalized to [-1.0, 1.0])
    LX: float = 0.0  # Left stick X
    LY: float = 0.0  # Left stick Y
    RX: float = 0.0  # Right stick X
    RY: float = 0.0  # Right stick Y

