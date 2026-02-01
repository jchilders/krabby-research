"""HAL observation types.

Generic HAL types for robot control.
"""

import time
from dataclasses import dataclass


@dataclass
class NavigationCommand:
    """Navigation command for robot movement.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        vx: Forward velocity (m/s)
        vy: Lateral velocity (m/s)
        yaw_rate: Angular velocity (rad/s)
    """

    timestamp_ns: int
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate navigation command."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")

    @classmethod
    def create_now(cls, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0) -> "NavigationCommand":
        """Create navigation command with current timestamp."""
        return cls(
            timestamp_ns=time.time_ns(),
            vx=vx,
            vy=vy,
            yaw_rate=yaw_rate,
        )
