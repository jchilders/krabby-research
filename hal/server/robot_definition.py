"""Robot definition for HAL: topology and observation scaling.

Contains base classes (ObservationScalingDefinition, RobotDefinition).
Observation dimension for a policy (obs_dim) is computed from robot + model
via model_definition.get_observation_dimensions(robot_definition).obs_dim in compute.parkour.model_definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ObservationScalingDefinition:
    """Observation scaling factors."""

    base_ang_vel: float
    joint_vel: float
    base_lin_vel: float


@dataclass(frozen=True)
class RobotDefinition:
    """Top-level robot definition: topology and observation scaling."""

    name: str
    legs: Tuple[str, ...]
    joint_types: Tuple[str, ...]
    observation_scaling: ObservationScalingDefinition
    mcu_joints: Tuple[str, ...] = ()  # Joint names for the MCU (subset of get_joint_names()); empty if no MCU

    def get_total_joint_count(self) -> int:
        """Total DOF from topology (len(legs) × len(joint_types))."""
        return len(self.legs) * len(self.joint_types)

    def get_joint_names(self) -> Tuple[str, ...]:
        """Ordered joint names for this robot: '{leg}_{joint_type}' per leg, then per joint type.
        Same order as get_joint_index(leg, joint_type) and as policy action/observation vectors.
        Use this tuple (not dict key iteration) whenever converting between dict[str, float] and
        ordered sequences (arrays, packets) so indices stay in sync.
        """
        return tuple(
            f"{leg}_{joint_type}"
            for leg in self.legs
            for joint_type in self.joint_types
        )

    def get_mcu_joints(self) -> Tuple[str, ...]:
        """Joint names for the MCU. Empty if this robot has no MCU mapping."""
        return self.mcu_joints

    def get_observation_joint_count(self) -> int:
        """Observation joint count (matches total joint count)."""
        return self.get_total_joint_count()

    def get_contact_force_count(self) -> int:
        """Contact force count (one per leg)."""
        return len(self.legs)

    def get_joint_index(self, leg_name: str, joint_type: str) -> int:
        """Global joint index for leg/joint type."""
        leg_idx = self.legs.index(leg_name)
        joint_idx = self.joint_types.index(joint_type)
        return leg_idx * len(self.joint_types) + joint_idx

    def get_leg_joint_indices(self, leg_name: str) -> Tuple[int, ...]:
        """All joint indices for a leg."""
        leg_idx = self.legs.index(leg_name)
        start_idx = leg_idx * len(self.joint_types)
        return tuple(range(start_idx, start_idx + len(self.joint_types)))

    def get_num_prop(self) -> int:
        """Proprioceptive feature dimension (12 + 3×joint_count + contact_count)."""
        fixed_features = 12
        joint_features = 3 * self.get_observation_joint_count()
        return fixed_features + joint_features + self.get_contact_force_count()

    def validate(self) -> None:
        """Validate definition consistency."""
        if len(self.legs) == 0:
            raise ValueError("legs must be non-empty")
        if len(self.joint_types) == 0:
            raise ValueError("joint_types must be non-empty")
