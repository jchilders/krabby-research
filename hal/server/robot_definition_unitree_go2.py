"""Robot definition: Unitree Go2 (quad, parkour-compatible num_prop=53)."""

from hal.server.robot_definition import (
    ObservationScalingDefinition,
    RobotDefinition,
)


class UnitreeGo2RobotDefinition(RobotDefinition):
    """Unitree Go2 topology with parkour-compatible proprioceptive dimension."""

    def get_num_prop(self) -> int:
        # Matches teacher/student parkour checkpoints trained with num_prop=53.
        return 53


UNITREE_GO2_DEFINITION = UnitreeGo2RobotDefinition(
    name="unitree_go2",
    legs=("FL", "FR", "RL", "RR"),
    joint_types=("hip_yaw", "hip_pitch", "knee"),
    observation_scaling=ObservationScalingDefinition(
        base_ang_vel=0.25,
        joint_vel=0.05,
        base_lin_vel=2.0,
    ),
)
