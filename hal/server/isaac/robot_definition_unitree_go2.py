"""Robot definition for Unitree Go2 (matches parkour teacher checkpoint: num_prop=53)."""

from hal.server.robot_definition import (
    ObservationScalingDefinition,
    RobotDefinition,
)


class UnitreeGo2RobotDefinition(RobotDefinition):
    """Unitree Go2: same topology as quad but num_prop=53 to match parkour_rl_cfg training."""

    def get_num_prop(self) -> int:
        return 53  # ParkourRslRlBaseCfg num_prop: 3+2+3+4+36+5


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
