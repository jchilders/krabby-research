"""Robot definition: quad (4-legged)."""

from hal.server.robot_definition import (
    ObservationScalingDefinition,
    RobotDefinition,
)

KRABBY_QUAD_DEFINITION = RobotDefinition(
    name="krabby_quad",
    legs=("FL", "FR", "RL", "RR"),
    joint_types=("hip_yaw", "hip_pitch", "knee"),
    observation_scaling=ObservationScalingDefinition(
        base_ang_vel=0.25,
        joint_vel=0.05,
        base_lin_vel=2.0,
    ),
)
