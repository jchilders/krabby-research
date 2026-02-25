"""Robot definition: hex (6-legged)."""

from hal.server.robot_definition import (
    ObservationScalingDefinition,
    RobotDefinition,
)

KRABBY_HEX_DEFINITION = RobotDefinition(
    name="krabby_hex",
    legs=("FL", "FR", "ML", "MR", "RL", "RR"),
    joint_types=("hip_yaw", "hip_pitch", "knee"),
    observation_scaling=ObservationScalingDefinition(
        base_ang_vel=0.25,
        joint_vel=0.05,
        base_lin_vel=2.0,
    ),
    mcu_joints=(
        "FL_hip_yaw", "FL_hip_pitch", "FL_knee",
        "FR_hip_yaw", "FR_hip_pitch", "FR_knee",
        "ML_hip_yaw", "ML_hip_pitch", "ML_knee",
        "MR_hip_yaw", "MR_hip_pitch", "MR_knee",
        "RL_hip_yaw", "RL_hip_pitch", "RL_knee",
        "RR_hip_yaw", "RR_hip_pitch", "RR_knee",
    ),
)
