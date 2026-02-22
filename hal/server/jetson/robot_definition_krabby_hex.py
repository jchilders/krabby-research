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
    # 6 joints the MCU drives (FL/FR): names match firmware slots LHY, RHY, LHL, LKL, RHL, RKL
    mcu_joints=(
        "FL_hip_yaw", "FR_hip_yaw", "FL_hip_pitch", "FL_knee", "FR_hip_pitch", "FR_knee",
    ),
)
