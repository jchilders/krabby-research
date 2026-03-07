"""Test helpers for creating dummy hardware data structures."""

import time
from typing import Optional

import numpy as np

from hal.client.data_structures.hardware import (
    HardwareObservations,
    JointCommand,
)
from hal.server.robot_definition import (
    ObservationScalingDefinition,
    RobotDefinition,
)

# Test-only robot definition (quad, 12 joints). Avoids importing hal.server.isaac
# so hal-server publish tests can run without torch.
TEST_QUAD_DEFINITION = RobotDefinition(
    name="test_quad",
    legs=("FL", "FR", "RL", "RR"),
    joint_types=("hip_yaw", "hip_pitch", "knee"),
    observation_scaling=ObservationScalingDefinition(
        base_ang_vel=0.25,
        joint_vel=0.05,
        base_lin_vel=2.0,
    ),
)

# Test-only robot definition (hex, 18 joints). Avoids importing hal.server.jetson
# so hal-server publish tests can run without ZED/camera/compute deps.
TEST_HEX_DEFINITION = RobotDefinition(
    name="test_hex",
    legs=("FL", "FR", "ML", "MR", "RL", "RR"),
    joint_types=("hip_yaw", "hip_pitch", "knee"),
    observation_scaling=ObservationScalingDefinition(
        base_ang_vel=0.25,
        joint_vel=0.05,
        base_lin_vel=2.0,
    ),
)


def create_dummy_hw_obs(
    camera_height: int = 480,
    camera_width: int = 640,
    timestamp_ns: Optional[int] = None,
) -> HardwareObservations:
    """Create dummy hardware observations for testing.
    
    Args:
        camera_height: Height of camera images (default 480)
        camera_width: Width of camera images (default 640)
        timestamp_ns: Optional timestamp (defaults to current time)
    
    Returns:
        HardwareObservations with dummy data
    """
    if timestamp_ns is None:
        timestamp_ns = time.time_ns()
    
    return HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        camera_height=camera_height,
        camera_width=camera_width,
        timestamp_ns=timestamp_ns,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # Identity quaternion
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
    )


def create_dummy_joint_positions(
    timestamp_ns: Optional[int] = None,
) -> JointCommand:
    """Create dummy joint command for testing.
    
    Args:
        timestamp_ns: Optional timestamp (defaults to current time)
    
    Returns:
        JointCommand with dummy data
    """
    if timestamp_ns is None:
        timestamp_ns = time.time_ns()
    
    return JointCommand(
        _joint_positions=np.zeros(12, dtype=np.float32),
        timestamp_ns=timestamp_ns,
        observation_timestamp_ns=timestamp_ns,
        joint_names=TEST_QUAD_DEFINITION.get_joint_names(),
    )

