"""Math utilities for parkour inference.

This module provides math functions needed for parkour inference that are
independent of Isaac Lab/Isaac Sim, allowing the code to run on hardware
deployments without simulation dependencies.
"""

import torch


def euler_xyz_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternion to Euler angles (roll, pitch, yaw) in XYZ order.
    
    This matches the implementation from isaaclab.utils.math for compatibility.
    
    Args:
        quat: Quaternion tensor of shape (..., 4) in (w, x, y, z) format
        
    Returns:
        Tuple of (roll, pitch, yaw) tensors in radians
    """
    # Ensure quat is in (w, x, y, z) format
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion must have 4 components, got shape {quat.shape}")
    
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Clamp to avoid NaN
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi] range.
    
    This matches the implementation from isaaclab.utils.math for compatibility.
    
    Args:
        angle: Angle tensor in radians
        
    Returns:
        Wrapped angle tensor in [-pi, pi] range
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))
