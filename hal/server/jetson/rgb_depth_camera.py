"""Protocol for cameras that provide RGB, depth, and depth-derived scan features.

Use this interface for any camera (front, left, or otherwise) that supplies
RGB frames, depth frames, and policy-style depth features. Implementations
include ZedCamera (ZED SDK); other depth cameras can implement this protocol
without depending on ZED.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class RgbDepthCamera(Protocol):
    """Protocol for cameras that provide RGB, depth, and depth-derived features.

    Implementations must provide:
    - get_camera_frames(): one capture returning (rgb, depth) in standard formats
    - get_depth_features(): scan-like feature vector for the policy
    - is_ready(): whether the camera is initialized and can capture
    - close(): release resources
    """

    def get_camera_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture one frame and return (RGB, depth).

        Returns:
            (rgb, depth): rgb (H, W, 3) uint8, depth (H, W) float32 in meters;
            either may be None on capture failure.
        """
        ...

    def get_depth_features(self) -> Optional[np.ndarray]:
        """Return depth-derived scan features (e.g. 132-dim for policy).

        Returns:
            float32 array of shape (depth_feature_dim,) or None on failure.
        """
        ...

    def is_ready(self) -> bool:
        """Return True if the camera is initialized and ready to capture."""
        ...

    def close(self) -> None:
        """Release camera resources."""
        ...
