"""Convert a depth map (meters, H×W float32) into the policy scan feature vector.

Used by :class:`~hal.server.jetson.hal_server.JetsonHalServer` so all RGB-D sources
produce comparable policy scan vectors of length ``depth_feature_dim``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def validate_depth_frame(depth_frame: np.ndarray) -> bool:
    """Return True if ``depth_frame`` looks usable for feature extraction."""
    if depth_frame is None:
        return False
    if depth_frame.ndim != 2:
        logger.error("Invalid depth frame dimensions: %s, expected 2", depth_frame.ndim)
        return False
    if np.any(np.isnan(depth_frame)) or np.any(np.isinf(depth_frame)):
        logger.warning("Depth frame contains NaN or Inf values")
        return False
    valid_mask = (depth_frame > 0.1) & (depth_frame < 10.0)
    valid_ratio = float(np.sum(valid_mask)) / float(depth_frame.size)
    if valid_ratio < 0.5:
        logger.warning("Low valid depth ratio: %.2f%%", valid_ratio * 100.0)
        return False
    return True


def extract_depth_features_from_map(
    depth_frame: np.ndarray, depth_feature_dim: int
) -> np.ndarray:
    """Grid-sample depth (meters) into ``depth_feature_dim`` features (same layout as ZED path)."""
    height, width = depth_frame.shape
    num_features = depth_feature_dim
    grid_rows = int(np.sqrt(num_features))
    grid_cols = (num_features + grid_rows - 1) // grid_rows
    features = np.zeros(num_features, dtype=np.float32)
    row_indices = np.linspace(0, height - 1, grid_rows, dtype=np.int32)
    col_indices = np.linspace(0, width - 1, grid_cols, dtype=np.int32)
    idx = 0
    for row in row_indices:
        for col in col_indices:
            if idx < num_features:
                depth_value = depth_frame[row, col]
                height_measurement = depth_value - 0.3
                features[idx] = np.clip(height_measurement, -1.0, 1.0)
                idx += 1
    if idx < num_features:
        features[idx:] = 0.0
    return features.astype(np.float32)
