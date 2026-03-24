"""Shared depth-map → scan feature path (ZED / MaixSense)."""

import numpy as np

from hal.server.jetson.depth_scan_features import (
    extract_depth_features_from_map,
    validate_depth_frame,
)


def test_validate_depth_frame_rejects_none_and_invalid_shape():
    assert validate_depth_frame(None) is False
    assert validate_depth_frame(np.zeros((10,), dtype=np.float32)) is False


def test_validate_depth_frame_accepts_reasonable_map():
    d = np.full((32, 40), 1.5, dtype=np.float32)
    assert validate_depth_frame(d) is True


def test_extract_depth_features_from_map_shape_132():
    d = np.full((48, 64), 1.5, dtype=np.float32)
    f = extract_depth_features_from_map(d, 132)
    assert f.shape == (132,)
    assert f.dtype == np.float32
