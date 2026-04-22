"""Tests for MaixSense-A075V decode helpers and optional live HTTP fetch."""

import os

import numpy as np
import pytest

from hal.server.jetson.maixsense_a075v import (
    A075V_UINT16_RAW_TO_METERS,
    MaixSenseA075VClient,
    a075v_depth_raw_to_meters,
    a075v_set_cfg_bytes_hal,
    frame_config_decode,
    frame_config_encode,
    peek_frame_config_from_getdeep_body,
    rgb_resolution_from_frame_config,
)


def test_frame_config_roundtrip() -> None:
    cfg = frame_config_encode(1, 1, 255, 0, 2, 7, 1, 0, 0)
    assert len(cfg) == 12
    assert frame_config_decode(cfg) == (1, 1, 255, 0, 2, 7, 1, 0, 0)


def test_a075v_hal_set_cfg_matches_tutorial() -> None:
    assert a075v_set_cfg_bytes_hal() == frame_config_encode(1, 0, 255, 0, 2, 7, 1, 0, 0)
    assert frame_config_decode(a075v_set_cfg_bytes_hal()) == (1, 0, 255, 0, 2, 7, 1, 0, 0)


def test_a075v_depth_to_meters_uint16() -> None:
    z = np.array([[4000, 0], [0, 2000]], dtype=np.uint16)
    m = a075v_depth_raw_to_meters(z)
    assert m.dtype == np.float32
    assert np.isclose(m[0, 0], 1.0)
    assert m[0, 0] == 4000.0 * A075V_UINT16_RAW_TO_METERS
    assert np.isclose(m[1, 1], 0.5)


def test_a075v_depth_to_meters_uint8_nonlinear() -> None:
    u8 = np.array([[0, 255]], dtype=np.uint8)
    m = a075v_depth_raw_to_meters(u8)
    mm0 = 0.0
    mm1 = (255.0 / 5.1) ** 2
    assert float(m[0, 0]) == pytest.approx(mm0 / 1000.0)
    assert float(m[0, 1]) == pytest.approx(mm1 / 1000.0)


def test_rgb_resolution_hint_and_peek() -> None:
    cfg = frame_config_encode(1, 1, 255, 0, 2, 7, 1, 0, 0)
    assert rgb_resolution_from_frame_config(frame_config_decode(cfg)) == (600, 800)
    header = b"\x00" * 16 + cfg + b"\x00\x00\x00\x00"
    assert peek_frame_config_from_getdeep_body(header + b"x") == frame_config_decode(cfg)


@pytest.mark.jetson
def test_maixsense_live_http_rgb_not_blank() -> None:
    """Hit the device over HTTP; RGB must look like a real frame (not flat / empty).

    Requires ``RUN_JETSON_MAIXSENSE_HW_TEST=1``, ``KRABBY_MAIXSENSE_LIVE_TEST_HOST``, and
    ``requests`` + OpenCV (JPEG decode). Optional ``KRABBY_MAIXSENSE_LIVE_TEST_PORT`` (default 80).
    """
    if os.environ.get("RUN_JETSON_MAIXSENSE_HW_TEST") != "1":
        pytest.skip("Set RUN_JETSON_MAIXSENSE_HW_TEST=1 for live MaixSense HTTP test")
    host = os.environ.get("KRABBY_MAIXSENSE_LIVE_TEST_HOST", "").strip()
    if not host:
        pytest.skip("Set KRABBY_MAIXSENSE_LIVE_TEST_HOST to the camera IP")
    pytest.importorskip("requests")
    pytest.importorskip("cv2")
    try:
        port = int(
            os.environ.get("KRABBY_MAIXSENSE_LIVE_TEST_PORT", "80").strip() or "80"
        )
    except ValueError:
        pytest.fail("KRABBY_MAIXSENSE_LIVE_TEST_PORT must be an integer")
    client = MaixSenseA075VClient(host=host, port=port)
    client.post_encode_config()
    frame = client.fetch_decoded()
    assert frame.rgb is not None, "decoded frame has no RGB"
    rgb = np.asarray(frame.rgb, dtype=np.float64)
    assert float(np.std(rgb)) > 4.0, "MaixSense RGB looks flat (blank or solid fill)"
    nonzero_frac = float(np.count_nonzero(rgb)) / float(rgb.size)
    assert nonzero_frac > 0.01, "MaixSense RGB is nearly all zeros"
