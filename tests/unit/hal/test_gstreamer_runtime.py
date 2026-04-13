"""Tests for ``hal.server.gstreamer_runtime`` (§3.1 shared Gst appsrc path)."""

from __future__ import annotations

import pytest

from hal.server.gstreamer_runtime import (
    AppSrcPipelineResult,
    build_software_appsrc_encode_pipeline_string,
    float32_depth_to_gray16_le,
    run_pipeline_with_appsrc_sync,
    smoke_from_sensor_interface,
)


def test_build_software_appsrc_encode_pipeline_string_h264() -> None:
    s = build_software_appsrc_encode_pipeline_string(
        320, 240, 15, encoding="h264", output_element="fakesink"
    )
    assert "appsrc name=src" in s
    assert "width=320" in s and "height=240" in s
    assert "framerate=15/1" in s
    assert "x264enc" in s and "h264parse" in s
    assert s.endswith("fakesink")


def test_build_software_appsrc_encode_pipeline_string_h265() -> None:
    s = build_software_appsrc_encode_pipeline_string(
        64, 48, 30, encoding="h265", output_element="fakesink"
    )
    assert "x265enc" in s and "h265parse" in s


def test_build_software_appsrc_encode_pipeline_string_raw() -> None:
    s = build_software_appsrc_encode_pipeline_string(
        80, 60, 10, encoding="raw", output_element="fakesink"
    )
    assert "videoconvert" in s
    assert "x264enc" not in s


def test_build_software_appsrc_encode_pipeline_string_gray16_le() -> None:
    s = build_software_appsrc_encode_pipeline_string(
        64,
        48,
        30,
        format_caps="GRAY16_LE",
        encoding="h264",
        output_element="fakesink sync=true",
    )
    assert "format=GRAY16_LE" in s
    assert "x264enc" in s


def test_float32_depth_to_gray16_le_sentinel_and_roundtrip() -> None:
    import numpy as np

    d_min, d_max = 0.2, 2.0
    z = np.array([[d_min, (d_min + d_max) / 2, d_max]], dtype=np.float32)
    u = float32_depth_to_gray16_le(z, d_min, d_max)
    assert u.dtype == np.uint16
    assert tuple(u[0].tolist()) == (0, 32767, 65534)
    inv = d_min + (u.astype(np.float64) / 65534.0) * (d_max - d_min)
    assert abs(float(inv[0, 1]) - 1.1) < 0.02
    bad = float32_depth_to_gray16_le(
        np.array([[np.nan, 100.0]], dtype=np.float32), d_min, d_max
    )
    assert int(bad[0, 0]) == 65535 and int(bad[0, 1]) == 65535


def test_run_pipeline_with_appsrc_sync_no_frames() -> None:
    r = run_pipeline_with_appsrc_sync("fakesink", [], fps=30)
    assert r == AppSrcPipelineResult(False, 0, "no frames")


@pytest.fixture(scope="module")
def _gst_initialized() -> None:
    """One Gst.init per module (krabby-testing-x86 image provides gi + typelibs + plugins)."""
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst

    Gst.init(None)


@pytest.mark.usefixtures("_gst_initialized")
def test_run_pipeline_with_appsrc_sync_fakesink_smoke() -> None:
    import numpy as np

    from hal.server.gstreamer_runtime import ensure_gst_initialized

    ensure_gst_initialized()
    pipe = build_software_appsrc_encode_pipeline_string(
        64, 48, 30, encoding="h264", output_element="fakesink sync=true"
    )
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(2)]
    r = run_pipeline_with_appsrc_sync(pipe, frames, fps=30, timeout_s=60.0)
    assert r.success, r.error_message
    assert r.n_pushed == 2


@pytest.mark.usefixtures("_gst_initialized")
def test_run_pipeline_with_appsrc_sync_gray16_depth_smoke() -> None:
    import numpy as np

    from hal.server.gstreamer_runtime import ensure_gst_initialized
    from hal.server.isaac.sensor_backend_isaac import (
        ISAAC_PIPELINE_EXAMPLE_SENSORS,
        IsaacSensorInterface,
    )

    ensure_gst_initialized()
    iface = IsaacSensorInterface(configured_sensors=ISAAC_PIPELINE_EXAMPLE_SENSORS)
    depth = next(s for s in iface.list_sensors() if s.id == "front_rgbd_gray16_depth")
    handle = iface.get_gstreamer_handle(depth)
    pipe = iface.build_pipeline(handle, encoding="h264", output_element="fakesink sync=true")
    d_min, d_max = handle.depth_range_m  # type: ignore[misc]
    h, w = handle.resolution[1], handle.resolution[0]
    z = np.full((h, w), (d_min + d_max) / 2, dtype=np.float32)
    frames = [float32_depth_to_gray16_le(z, d_min, d_max) for _ in range(2)]
    r = run_pipeline_with_appsrc_sync(pipe, frames, fps=handle.fps, timeout_s=60.0)
    assert r.success, r.error_message
    assert r.n_pushed == 2


@pytest.mark.usefixtures("_gst_initialized")
def test_smoke_from_isaac_sensor_interface() -> None:
    from hal.server.isaac.sensor_backend_isaac import (
        ISAAC_PIPELINE_EXAMPLE_SENSORS,
        IsaacSensorInterface,
    )

    iface = IsaacSensorInterface(configured_sensors=ISAAC_PIPELINE_EXAMPLE_SENSORS)
    r = smoke_from_sensor_interface(iface, n_frames=2)
    assert r.success, r.error_message


@pytest.mark.usefixtures("_gst_initialized")
def test_smoke_from_jetson_sensor_interface_sw_only() -> None:
    from hal.server.jetson.sensor_backend_jetson import JetsonSensorInterface

    iface = JetsonSensorInterface()
    r = smoke_from_sensor_interface(
        iface, n_frames=2, build_pipeline_kwargs={"use_nvenc": False}
    )
    assert r.success, r.error_message


def test_smoke_from_sensor_interface_no_sensors() -> None:
    class _Empty:
        def list_sensors(self):
            return []

    r = smoke_from_sensor_interface(_Empty())
    assert not r.success
    assert "no sensors" in (r.error_message or "")
