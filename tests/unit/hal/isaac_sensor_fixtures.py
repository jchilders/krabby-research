"""Fixed ``SensorInfo`` rows for Isaac ``IsaacSensorInterface`` tests (no scene required)."""

from __future__ import annotations

from hal.server.sensor_interface import SensorInfo, SensorPose

ISAAC_CONFIGURED_SENSORS_FIXTURE: tuple[SensorInfo, ...] = (
    SensorInfo(
        id="front_rgbd",
        type="rgbd",
        modality="rgbd",
        resolution=(640, 480),
        fps=30,
        pose=SensorPose(0.33, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0),
        camera_driver="isaac_scene",
        extra={"isaac_sensor_rgb": "front_rgb", "isaac_sensor_depth": "front_camera"},
    ),
    SensorInfo(
        id="front_rgbd_gray16_depth",
        type="depth",
        modality="depth",
        resolution=(640, 480),
        fps=30,
        pose=SensorPose(0.33, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0),
        camera_driver="isaac_scene",
        extra={
            "isaac_sensor_depth": "front_camera",
            "depth_range_m": (0.2, 2.0),
            "gst_depth_source_catalog_id": "front_rgbd",
        },
    ),
    SensorInfo(
        id="side_left_rgb",
        type="rgb",
        modality="rgb",
        resolution=(640, 480),
        fps=15,
        pose=SensorPose(0.0, 0.15, 0.06, 0.0, 0.0, 0.707, 0.707),
    ),
    SensorInfo(
        id="radar_front",
        type="radar",
        modality="radar",
        resolution=(320, 240),
        fps=10,
        pose=SensorPose(0.2, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0),
    ),
)
