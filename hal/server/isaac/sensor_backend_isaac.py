"""Isaac Sim GStreamer sensor backend.

**Live HAL**: pass ``scene_sensors`` (e.g. ``IsaacSimHalServer.camera_sensors``). Sensors
are **introspected** from the scene (today: ``front_rgb`` + ``front_camera`` → one
``front_rgbd`` row with ``camera_driver="isaac_scene"``).

**Explicit configuration**: pass ``configured_sensors=(...)`` to supply a fixed
``SensorInfo`` list (tools, tests). If set, **scene introspection is skipped**.

**No scene and no config** → empty ``list_sensors()`` (nothing is invented).

Pipelines use appsrc; the application pushes rendered frames into GStreamer.
"""

from __future__ import annotations

from typing import Any, Optional

from hal.server.sensor_interface import (
    GStreamerHandle,
    SensorInfo,
    SensorInterface,
    SensorPose,
)


# Pass as ``configured_sensors=ISAAC_PIPELINE_EXAMPLE_SENSORS`` when printing example
# pipeline strings without a running scene (not used by ``IsaacSimHalServer``).
ISAAC_PIPELINE_EXAMPLE_SENSORS: tuple[SensorInfo, ...] = (
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


def _tensor_list1d(t) -> list[float]:
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return [float(x) for x in t.detach().cpu().flatten().tolist()]
    except ImportError:
        pass
    if hasattr(t, "tolist"):
        return [float(x) for x in t.tolist()]
    return [float(x) for x in t]


def _pose_from_sensor_cfg(cfg) -> Optional[SensorPose]:
    """Pose from Isaac Lab sensor ``cfg.offset`` (pos + quat), or None if unavailable."""
    if cfg is None:
        return None
    off = getattr(cfg, "offset", None)
    if off is None:
        return None
    pos = getattr(off, "pos", None)
    rot = getattr(off, "rot", None)
    if pos is None:
        return None
    p = _tensor_list1d(pos)
    if len(p) < 3:
        return None
    px, py, pz = p[0], p[1], p[2]
    if rot is None:
        return SensorPose(px, py, pz, 0.0, 0.0, 0.0, 1.0)
    r = _tensor_list1d(rot)
    if len(r) < 4:
        return SensorPose(px, py, pz, 0.0, 0.0, 0.0, 1.0)
    # Isaac Lab quat_from_euler_* returns (w, x, y, z)
    qw, qx, qy, qz = r[0], r[1], r[2], r[3]
    return SensorPose(px, py, pz, qx, qy, qz, qw)


def _resolution_and_fps_from_cfg(cfg, default_res: tuple[int, int], default_fps: int) -> tuple[tuple[int, int], int]:
    if cfg is None:
        return default_res, default_fps
    w = getattr(cfg, "width", None)
    h = getattr(cfg, "height", None)
    pc = getattr(cfg, "pattern_cfg", None)
    if (w is None or h is None) and pc is not None:
        w = getattr(pc, "width", w)
        h = getattr(pc, "height", h)
    res = (int(w), int(h)) if w is not None and h is not None else default_res
    upd = getattr(cfg, "update_period", None)
    if upd is None or float(upd) <= 0:
        return res, default_fps
    fps = max(1, int(round(1.0 / float(upd))))
    return res, fps


def _sensor_infos_from_scene_cameras(scene_sensors: dict) -> list[SensorInfo]:
    """``front_rgb`` + ``front_camera`` → one introspected ``front_rgbd``; else []."""
    if "front_rgb" not in scene_sensors or "front_camera" not in scene_sensors:
        return []
    rgb_sensor = scene_sensors["front_rgb"]
    rgb_cfg = getattr(rgb_sensor, "cfg", None)
    res, fps = _resolution_and_fps_from_cfg(rgb_cfg, (640, 480), 30)
    pose = _pose_from_sensor_cfg(rgb_cfg)
    return [
        SensorInfo(
            id="front_rgbd",
            type="rgbd",
            modality="rgbd",
            resolution=res,
            fps=fps,
            pose=pose,
            camera_driver="isaac_scene",
            extra={"isaac_sensor_rgb": "front_rgb", "isaac_sensor_depth": "front_camera"},
        )
    ]


def _appsrc_sw_encode_pipeline(
    width: int,
    height: int,
    fps: int,
    format_caps: str = "RGB",
    encoding: str = "h264",
    output_element: str = "fakesink",
) -> str:
    """Build appsrc -> videoconvert -> software encode -> sink (for Isaac synthetic)."""
    caps = (
        f"appsrc name=src is-live=true format=time ! "
        f"video/x-raw,format={format_caps},width={width},height={height},framerate={fps}/1"
    )
    if encoding == "raw":
        return f"{caps} ! videoconvert ! {output_element}"
    if encoding == "h264":
        enc = "x264enc tune=zerolatency ! h264parse"
    elif encoding == "h265":
        enc = "x265enc ! h265parse"
    else:
        enc = "x264enc tune=zerolatency ! h264parse"
    return f"{caps} ! videoconvert ! {enc} ! {output_element}"


class IsaacSensorInterface(SensorInterface):
    """Isaac sensors: explicit ``configured_sensors``, or introspection from ``scene_sensors``."""

    def __init__(
        self,
        scene_sensors: Optional[dict] = None,
        *,
        configured_sensors: Optional[tuple[SensorInfo, ...]] = None,
    ) -> None:
        self.scene_sensors = dict(scene_sensors) if scene_sensors else {}
        if configured_sensors is not None:
            self._sensors = list(configured_sensors)
        else:
            self._sensors = _sensor_infos_from_scene_cameras(self.scene_sensors)

    def list_sensors(self) -> list[SensorInfo]:
        return list(self._sensors)

    def get_gstreamer_handle(self, sensor: SensorInfo) -> GStreamerHandle:
        fmt = "GRAY8" if sensor.type == "radar" else "RGB"
        return GStreamerHandle(
            sensor_id=sensor.id,
            sensor_type=sensor.type,
            modality=sensor.modality,
            resolution=sensor.resolution,
            fps=sensor.fps,
            camera_driver=sensor.camera_driver,
            appsrc_pixel_format=fmt,
            backend_data={"extra": sensor.extra} if sensor.extra else None,
        )

    def build_pipeline(
        self,
        handle: GStreamerHandle,
        encoding: str = "h264",
        output_element: str = "fakesink",
        **kwargs: Any,
    ) -> str:
        w, h = handle.resolution
        fps = handle.fps
        return _appsrc_sw_encode_pipeline(
            width=w,
            height=h,
            fps=fps,
            format_caps=handle.appsrc_pixel_format,
            encoding=encoding,
            output_element=output_element,
        )
