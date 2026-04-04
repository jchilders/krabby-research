"""GStreamer multi-sensor interface for HAL.

Provides a backend-agnostic API to list sensors, obtain GStreamer handles,
and build encoded or visualization pipelines. Implementations exist for
Jetson (front RGB-D + side RGB/RGB-D, radar + nvenc) and Isaac (synthetic sensors + software encode).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SensorPose:
    """Pose of a sensor relative to the **robot base frame** (fixed to the robot).

    **Position** ``x``, ``y``, ``z``: sensor origin in **meters**, expressed in base-frame
    axes (same convention as ``SENSOR_INTERFACE.md``).

    **Orientation** ``qx``, ``qy``, ``qz``, ``qw``: a **unit quaternion** in **scalar-last**
    (Hamilton) form: vector part ``(qx, qy, qz)`` then scalar ``qw``. It encodes the
    **orientation of the sensor frame relative to the base frame** (exact passive/active
    use depends on the consumer). Same component order as common ``(x, y, z, w)``
    quaternion fields (e.g. ROS ``geometry_msgs/Quaternion``); the ``q*`` prefix
    disambiguates from position ``x``, ``y``, ``z``.
    """

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    def to_tuple(self) -> tuple[float, float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw)


@dataclass(frozen=True)
class SensorInfo:
    """Metadata for one sensor exposed by the HAL sensor interface."""

    id: str
    type: str  # GStreamer branch: "rgb", "rgbd", "radar"
    modality: str  # "rgb", "depth", "rgbd", "radar"
    resolution: tuple[int, int]  # (width, height)
    fps: int
    pose: Optional[SensorPose] = None
    camera_driver: str | None = None  # HAL in-process capture driver (catalog); None if not applicable
    extra: Optional[dict[str, Any]] = None


@dataclass
class GStreamerHandle:
    """Opaque handle for pipeline construction. Backend-specific payload in .backend_data."""

    sensor_id: str
    sensor_type: str
    modality: str
    resolution: tuple[int, int]
    fps: int
    camera_driver: str | None = None  # Same as matching ``SensorInfo.camera_driver`` when applicable
    appsrc_pixel_format: str = "RGB"  # GStreamer ``video/x-raw,format=`` for appsrc caps
    backend_data: Any = None  # Backend-specific (e.g. Isaac sensor name map in extra)


class SensorInterface(ABC):
    """Abstract interface for listing sensors and building GStreamer pipelines.

    Implementations: JetsonSensorInterface (real hardware catalog),
    IsaacSensorInterface (synthetic sensors with same API).
    """

    @abstractmethod
    def list_sensors(self) -> list[SensorInfo]:
        """Return available sensors with metadata (id, type, pose, modality, resolution, fps, camera_driver)."""
        ...

    @abstractmethod
    def get_gstreamer_handle(self, sensor: SensorInfo) -> GStreamerHandle:
        """Map ``sensor`` to a ``GStreamerHandle`` for ``build_pipeline`` (no id lookup)."""
        ...

    @abstractmethod
    def build_pipeline(
        self,
        handle: GStreamerHandle,
        encoding: str = "h264",
        output_element: str = "fakesink",
        **kwargs: Any,
    ) -> str:
        """Return a GStreamer pipeline string that outputs an encoded or raw stream.

        Args:
            handle: From ``get_gstreamer_handle(sensor)`` (typically a ``SensorInfo`` from ``list_sensors()``).
            encoding: "h264", "h265", or "raw" (no encode; for visualization).
            output_element: Sink element name, e.g. "fakesink", "autovideosink", "appsink".
            **kwargs: Backend-specific options (e.g. bitrate).

        Returns:
            GStreamer pipeline string (e.g. for gst_parse_launch).
        """
        ...
