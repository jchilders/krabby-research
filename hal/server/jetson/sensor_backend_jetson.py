"""Jetson GStreamer sensor backend for the deployed front RGB-D camera.

Pipeline generation uses appsrc + nvenc (or software encode). Broader multi-sensor
layouts are documented separately; this catalog only registers what the Jetson image runs today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from hal.server.sensor_interface import (
    GStreamerHandle,
    SensorInfo,
    SensorInterface,
    SensorPose,
)


@dataclass(frozen=True, slots=True)
class JetsonSensorCatalogEntry:
    """One logical sensor: all static metadata in a single row."""

    id: str
    type: str  # GStreamer branch: rgb, rgbd, radar (must match build_pipeline caps lookup)
    modality: str
    resolution: tuple[int, int]
    fps: int
    pose: SensorPose
    appsrc_pixel_format: str  # caps: RGB or GRAY8 for appsrc
    is_primary: bool = False  # exactly one row: HAL front observation camera (resolution/fps defaults)
    camera_driver: str | None = None  # HAL capture driver id; required on ``is_primary`` (registry key in ``front_camera_factory``)

# Jetson hardware present on the deployed stack (front RGB-D only today).
# ``JetsonHalServer`` uses ``front_observation_camera_catalog_entry()`` for resolution/fps and driver.
JETSON_SENSOR_CATALOG: tuple[JetsonSensorCatalogEntry, ...] = (
    JetsonSensorCatalogEntry(
        id="front_rgbd",
        type="rgbd",
        modality="rgbd",
        resolution=(640, 480),
        fps=30,
        pose=SensorPose(
            x=0.33, y=0.0, z=0.08, qx=0.0, qy=0.0, qz=0.0, qw=1.0
        ),
        appsrc_pixel_format="RGB",
        is_primary=True,
        camera_driver="zed",
    ),
)

_APPSRC_PIXEL_FORMAT_BY_SENSOR_ID: dict[str, str] = {
    e.id: e.appsrc_pixel_format for e in JETSON_SENSOR_CATALOG
}


def front_observation_camera_catalog_entry() -> JetsonSensorCatalogEntry:
    """The unique catalog row with ``is_primary`` True (HAL front RGB-D defaults + driver id)."""
    primaries = [e for e in JETSON_SENSOR_CATALOG if e.is_primary]
    if len(primaries) != 1:
        raise RuntimeError(
            "JETSON_SENSOR_CATALOG must contain exactly one entry with is_primary=True"
        )
    entry = primaries[0]
    if not entry.camera_driver:
        raise RuntimeError(
            "JETSON_SENSOR_CATALOG: the is_primary row must set camera_driver "
            "(registered key in hal.server.jetson.front_camera_factory.FRONT_RGB_DEPTH_CAMERA_FACTORIES)"
        )
    return entry


def _jetson_default_sink_tail(encoding: str, use_nvenc: bool) -> str:
    """GStreamer tail after parser for encoded streams: decode + real display sink (Jetson)."""
    if use_nvenc:
        return "nvv4l2decoder ! nv3dsink"
    if encoding == "h265":
        return "avdec_h265 ! videoconvert ! autovideosink"
    return "avdec_h264 ! videoconvert ! autovideosink"


def _nvenc_pipeline(
    caps: str,
    width: int,
    height: int,
    fps: int,
    output_element: str,
    encoding: str = "h264",
    bitrate: int = 4_000_000,
) -> str:
    """Build Jetson nvenc pipeline: caps -> nvvidconv -> nvv4l2h264enc -> parse -> sink."""
    # nvvidconv expects raw input; output NV12 in NVMM for encoder
    if encoding == "h264":
        enc = f"nvv4l2h264enc bitrate={bitrate} ! h264parse"
    elif encoding == "h265":
        enc = "nvv4l2h265enc ! h265parse"
    elif encoding == "raw":
        enc = "identity"
    else:
        enc = f"nvv4l2h264enc bitrate={bitrate} ! h264parse"

    if encoding == "raw":
        return f"{caps} ! {output_element}"
    return f"{caps} ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! {enc} ! {output_element}"


def _sw_encode_pipeline(
    caps: str,
    output_element: str,
    encoding: str = "h264",
) -> str:
    """Software encode fallback when nvenc is not available (e.g. Orin Nano)."""
    if encoding == "h264":
        enc = "x264enc tune=zerolatency ! h264parse"
    elif encoding == "h265":
        enc = "x265enc ! h265parse"
    else:
        enc = "x264enc tune=zerolatency ! h264parse"
    return f"{caps} ! videoconvert ! {enc} ! {output_element}"


class JetsonSensorInterface(SensorInterface):
    """Jetson sensor interface: one ``SensorInfo`` per ``JETSON_SENSOR_CATALOG`` row.

    ``build_pipeline(..., output_element=None)`` defaults to a **decode + display** tail
    (``nvv4l2decoder ! nv3dsink`` with nvenc, or software decode + ``autovideosink``).
    Pass ``output_element="fakesink"`` (or ``filesink`` / ``udpsink`` / etc.) when you
    do not want a real video sink.
    """

    def __init__(self, *, use_nvenc: bool = True) -> None:
        self.use_nvenc = use_nvenc
        self._sensors: list[SensorInfo] = []
        for entry in JETSON_SENSOR_CATALOG:
            self._sensors.append(
                SensorInfo(
                    id=entry.id,
                    type=entry.type,
                    modality=entry.modality,
                    resolution=entry.resolution,
                    fps=entry.fps,
                    pose=entry.pose,
                    camera_driver=entry.camera_driver,
                )
            )

    def list_sensors(self) -> list[SensorInfo]:
        return list(self._sensors)

    def get_gstreamer_handle(self, sensor: SensorInfo) -> GStreamerHandle:
        fmt = _APPSRC_PIXEL_FORMAT_BY_SENSOR_ID.get(sensor.id, "RGB")
        return GStreamerHandle(
            sensor_id=sensor.id,
            sensor_type=sensor.type,
            modality=sensor.modality,
            resolution=sensor.resolution,
            fps=sensor.fps,
            camera_driver=sensor.camera_driver,
            appsrc_pixel_format=fmt,
            backend_data={"use_nvenc": self.use_nvenc},
        )

    def build_pipeline(
        self,
        handle: GStreamerHandle,
        encoding: str = "h264",
        output_element: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        use_nvenc = kwargs.get(
            "use_nvenc",
            (handle.backend_data or {}).get("use_nvenc", self.use_nvenc)
            if isinstance(handle.backend_data, dict)
            else self.use_nvenc,
        )
        w, h = handle.resolution
        fps = handle.fps
        bitrate = kwargs.get("bitrate", 4_000_000)

        fmt = handle.appsrc_pixel_format
        caps = (
            f"appsrc name=src is-live=true format=time ! "
            f"video/x-raw,format={fmt},width={w},height={h},framerate={fps}/1"
        )

        if encoding == "raw":
            sink = output_element if output_element is not None else "autovideosink"
            return f"{caps} ! videoconvert ! {sink}"

        sink = output_element if output_element is not None else _jetson_default_sink_tail(encoding, use_nvenc)
        if use_nvenc:
            return _nvenc_pipeline(
                caps, w, h, fps, sink, encoding=encoding, bitrate=bitrate
            )
        return _sw_encode_pipeline(caps, sink, encoding=encoding)
