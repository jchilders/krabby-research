"""Jetson sensor catalog: GStreamer caps + HAL RGB-D observation wiring.

Pipeline generation uses appsrc + nvenc (or software encode). ``JETSON_SENSOR_CATALOG``
rows describe every logical sensor; rgbd rows can opt into HAL capture via ``hal_open_rgbd``,
``policy_scan_slot``, ``zed_usb_serial_env``, and (for MaixSense) ``maixsense_host_env`` /
optional ``maixsense_port_env`` (see ``JetsonHalServer``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

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
    # Required on ``is_primary`` rgbd rows: key in ``front_camera_factory.FRONT_RGB_DEPTH_CAMERA_FACTORIES``
    # (e.g. ``zed``, ``maixsense_a075v``). Other rows: logical capture id for listing/pipelines.
    camera_driver: str | None = None
    # HAL ``RgbDepthCamera`` capture (``JetsonHalServer``): primary row always opens when rgbd + driver
    # is registered. Non-primary rgbd rows open only when ``hal_open_rgbd`` is True.
    hal_open_rgbd: bool = False
    # Maps this row into legacy ``HardwareObservations`` policy scan slots (``scan_features`` /
    # ``side_scan_features``). Primary uses ``is_primary``; at most one other row may use ``"side"``.
    policy_scan_slot: Literal["side"] | None = None
    # For ``camera_driver=="zed"``: read USB serial from this env var (int). Primary may omit (first ZED).
    # Non-primary ZED rows with ``hal_open_rgbd`` should set this so the correct unit opens.
    zed_usb_serial_env: str | None = None
    # For ``camera_driver=="maixsense_a075v"``: required HTTP host env var *name*; optional port
    # env var name (if omitted, port 80 is used without reading an env var).
    maixsense_host_env: str | None = None
    maixsense_port_env: str | None = None

# ``JetsonHalServer`` uses ``front_observation_camera_catalog_entry()`` for resolution/fps and driver.
# Swap primary ``camera_driver`` between ``zed`` and ``maixsense_a075v`` for interchangeable front RGB-D.
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
    # Second RGB-D (policy side slot when ``policy_scan_slot="side"``): driver is per-row
    # (default ``zed`` here). Non-primary ZED rows should set ``zed_usb_serial_env`` to an
    # env var holding that unit's integer USB serial.
    JetsonSensorCatalogEntry(
        id="side_rgbd",
        type="rgbd",
        modality="rgbd",
        resolution=(640, 480),
        fps=30,
        pose=SensorPose(
            x=0.0,
            y=-0.08,
            z=0.12,
            qx=0.0,
            qy=0.0,
            qz=-0.7071067811865476,
            qw=0.7071067811865476,
        ),
        appsrc_pixel_format="RGB",
        is_primary=False,
        camera_driver="zed",
        hal_open_rgbd=True,
        policy_scan_slot="side",
        zed_usb_serial_env="KRABBY_SIDE_ZED_USB_SERIAL",
    ),
)


_APPSRC_PIXEL_FORMAT_BY_SENSOR_ID: dict[str, str] = {
    e.id: e.appsrc_pixel_format for e in JETSON_SENSOR_CATALOG
}

JETSON_SENSOR_CATALOG_BY_ID: dict[str, JetsonSensorCatalogEntry] = {
    e.id: e for e in JETSON_SENSOR_CATALOG
}


def assert_hal_rgbd_catalog_config() -> None:
    """Validate rgbd HAL-related catalog rows (call before opening cameras)."""
    from hal.server.jetson.front_camera_factory import FRONT_RGB_DEPTH_CAMERA_FACTORIES

    primary_rgbd = [
        e
        for e in JETSON_SENSOR_CATALOG
        if e.is_primary and e.type == "rgbd"
    ]
    if len(primary_rgbd) != 1:
        raise RuntimeError(
            "JETSON_SENSOR_CATALOG must contain exactly one is_primary rgbd row"
        )
    p = primary_rgbd[0]
    if not p.camera_driver or p.camera_driver not in FRONT_RGB_DEPTH_CAMERA_FACTORIES:
        raise RuntimeError(
            f"Primary rgbd row must set camera_driver registered in "
            f"FRONT_RGB_DEPTH_CAMERA_FACTORIES, got {p.camera_driver!r}"
        )
    side_rows = [e for e in JETSON_SENSOR_CATALOG if e.policy_scan_slot == "side"]
    if len(side_rows) > 1:
        raise RuntimeError(
            "JETSON_SENSOR_CATALOG: at most one row may set policy_scan_slot='side'"
        )
    for e in side_rows:
        if e.is_primary:
            raise RuntimeError("policy_scan_slot='side' must not be set on the primary row")
        if not e.hal_open_rgbd:
            raise RuntimeError(
                f"Catalog {e.id!r}: policy_scan_slot='side' requires hal_open_rgbd=True"
            )
    for e in JETSON_SENSOR_CATALOG:
        if e.type != "rgbd" or e.camera_driver != "maixsense_a075v":
            continue
        if not (e.maixsense_host_env and str(e.maixsense_host_env).strip()):
            raise RuntimeError(
                f"Catalog {e.id!r}: camera_driver='maixsense_a075v' requires "
                f"non-empty maixsense_host_env (env var name for HTTP host)"
            )


def front_observation_camera_catalog_entry() -> JetsonSensorCatalogEntry:
    """The unique catalog row with ``is_primary`` True (HAL front RGB-D defaults + driver id)."""
    from hal.server.jetson.front_camera_factory import FRONT_RGB_DEPTH_CAMERA_FACTORIES

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
    if entry.camera_driver not in FRONT_RGB_DEPTH_CAMERA_FACTORIES:
        known = ", ".join(sorted(FRONT_RGB_DEPTH_CAMERA_FACTORIES))
        raise RuntimeError(
            "JETSON_SENSOR_CATALOG: is_primary camera_driver "
            f"{entry.camera_driver!r} is not registered. Known: {known}"
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
