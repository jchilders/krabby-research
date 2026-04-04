# SPDX-License-Identifier: Apache-2.0
"""MaixSense-A075V HTTP client and binary frame decode.

Logic is ported from Sipeed's tutorial material (same layout as ``stream.py`` on the
MaixSense A010 code page, used for A075V over RNDIS):

- Notebook (download): https://dl.sipeed.com/fileList/others/maixsense_example/maixsense_075_tutorial.ipynb
- Embedded ``stream.py`` reference: https://wiki.sipeed.com/hardware/zh/maixsense/maixsense-a010/code.html#streampy
- Frame layout table: https://wiki.sipeed.com/hardware/zh/metasense/metasense-a075v/matasense_075_tutorial.html

The A075V returns **RGB + depth** over HTTP (see :class:`MaixSenseDecodedFrame`). For HAL
observations, set ``camera_driver="maixsense_a075v"`` on the catalog **primary** ``rgbd`` row
(``front_rgbd``) or add additional ``rgbd`` rows with ``hal_open_rgbd`` (see
:class:`~hal.server.jetson.maixsense_rgb_depth_camera.MaixSenseA075VRgbDepthCamera`).
Use :class:`MaixSenseA075VClient` directly when feeding a custom ``appsrc`` pipeline from
:class:`~hal.server.jetson.sensor_backend_jetson.JetsonSensorInterface`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import requests
except ImportError as e:  # pragma: no cover - optional until jetson extra installed
    requests = None  # type: ignore[misc, assignment]
    _REQUESTS_IMPORT_ERROR = e
else:
    _REQUESTS_IMPORT_ERROR = None

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[misc, assignment]


def frame_config_decode(frame_config: bytes) -> tuple[int, ...]:
    """Unpack 12-byte ``config`` from the frame (see Sipeed wiki table).

    Returns:
        (trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask,
         rgb_mode, rgb_res, expose_time)
    """
    return struct.unpack("<BBBBBBBBi", frame_config)


def rgb_resolution_from_frame_config(
    config: tuple[int, ...],
) -> tuple[int, int] | None:
    """Nominal RGB **(height, width)** from the 12-byte frame ``config`` (Sipeed ``rgb_res``).

    Indexing matches :func:`frame_config_decode` / Sipeed wiki: ``rgb_mode`` (0=YUV, 1=JPG, 2=none),
    ``rgb_res`` (0=800×600, 1=1600×1200). For **JPG** (``rgb_mode == 1``) the decoded bitmap size
    can differ slightly from nominal—use :attr:`MaixSenseDecodedFrame.rgb` ``.shape`` to confirm.

    Returns:
        ``(H, W)`` for packed RGB/YUV payloads, or ``None`` if RGB is disabled (``rgb_mode == 2``)
        or size is not defined by ``rgb_res`` alone (treat as “check decoded image”).
    """
    rgb_mode = config[6]
    rgb_res = config[7]
    if rgb_mode == 2:
        return None
    if rgb_mode == 1:
        # JPEG: nominal streaming resolution from device setting; actual H×W from decode.
        if rgb_res == 0:
            return (600, 800)
        if rgb_res == 1:
            return (1200, 1600)
        return None
    # YUV (rgb_mode == 0): same nominal sizes per wiki table.
    if rgb_res == 0:
        return (600, 800)
    if rgb_res == 1:
        return (1200, 1600)
    return None


def peek_frame_config_from_getdeep_body(raw: bytes) -> tuple[int, ...]:
    """Read the 12-byte encode ``config`` from a ``GET /getdeep`` body (no full decode).

    Raises:
        ValueError: if ``raw`` is too short.
    """
    if len(raw) < 28:
        raise ValueError(f"need at least 28 bytes, got {len(raw)}")
    return frame_config_decode(raw[16:28])


def frame_config_encode(
    trigger_mode: int = 1,
    deep_mode: int = 1,
    deep_shift: int = 255,
    ir_mode: int = 1,
    status_mode: int = 2,
    status_mask: int = 7,
    rgb_mode: int = 1,
    rgb_res: int = 0,
    expose_time: int = 0,
) -> bytes:
    """Build ``set_cfg`` POST body (12 bytes). Defaults match Sipeed ``stream.py``."""
    return struct.pack(
        "<BBBBBBBBi",
        trigger_mode,
        deep_mode,
        deep_shift,
        ir_mode,
        status_mode,
        status_mask,
        rgb_mode,
        rgb_res,
        expose_time,
    )


def frame_payload_decode(
    frame_data: bytes, with_config: tuple[int, ...]
) -> tuple[Optional[bytes], Optional[bytes], Optional[bytes], Optional[bytes]]:
    """Split payload after the 12-byte config (Sipeed ``frame_payload_decode``)."""
    deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
    frame_payload = frame_data[8:]

    # 0:16bit 1:8bit, resolution 320*240
    deepth_size = (320 * 240 * 2) >> with_config[1]
    deepth_img = (
        struct.unpack("<%us" % deepth_size, frame_payload[:deepth_size])[0]
        if deepth_size != 0
        else None
    )
    frame_payload = frame_payload[deepth_size:]

    ir_size = (320 * 240 * 2) >> with_config[3]
    ir_img = (
        struct.unpack("<%us" % ir_size, frame_payload[:ir_size])[0]
        if ir_size != 0
        else None
    )
    frame_payload = frame_payload[ir_size:]

    status_size = (320 * 240 // 8) * (
        16
        if with_config[4] == 0
        else 2
        if with_config[4] == 1
        else 8
        if with_config[4] == 2
        else 1
    )
    status_img = (
        struct.unpack("<%us" % status_size, frame_payload[:status_size])[0]
        if status_size != 0
        else None
    )
    frame_payload = frame_payload[status_size:]

    if deep_data_size != deepth_size + ir_size + status_size:
        raise ValueError(
            f"deep_data_size mismatch: header {deep_data_size} != "
            f"{deepth_size + ir_size + status_size}"
        )

    rgb_size = len(frame_payload)
    if rgb_data_size != rgb_size:
        raise ValueError(
            f"rgb_data_size mismatch: header {rgb_data_size} != payload tail {rgb_size}"
        )

    rgb_img = (
        struct.unpack("<%us" % rgb_size, frame_payload[:rgb_size])[0]
        if rgb_size != 0
        else None
    )

    if rgb_img is not None and with_config[6] == 1:
        if cv2 is None:
            raise ImportError(
                "opencv-python (cv2) is required to decode rgb_mode JPG (rgb_mode==1); "
                "install e.g. opencv-python-headless"
            )
        arr = np.frombuffer(rgb_img, dtype=np.uint8)
        jpeg = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if jpeg is not None:
            rgb = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)
            rgb_img = rgb.tobytes()
        else:
            rgb_img = None

    return (deepth_img, ir_img, status_img, rgb_img)


@dataclass
class MaixSenseDecodedFrame:
    """One decoded HTTP ``/getdeep`` frame."""

    frame_id: int
    stamp_ms: int
    config: tuple[int, ...]
    depth: Optional[np.ndarray]
    ir: Optional[np.ndarray]
    status: Optional[np.ndarray]
    rgb: Optional[np.ndarray]


def decode_getdeep_response(frame_data: bytes) -> MaixSenseDecodedFrame:
    """Parse full ``GET /getdeep`` body into numpy arrays (RGB is H×W×3 uint8 when present)."""
    if len(frame_data) < 28:
        raise ValueError(f"frame too short: {len(frame_data)} bytes")
    frame_id, stamp_ms = struct.unpack("<QQ", frame_data[0:16])
    config = frame_config_decode(frame_data[16:28])
    frame_bytes = frame_payload_decode(frame_data[28:], config)

    depth = (
        np.frombuffer(
            frame_bytes[0],
            dtype=np.uint16 if config[1] == 0 else np.uint8,
        ).reshape(240, 320)
        if frame_bytes[0]
        else None
    )

    ir = (
        np.frombuffer(
            frame_bytes[1],
            dtype=np.uint16 if config[3] == 0 else np.uint8,
        ).reshape(240, 320)
        if frame_bytes[1]
        else None
    )

    status = (
        np.frombuffer(
            frame_bytes[2],
            dtype=np.uint16 if config[4] == 0 else np.uint8,
        ).reshape(240, 320)
        if frame_bytes[2]
        else None
    )

    rgb: Optional[np.ndarray] = None
    if frame_bytes[3]:
        raw_rgb = frame_bytes[3]
        flat = np.frombuffer(raw_rgb, dtype=np.uint8)
        n = flat.size
        if n % 3 != 0:
            raise ValueError(f"RGB payload size {n} is not a multiple of 3")
        # Prefer sizes from Sipeed wiki (rgb_res); fall back to stream.py example (640×480).
        candidates = (
            (600, 800, 3),
            (1200, 1600, 3),
            (480, 640, 3),
        )
        rgb = None
        for shape in candidates:
            if n == shape[0] * shape[1] * shape[2]:
                rgb = flat.reshape(shape)
                break
        if rgb is None:
            raise ValueError(
                f"RGB payload is {n} bytes; expected one of "
                f"{[s[0] * s[1] * s[2] for s in candidates]}"
            )

    return MaixSenseDecodedFrame(
        frame_id=frame_id,
        stamp_ms=stamp_ms,
        config=config,
        depth=depth,
        ir=ir,
        status=status,
        rgb=rgb,
    )


class MaixSenseA075VClient:
    """HTTP client for MaixSense-A075V on ``192.168.233.1`` (RNDIS)."""

    def __init__(
        self,
        host: str = "192.168.233.1",
        port: int = 80,
        timeout: float = 10.0,
        session: Optional[Any] = None,
    ) -> None:
        if requests is None:
            raise ImportError(
                "requests is required for MaixSenseA075VClient; "
                "install the jetson package with maixsense extras or pip install requests"
            ) from _REQUESTS_IMPORT_ERROR
        self._host = host
        self._port = port
        self._timeout = timeout
        self._session = session or requests.Session()

    def post_encode_config(self, config: Optional[bytes] = None) -> bool:
        body = config if config is not None else frame_config_encode()
        url = f"http://{self._host}:{self._port}/set_cfg"
        r = self._session.post(url, data=body, timeout=self._timeout)
        return r.status_code == requests.codes.ok

    def fetch_raw(self) -> bytes:
        """GET ``/getdeep`` raw bytes (includes header + payload)."""
        url = f"http://{self._host}:{self._port}/getdeep"
        r = self._session.get(url, timeout=self._timeout)
        r.raise_for_status()
        return r.content

    def fetch_decoded(self) -> MaixSenseDecodedFrame:
        """GET one frame and decode to arrays."""
        return decode_getdeep_response(self.fetch_raw())
