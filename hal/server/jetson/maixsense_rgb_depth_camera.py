"""MaixSense-A075V as :class:`~hal.server.jetson.rgb_depth_camera.RgbDepthCamera`.

Use ``JETSON_SENSOR_CATALOG`` with ``camera_driver="maixsense_a075v"`` on any rgbd row (primary
or ``hal_open_rgbd`` side/extra). Each row must set ``maixsense_host_env`` to the **name** of an
environment variable holding that module's HTTP host. Optionally set ``maixsense_port_env`` to the
name of a variable for the port; if omitted, port **80** is used (no env lookup).

Depth is converted to meters (uint16 → mm/1000; uint8 → coarse 0–5 m) then resized to the catalog
resolution. Requires MaixSense extras (``requests``, ``cv2``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

from hal.server.jetson.rgb_depth_camera import RgbDepthCamera

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[misc, assignment]


class MaixSenseA075VRgbDepthCamera:
    """HTTP MaixSense RGB-D (host from catalog ``maixsense_host_env`` env var name)."""

    def __init__(
        self,
        resolution: tuple[int, int],
        fps: int = 30,
        depth_mode: str = "PERFORMANCE",
        *,
        maixsense_host_env: str,
        maixsense_port_env: Optional[str] = None,
    ) -> None:
        self._target_w, self._target_h = int(resolution[0]), int(resolution[1])
        self.fps = fps
        _ = depth_mode  # ZED factory parity; MaixSense HTTP path does not use ZED depth modes
        self._initialized = False
        self._client: Any = None
        self._last_rgb: Optional[np.ndarray] = None
        self._last_depth_m: Optional[np.ndarray] = None

        host_key = maixsense_host_env.strip()
        if not host_key:
            raise RuntimeError(
                "maixsense_host_env must be a non-empty env var name for camera_driver=maixsense_a075v"
            )
        host = os.environ.get(host_key, "").strip()
        if not host:
            raise RuntimeError(
                f"Environment variable {host_key!r} must be set when using camera_driver=maixsense_a075v"
            )
        port_key_stripped = (maixsense_port_env or "").strip()
        if not port_key_stripped:
            port = 80
            port_key = "(default 80)"
        else:
            port_key = port_key_stripped
            port_raw = os.environ.get(port_key, "").strip()
            if not port_raw:
                raise RuntimeError(
                    f"Environment variable {port_key!r} must be set when maixsense_port_env is configured"
                )
            try:
                port = int(port_raw)
            except ValueError as e:
                raise RuntimeError(
                    f"Environment variable {port_key!r} must be an integer, got {port_raw!r}"
                ) from e

        from hal.server.jetson.maixsense_a075v import MaixSenseA075VClient

        self._client = MaixSenseA075VClient(host=host, port=port)
        if not self._client.post_encode_config():
            logger.warning("MaixSense post_encode_config failed; still attempting /getdeep")
        self._initialized = True
        logger.info(
            "MaixSense A075V RGB-D ready (env_host=%s env_port=%s → %s:%s → %dx%d)",
            host_key,
            port_key,
            host,
            port,
            self._target_w,
            self._target_h,
        )

    def _maix_depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0
        if depth.dtype == np.uint8:
            return depth.astype(np.float32) / 255.0 * 5.0
        return depth.astype(np.float32)

    def _resize_pair(
        self, rgb: np.ndarray, depth_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if cv2 is None:
            raise RuntimeError(
                "opencv (cv2) is required to resize MaixSense frames; "
                "install jetson [maixsense] extras or opencv-python-headless"
            )
        th, tw = self._target_h, self._target_w
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb_bgr_r = cv2.resize(rgb_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
        rgb_out = cv2.cvtColor(rgb_bgr_r, cv2.COLOR_BGR2RGB)
        depth_r = cv2.resize(depth_m, (tw, th), interpolation=cv2.INTER_NEAREST)
        return np.ascontiguousarray(rgb_out.astype(np.uint8)), depth_r.astype(np.float32)

    def _fetch_and_process(self) -> bool:
        if self._client is None:
            return False
        try:
            frame = self._client.fetch_decoded()
        except Exception as e:
            logger.warning("MaixSense fetch failed: %s", e)
            return False
        if frame.rgb is None:
            logger.warning("MaixSense frame has no RGB")
            return False
        rgb = np.asarray(frame.rgb, dtype=np.uint8)
        if frame.depth is None:
            logger.warning("MaixSense frame has no depth payload (enable depth in device config)")
            return False
        depth_m = self._maix_depth_to_meters(np.asarray(frame.depth))
        try:
            self._last_rgb, self._last_depth_m = self._resize_pair(rgb, depth_m)
        except Exception as e:
            logger.error("MaixSense resize failed: %s", e)
            return False
        return True

    def get_camera_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._initialized:
            return None, None
        if not self._fetch_and_process():
            return None, None
        return self._last_rgb, self._last_depth_m

    def is_ready(self) -> bool:
        return self._initialized

    def close(self) -> None:
        self._initialized = False
        self._client = None
        self._last_rgb = None
        self._last_depth_m = None
        logger.info("MaixSense A075V RGB-D camera closed")


def create_maixsense_a075v_rgb_depth_camera(
    resolution: tuple[int, int] = (640, 480),
    fps: int = 30,
    depth_mode: str = "PERFORMANCE",
    *,
    maixsense_host_env: str,
    maixsense_port_env: Optional[str] = None,
) -> Optional[RgbDepthCamera]:
    """Build MaixSense RGB-D camera; returns ``None`` on configuration or import errors."""
    try:
        cam: RgbDepthCamera = MaixSenseA075VRgbDepthCamera(
            resolution=resolution,
            fps=fps,
            depth_mode=depth_mode,
            maixsense_host_env=maixsense_host_env,
            maixsense_port_env=maixsense_port_env,
        )
        return cam
    except RuntimeError as e:
        logger.error("%s", e)
        return None
    except ImportError as e:
        logger.error("MaixSense RGB-D camera import failed: %s", e)
        return None
    except Exception as e:
        logger.error("MaixSense RGB-D camera unexpected error: %s", e, exc_info=True)
        return None
