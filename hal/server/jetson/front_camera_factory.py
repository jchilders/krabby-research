"""Pluggable factories for the HAL front RGB-D camera (``RgbDepthCamera`` protocol).

Register drivers in ``FRONT_RGB_DEPTH_CAMERA_FACTORIES``. The Jetson catalog row
marked ``is_primary`` sets ``camera_driver`` (also on ``SensorInfo`` from ``list_sensors()``).
"""

from __future__ import annotations

from typing import Callable, Optional

from hal.server.jetson.rgb_depth_camera import RgbDepthCamera

FrontRgbDepthCameraFactory = Callable[
    ...,
    Optional[RgbDepthCamera],
]


def _factory_zed(
    *,
    resolution: tuple[int, int],
    fps: int,
    depth_mode: str,
    depth_feature_dim: int,
) -> Optional[RgbDepthCamera]:
    from hal.server.jetson.camera import create_zed_camera

    return create_zed_camera(
        resolution=resolution,
        fps=fps,
        depth_mode=depth_mode,
        depth_feature_dim=depth_feature_dim,
    )


FRONT_RGB_DEPTH_CAMERA_FACTORIES: dict[str, FrontRgbDepthCameraFactory] = {
    "zed": _factory_zed,
}


def create_front_rgb_depth_camera(
    driver: str,
    *,
    resolution: tuple[int, int],
    fps: int,
    depth_mode: str = "PERFORMANCE",
    depth_feature_dim: int,
) -> Optional[RgbDepthCamera]:
    """Build the front observation camera using a registered ``driver`` name."""
    try:
        factory = FRONT_RGB_DEPTH_CAMERA_FACTORIES[driver]
    except KeyError as e:
        known = ", ".join(sorted(FRONT_RGB_DEPTH_CAMERA_FACTORIES))
        raise ValueError(
            f"Unknown front RGB-D camera driver {driver!r}. Registered: {known}"
        ) from e
    return factory(
        resolution=resolution,
        fps=fps,
        depth_mode=depth_mode,
        depth_feature_dim=depth_feature_dim,
    )
