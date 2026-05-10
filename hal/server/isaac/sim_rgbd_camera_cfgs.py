"""Simulated ZED (front) + MaixSense-style (right side) RGB-D cameras on a robot link.

Used by ``hal.server.isaac.main`` to attach sensors to an env scene cfg **without** modifying
task packages under ``parkour/``. Imports ``CAMERA_CFG`` from ``parkour_tasks.default_cfg`` only.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils.math import quat_from_euler_xyz

from parkour_tasks.default_cfg import CAMERA_CFG

_FRONT_POS = (0.33, 0.0, 0.08)
_FRONT_EULER_DEG = (180.0, 70.0, -90.0)

_SIDE_POS = (0.0, -0.08, 0.12)
_SIDE_QUAT_WXYZ = (0.7071067811865476, 0.0, 0.0, -0.7071067811865476)


def sim_rgbd_camera_cfgs_for_robot_link(link_name: str) -> tuple[
    RayCasterCameraCfg,
    CameraCfg,
    RayCasterCameraCfg,
    CameraCfg,
]:
    """Return ``(front_camera, front_rgb, side_camera, side_rgb)`` under ``Robot/<link_name>/``."""

    base = f"{{ENV_REGEX_NS}}/Robot/{link_name}"

    front_rot = quat_from_euler_xyz(
        *tuple(torch.deg2rad(torch.tensor(_FRONT_EULER_DEG)))
    ).tolist()

    front_camera = CAMERA_CFG.replace(
        prim_path=f"{base}/front_camera",
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=_FRONT_POS,
            rot=front_rot,
            convention="ros",
        ),
        max_distance=2.0,
    )

    front_rgb = CameraCfg(
        prim_path=f"{base}/front_rgb",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.3, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_FRONT_POS,
            rot=front_rot,
            convention="ros",
        ),
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
    )

    side_rot = list(_SIDE_QUAT_WXYZ)

    side_camera = CAMERA_CFG.replace(
        prim_path=f"{base}/side_camera",
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=_SIDE_POS,
            rot=side_rot,
            convention="ros",
        ),
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=11.041,
            horizontal_aperture=20.955,
            vertical_aperture=12.240,
            height=48,
            width=64,
        ),
        max_distance=1.5,
    )

    side_rgb = CameraCfg(
        prim_path=f"{base}/side_rgb",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.15, 4.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_SIDE_POS,
            rot=side_rot,
            convention="ros",
        ),
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
    )

    return front_camera, front_rgb, side_camera, side_rgb


def attach_sim_rgbd_sensors_to_scene_cfg(scene_cfg, robot_link: str) -> None:
    """Mutate ``scene_cfg`` in place with HAL-facing sensor names (``front_rgb``, ``front_camera``, …)."""

    fc, fr, sc, sr = sim_rgbd_camera_cfgs_for_robot_link(robot_link)
    scene_cfg.front_camera = fc
    scene_cfg.front_rgb = fr
    scene_cfg.side_camera = sc
    scene_cfg.side_rgb = sr
