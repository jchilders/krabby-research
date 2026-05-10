"""Simulated ZED (front) + MaixSense-style (right side) RGB-D cameras on a robot link.

Used by ``hal.server.isaac.main`` to attach sensors to an env scene cfg **without** modifying
task packages under ``parkour/``. Imports ``CAMERA_CFG`` from ``parkour_tasks.default_cfg`` only.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

from parkour_tasks.default_cfg import CAMERA_CFG, quat_from_euler_xyz_tuple

_FRONT_POS = (0.33, 0.0, 0.08)
# Must match ``parkour_tasks.default_cfg.CAMERA_CFG``: same roll/pitch/yaw and quaternion helper
# (IsaacLab ``quat_from_euler_xyz`` omits parkour's sign convention on the quaternion).
_FRONT_EULER_DEG = (180.0, 70.0, -90.0)

# Right flank (~ ``-Body Y`` / ``_SIDE_POS``). Tuned vs front `(180°, 70°, −90°)`: ψ = ``0°`` aims the
# frustum sideways (prior ``ψ = 180°`` matched front’s modulo-360 yaw and pointed forward). Roll
# ``270°`` is front roll + ``90°`` to fix side image / lens axis twist.
_SIDE_POS = (0.0, -0.08, 0.12)
_SIDE_EULER_DEG = (270.0, 70.0, 0.0)


def _ros_offset_rot_from_euler_deg(euler_deg: tuple[float, float, float]) -> tuple[float, ...]:
    r, p, y_deg = euler_deg
    radians = torch.deg2rad(torch.tensor([float(r), float(p), float(y_deg)]))
    return quat_from_euler_xyz_tuple(*tuple(radians))


def sim_rgbd_camera_cfgs_for_robot_link(link_name: str) -> tuple[
    RayCasterCameraCfg,
    CameraCfg,
    RayCasterCameraCfg,
    CameraCfg,
]:
    """Return ``(front_camera, front_rgb, side_camera, side_rgb)`` for HAL catalog wiring.

    Ray-cast depth sensors **must** use an existing articulation link prim (same pattern as
    ``parkour_tasks.default_cfg.CAMERA_CFG``), not a fictitious ``.../front_camera`` child:
    Isaac Lab resolves ``prim_path`` to a physics/RigidBody prim; ``offset`` poses the rays.

    Pinhole RGB cameras use ``spawn=`` and may live at ``.../<link>/front_rgb`` etc.
    """

    base = f"{{ENV_REGEX_NS}}/Robot/{link_name}"
    link_prim = base

    front_rot = _ros_offset_rot_from_euler_deg(_FRONT_EULER_DEG)

    front_camera = CAMERA_CFG.replace(
        prim_path=link_prim,
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

    side_rot = _ros_offset_rot_from_euler_deg(_SIDE_EULER_DEG)

    side_camera = CAMERA_CFG.replace(
        prim_path=link_prim,
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
