"""Scene configuration that adds a front ZED-like camera (RGB + depth) to the parkour default scene.

Subclasses ParkourDefaultSceneCfg (parkour terrain, robot, sky) and adds front_camera
(depth) and front_rgb (RGB) without modifying any parkour files.
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from parkour_tasks.default_cfg import CAMERA_CFG, ParkourDefaultSceneCfg

# ZED-like pose: same as parkour front camera (0.33, 0, 0.08), ~70 deg down.
_CAMERA_POS = (0.33, 0.0, 0.08)
_CAMERA_EULER_DEG = (180.0, 70.0, -90.0)

# RGB-only camera at same pose (depth comes from front_camera RayCaster).
FRONT_RGB_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_rgb",
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
        pos=_CAMERA_POS,
        rot=quat_from_euler_xyz(
            *tuple(torch.deg2rad(torch.tensor(_CAMERA_EULER_DEG)))
        ).tolist(),
        convention="ros",
    ),
    colorize_semantic_segmentation=False,
    colorize_instance_id_segmentation=False,
    colorize_instance_segmentation=False,
)


@configclass
class ZedLikeSceneCfg(ParkourDefaultSceneCfg):
    """Parkour default scene plus front ZED-like camera (depth + RGB)."""

    # Depth from parkour's RayCaster camera config; unique prim_path.
    front_camera = CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/base/front_camera")
    front_rgb = FRONT_RGB_CAMERA_CFG
