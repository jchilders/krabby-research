import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.utils import configclass

from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg
from parkour_tasks.default_cfg import CAMERA_CFG
from parkour_tasks.extreme_parkour_task.config.go2.parkour_student_cfg import ParkourStudentSceneCfg
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import ParkourTeacherSceneCfg


def _crab_hex_usd_path() -> str:
    return os.environ.get("KRABBY_HEX_USD_PATH", "/workspace/assets/crab_hex.usd")


def _crab_hex_robot_cfg() -> ArticulationCfg:
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_crab_hex_usd_path(),
            activate_contact_sensors=True,
        ),
        actuators={
            "base_legs": ParkourDCMotorCfg(
                joint_names_expr=[".*"],
                effort_limit={".*": 40.0},
                saturation_effort={".*": 45.0},
                velocity_limit={".*": 30.1},
                stiffness=40.0,
                damping=1.0,
                friction=0.0,
            )
        },
    )
    if getattr(robot.spawn, "articulation_props", None) is not None:
        robot.spawn.articulation_props.enabled_self_collisions = True
    return robot


@configclass
class CrabHexTeacherSceneCfg(ParkourTeacherSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot = _crab_hex_robot_cfg()
        self.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/KrabbyUno/base_link"
        self.contact_forces = None


@configclass
class CrabHexStudentSceneCfg(ParkourStudentSceneCfg):
    depth_camera = CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/KrabbyUno/base_link/depth_camera")

    def __post_init__(self):
        super().__post_init__()
        self.robot = _crab_hex_robot_cfg()
        self.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/KrabbyUno/base_link"
        self.contact_forces = None
