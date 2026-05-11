import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg
from parkour_tasks.crab_hexapod_task.sensors import ParkourHexContactSensorCfg
from parkour_tasks.default_cfg import CAMERA_CFG
from parkour_tasks.extreme_parkour_task.config.go2.parkour_student_cfg import ParkourStudentSceneCfg
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import ParkourTeacherSceneCfg


def _crab_simple_usd_path() -> str:
    """USD for crab_hexapod_task. Override with KRABBY_HEX_USD_PATH; default is repo `crab_simple.usda` only."""
    override = os.environ.get("KRABBY_HEX_USD_PATH")
    if override:
        return override
    # .../krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/config/crab_hex/this_file.py
    repo_root = Path(__file__).resolve().parents[6]
    default = repo_root / "assets" / "crab_simple.usda"
    if default.is_file():
        return str(default)
    return "/workspace/krabby-research/assets/crab_simple.usda"


def _crab_simple_robot_cfg() -> ArticulationCfg:
    """``crab_simple.usda`` (``defaultPrim = "krabby"``): reference composes into ``{ENV_REGEX_NS}/Robot`` — leave
    ``articulation_root_prim_path`` unset so Isaac Lab discovers the root on ``Robot``. Base link ``chassis/body``."""
    # USD lifts ``krabby`` by +1 m; tune root spawn so feet sit on terrain without huge drop or penetration.
    # Default 0.79 m (override ``KRABBY_HEX_SPAWN_Z``); lower if hover-then-slam, raise if hips scrape or interpenetration.
    spawn_z = float(os.environ.get("KRABBY_HEX_SPAWN_Z", "0.79"))
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_crab_simple_usd_path(),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spawn_z),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*_Body_Hip_RevoluteJoint": 0.0,
                # Load-bearing stance (rad): slightly more thigh lift + straighter knee so feet—not hip boxes—meet terrain first.
                ".*_Hip_Femur_RevoluteJoint": 0.35,
                # Within soft knee limits (~±0.31 rad); less flex than -0.20 lengthens the leg chain under the chassis.
                ".*_Femur_Tibia_RevoluteJoint": -0.14,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "body_hip_yaw": ParkourDCMotorCfg(
                joint_names_expr=[".*_Body_Hip_RevoluteJoint"],
                effort_limit=80.0,
                saturation_effort=100.0,
                velocity_limit=6.0,
                stiffness=50.0,
                damping=1.35,
                friction=0.0,
            ),
            # Femur–tibia stiffer than hip–femur: knee chain dominates collapse under zero-action / gravity.
            "hip_femur": ParkourDCMotorCfg(
                joint_names_expr=[".*_Hip_Femur_RevoluteJoint"],
                effort_limit=120.0,
                saturation_effort=150.0,
                velocity_limit=6.0,
                stiffness=102.0,
                damping=2.4,
                friction=0.0,
            ),
            "femur_tibia": ParkourDCMotorCfg(
                joint_names_expr=[".*_Femur_Tibia_RevoluteJoint"],
                effort_limit=150.0,
                saturation_effort=185.0,
                velocity_limit=6.0,
                stiffness=132.0,
                damping=2.85,
                friction=0.0,
            ),
        },
    )


@configclass
class CrabHexTeacherSceneCfg(ParkourTeacherSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot = _crab_simple_robot_cfg()
        self.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/chassis/body"
        # Aggregate chassis + all leg links (``ParkourHexContactSensor``): default nested ``Robot/krabby/.*/.*``
        # only reports ``chassis/body``; Isaac composes ``krabby`` children flat under ``Robot`` at runtime.
        self.contact_forces = ParkourHexContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=2,
            track_air_time=True,
            debug_vis=False,
            force_threshold=1.0,
        )


@configclass
class CrabHexStudentSceneCfg(ParkourStudentSceneCfg):
    depth_camera = CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot/chassis/body")

    def __post_init__(self):
        super().__post_init__()
        self.robot = _crab_simple_robot_cfg()
        self.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/chassis/body"
        self.contact_forces = ParkourHexContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=2,
            track_air_time=True,
            debug_vis=False,
            force_threshold=1.0,
        )
