import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg
from parkour_tasks.default_cfg import CAMERA_CFG
from parkour_tasks.extreme_parkour_task.config.go2.parkour_student_cfg import ParkourStudentSceneCfg
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import ParkourTeacherSceneCfg


def _crab_hex_usd_path() -> str:
    return os.environ.get("KRABBY_HEX_USD_PATH", "/workspace/assets/crab_hex.usd")


def _crab_hex_robot_cfg() -> ArticulationCfg:
    """Hexapod: explicit PD (ParkourDCMotor) so PhysX does not double-actuate with USD drives."""
    _krabby_root_rot_wxyz = (
        0.9271838665008545,
        0.0,
        0.0,
        -0.37460654973983765,
    )
    # Nominal stand stroke (within USD limits ±0.2 / ±0.25); see init_state comment below.
    _stand_hip_femur_prismatic = 0.10
    _stand_femur_tibia_prismatic = 0.12
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_crab_hex_usd_path(),
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
                enabled_self_collisions=False,
                # Closed-loop legs (excludeFromArticulation pins in USD) need more solver work with stiff drives.
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
            ),
        ),
        # Root pos.z matches KrabbyUno's USD xformOp:translate (0, 0, 1.25). USD drive targetPosition is 0 on
        # prismatics, but that stroke is mid-range in [-0.2,0.2]/[-0.25,0.25] — not a guaranteed stand: with all
        # prismatics at 0 the closed-loop legs often collapse so the chassis rests on the terrain ("belly").
        # Positive extension on both prismatic DOFs per leg gives a nominal load-bearing stand at reset.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.25),
            rot=_krabby_root_rot_wxyz,
            joint_pos={
                ".*_HipMount_HipRevoluteJoint": 0.0,
                ".*_Hip_FemurPrismatic_PrismaticJoint": _stand_hip_femur_prismatic,
                ".*_Femur_TibiaPrismatic_PrismaticJoint": _stand_femur_tibia_prismatic,
                ".*_Hip_Femur_RevoluteJoint": 0.0,
                ".*_Femur_Tibia_RevoluteJoint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "hip_revolutes": ParkourDCMotorCfg(
                joint_names_expr=[".*_HipMount_HipRevoluteJoint"],
                effort_limit=100.0,
                saturation_effort=120.0,
                velocity_limit=6.0,
                stiffness=50.0,
                damping=1.2,
                friction=0.0,
            ),
            "leg_prismatics": ParkourDCMotorCfg(
                joint_names_expr=[
                    ".*_Hip_FemurPrismatic_PrismaticJoint",
                    ".*_Femur_TibiaPrismatic_PrismaticJoint",
                ],
                effort_limit=5000.0,
                saturation_effort=6500.0,
                velocity_limit=0.45,
                stiffness=1800.0,
                damping=120.0,
                friction=0.0,
            ),
            # 12 DOFs: in-articulation passive revolutes (no RL action). Without these, Isaac logs
            # "18 != 30" and leaves their sim gains in an ill-defined state vs the explicit groups.
            "passive_hip_femur_revolute": ImplicitActuatorCfg(
                joint_names_expr=[".*_Hip_Femur_RevoluteJoint"],
                stiffness=0.0,
                damping=0.5,
            ),
            "passive_femur_tibia_revolute": ImplicitActuatorCfg(
                joint_names_expr=[".*_Femur_Tibia_RevoluteJoint"],
                stiffness=0.0,
                damping=0.5,
            ),
        },
    )


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
