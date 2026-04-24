from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from parkour_isaaclab.envs.mdp import observations as mdp_observations
from parkour_isaaclab.envs.mdp import rewards as mdp_rewards
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg
from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import (
    CommandsCfg,
    EventCfg,
    ParkourEventsCfg,
    StudentObservationsCfg,
    TerminationsCfg,
)


@configclass
class CrabHexTeacherObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=mdp_observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("robot", body_names=".*_Tibia"),
                "parkour_name": "base_parkour",
                "history_length": 10,
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CrabHexStudentObservationsCfg(StudentObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=mdp_observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("robot", body_names=".*_Tibia"),
                "parkour_name": "base_parkour",
                "history_length": 10,
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CrabHexRewardsCfg:
    reward_collision = RewTerm(
        func=mdp_rewards.reward_collision,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("robot", body_names=["Plate_Bottom", ".*_Tibia", ".*_Femur"]),
        },
    )
    reward_feet_edge = RewTerm(
        func=mdp_rewards.reward_feet_edge,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Tibia"),
            "sensor_cfg": SceneEntityCfg("robot", body_names=".*_Tibia"),
            "parkour_name": "base_parkour",
        },
    )
    reward_feet_stumble = RewTerm(
        func=mdp_rewards.reward_feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("robot", body_names=".*_Tibia")},
    )
    reward_hip_pos = RewTerm(
        func=mdp_rewards.reward_hip_pos,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipMount_HipRevoluteJoint"])},
    )


@configclass
class CrabHexActionsCfg:
    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_HipMount_HipRevoluteJoint",
            ".*_Hip_FemurPrismatic_PrismaticJoint",
            ".*_Femur_TibiaPrismatic_PrismaticJoint",
        ],
        scale=0.25,
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=8,
        use_delay=True,
        clip={".*": (-4.8, 4.8)},
    )

