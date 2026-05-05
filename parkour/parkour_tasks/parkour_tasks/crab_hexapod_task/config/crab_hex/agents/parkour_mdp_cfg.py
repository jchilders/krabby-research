from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from parkour_isaaclab.envs.mdp import observations as mdp_observations
from parkour_isaaclab.envs.mdp import rewards as mdp_rewards
from parkour_isaaclab.envs.mdp import terminations as mdp_terminations
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg
from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import (
    CommandsCfg,
    EventCfg,
    ParkourEventsCfg,
    StudentObservationsCfg,
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
    reward_tracking_goal_vel = RewTerm(
        func=mdp_rewards.reward_tracking_goal_vel,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=mdp_rewards.reward_tracking_yaw,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"        },
    )


@configclass
class CrabHexTerminationsCfg:
    """Mirror Go2: use one combined termination term."""

    total_terminates = DoneTerm(
        func=mdp_terminations.terminate_episode,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot")},
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
        scale={
            ".*_HipMount_HipRevoluteJoint": 0.25,
            # Leg extension authority (too low → little surge; too high → unstable contacts).
            ".*_Hip_FemurPrismatic_PrismaticJoint": 0.05,
            ".*_Femur_TibiaPrismatic_PrismaticJoint": 0.05,
        },
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=8,
        use_delay=True,
        clip={
            ".*_HipMount_HipRevoluteJoint": (-3.5, 3.5),
            ".*_Hip_FemurPrismatic_PrismaticJoint": (-3.5, 3.5),
            ".*_Femur_TibiaPrismatic_PrismaticJoint": (-3.5, 3.5),
        },
    )

