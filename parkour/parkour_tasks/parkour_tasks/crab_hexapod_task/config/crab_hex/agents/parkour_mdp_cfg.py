from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from parkour_isaaclab.envs.mdp import observations as mdp_observations
from parkour_isaaclab.envs.mdp import rewards as mdp_rewards
from parkour_isaaclab.envs.mdp import terminations as parkour_terminations
from parkour_tasks.crab_hexapod_task.config.crab_hex.crab_hex_mdp_terminations import (
    terminate_crab_hex_failure,
)
from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import (
    ActionsCfg,
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
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Tibia"),
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
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Tibia"),
                "parkour_name": "base_parkour",
                "history_length": 10,
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CrabHexRewardsCfg:
    """Parkour rewards aligned with Go2 ``TeacherRewardsCfg`` weights; contact terms use ``contact_forces``."""

    reward_collision = RewTerm(
        func=mdp_rewards.reward_collision,
        weight=-6.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                # Include hip links: otherwise hip-on-terrain is not counted and the policy can
                # minimize torques by sitting on the hip (knees bent) with little collision signal.
                body_names=["body", ".*_Hip", ".*_Femur", ".*_Tibia"],
            ),
        },
    )
    reward_feet_edge = RewTerm(
        func=mdp_rewards.reward_feet_edge,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Tibia"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Tibia"),
            "parkour_name": "base_parkour",
        },
    )
    reward_torques = RewTerm(
        func=mdp_rewards.reward_torques,
        weight=-0.00001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_error = RewTerm(
        func=mdp_rewards.reward_dof_error,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_hip_pos = RewTerm(
        func=mdp_rewards.reward_hip_pos,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Body_Hip_RevoluteJoint"])},
    )
    reward_ang_vel_xy = RewTerm(
        func=mdp_rewards.reward_ang_vel_xy,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_action_rate = RewTerm(
        func=mdp_rewards.reward_action_rate,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_acc = RewTerm(
        func=mdp_rewards.reward_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_lin_vel_z = RewTerm(
        func=mdp_rewards.reward_lin_vel_z,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    reward_orientation = RewTerm(
        func=mdp_rewards.reward_orientation,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    reward_feet_stumble = RewTerm(
        func=mdp_rewards.reward_feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Tibia")},
    )
    reward_tracking_goal_vel = RewTerm(
        func=mdp_rewards.reward_tracking_goal_vel,
        weight=2.25,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_tracking_yaw = RewTerm(
        func=mdp_rewards.reward_tracking_yaw,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "parkour_name": "base_parkour"},
    )
    reward_delta_torques = RewTerm(
        func=mdp_rewards.reward_delta_torques,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CrabHexStudentRewardsCfg:
    """Same pattern as Go2 ``StudentRewardsCfg``: collision term weight 0; hex collision bodies on ``contact_forces``."""

    reward_collision = RewTerm(
        func=mdp_rewards.reward_collision,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["body", ".*_Hip", ".*_Femur", ".*_Tibia"],
            ),
        },
    )


@configclass
class CrabHexTerminationsCfg:
    """Parkour episode term (timeout / goal / legacy fall) plus crab-specific early failure.

    Tune ``crab_failure.params``: ``limit_angle``, ``contact_force_threshold``, optional
    ``minimum_root_height_z``, ``hip_contact_sensor_cfg``. Env: ``KRABBY_HEX_TRAIN_EASY`` / spawn documented in scene/env cfgs.
    """

    total_terminates = DoneTerm(
        func=parkour_terminations.terminate_episode,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    crab_failure = DoneTerm(
        func=terminate_crab_hex_failure,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # Slightly looser than 1.35 / 65 N: fewer false resets; tighten again if hip-scrape persists.
            "limit_angle": 1.5,
            "minimum_root_height_z": None,
            "contact_force_threshold": 85.0,
            # Hips only: chassis ``body`` contact was ending episodes during benign brushes.
            "hip_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_Hip"]),
        },
    )
