import os

from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from parkour_tasks.crab_hexapod_task.config.crab_hex.agents.parkour_mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    CrabHexFlatWalkActionsCfg,
    CrabHexFlatWalkRewardsCfg,
    CrabHexFlatWalkTerminationsCfg,
    CrabHexRewardsCfg,
    CrabHexStudentObservationsCfg,
    CrabHexStudentRewardsCfg,
    CrabHexTeacherObservationsCfg,
    CrabHexTeacherBridgeRewardsCfg,
    CrabHexTerminationsCfg,
    EventCfg,
    ParkourEventsCfg,
)
from parkour_tasks.crab_hexapod_task.config.crab_hex.crab_hex_scene_cfg import (
    CrabHexStudentSceneCfg,
    CrabHexTeacherSceneCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.parkour_student_cfg import (
    UnitreeGo2StudentParkourEnvCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import (
    UnitreeGo2TeacherParkourEnvCfg,
)

# Front 3/4 view: FL/FR at −x; Go2 ``VIEWER`` is a tight +y side shot.
CRAB_HEX_VIEWER = ViewerCfg(
    eye=(-4.0, 0.5, 1.55),
    lookat=(0.0, 0.0, 0.35),
    asset_name="robot",
    origin_type="asset_root",
)
# Top view: directly above root (raise z for wider view)
# CRAB_HEX_VIEWER = ViewerCfg(
#     eye=(0.0, 0.0, 6.0),      # directly above root (raise z for wider view)
#     lookat=(0.0, 0.0, 0.35),  # same as now — chassis height
#     asset_name="robot",
#     origin_type="asset_root",
# )


def _crab_hex_train_easy_enabled() -> bool:
    """True when flat-walk → teacher bridge mode is active (``KRABBY_HEX_TRAIN_EASY=1``)."""
    return os.environ.get("KRABBY_HEX_TRAIN_EASY", "").strip() == "1"


def _apply_crab_hex_bridge_actions_and_events(cfg, *, action_scale: float = 0.24) -> None:
    """Flat-walk-compatible actions/events for the flat-walk → teacher bridge."""
    cfg.actions.joint_pos.scale = action_scale
    cfg.actions.joint_pos.clip = {".*": (-1.0, 1.0)}
    cfg.actions.joint_pos.use_delay = False
    cfg.actions.joint_pos.history_length = 1

    cfg.events.push_by_setting_velocity = None
    cfg.events.randomize_rigid_body_mass = None
    cfg.events.randomize_rigid_body_com = None

    cfg.terminations.crab_failure.params["contact_force_threshold"] = 800.0


def _apply_crab_hex_easy_mixed_terrain(
    tg, *, flat_proportion: float, difficulty_range: tuple[float, float]
) -> None:
    """``parkour_flat`` + easy parkour sub-terrains (e.g. 50/50 or 70/30)."""
    tg.curriculum = False
    tg.difficulty_range = difficulty_range
    active = [k for k in tg.sub_terrains if k not in ("parkour_flat", "parkour_demo")]
    n_other = len(active)
    share = ((1.0 - flat_proportion) / n_other) if n_other else 0.0
    for key, sub_terrain in tg.sub_terrains.items():
        if key == "parkour_flat":
            sub_terrain.proportion = flat_proportion
        elif key == "parkour_demo":
            sub_terrain.proportion = 0.0
        else:
            sub_terrain.proportion = share
        sub_terrain.noise_range = (0.02, 0.02)


def _apply_crab_hex_bridge_shallow_parkour_geometry(tg) -> None:
    """Shallow, narrow gaps/pits for bridge (depth is not scaled by ``difficulty_range``)."""
    if "parkour_gap" in tg.sub_terrains:
        gap = tg.sub_terrains["parkour_gap"]
        gap.gap_depth = (0.05, 0.12)
        gap.gap_size = "0.08 + 0.35 * difficulty"
        gap.half_valid_width = (0.9, 1.2)
    if "parkour" in tg.sub_terrains:
        tg.sub_terrains["parkour"].pit_depth = (0.05, 0.12)


@configclass
class CrabHexTeacherEnvCfg(UnitreeGo2TeacherParkourEnvCfg):
    viewer = CRAB_HEX_VIEWER
    scene: CrabHexTeacherSceneCfg = CrabHexTeacherSceneCfg(num_envs=6144, env_spacing=1.0)
    observations: CrabHexTeacherObservationsCfg = CrabHexTeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: CrabHexRewardsCfg = CrabHexRewardsCfg()
    terminations: CrabHexTerminationsCfg = CrabHexTerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physx.enable_external_forces_every_iteration = True
        # Skew velocity commands toward meaningful forward speed (reduces near-zero command_vel in rewards).
        self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.85)
        base_body_cfg = SceneEntityCfg("robot", body_names="body")
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"] = base_body_cfg
        if self.events.randomize_rigid_body_mass is not None:
            self.events.randomize_rigid_body_mass.params["asset_cfg"] = base_body_cfg
        if self.events.randomize_rigid_body_com is not None:
            self.events.randomize_rigid_body_com.params["asset_cfg"] = base_body_cfg

        # Teacher bridge: flat-walk-compatible actions/commands/events, ~82/18 flat/parkour mix.
        if _crab_hex_train_easy_enabled():
            _apply_crab_hex_bridge_actions_and_events(self)

            self.commands.base_velocity.ranges.heading = (0.0, 0.0)
            self.commands.base_velocity.heading_control_stiffness = 1.5
            self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.85)

            self.rewards = CrabHexTeacherBridgeRewardsCfg()
            self.parkours.base_parkour.freeze_terrain_levels = True
            tg = getattr(self.scene.terrain, "terrain_generator", None) if self.scene.terrain else None
            if tg is not None:
                _apply_crab_hex_easy_mixed_terrain(
                    tg, flat_proportion=0.825, difficulty_range=(0.08, 0.30)
                )
                _apply_crab_hex_bridge_shallow_parkour_geometry(tg)


@configclass
class CrabHexFlatWalkEnvCfg(CrabHexTeacherEnvCfg):
    """Stage-1 flat walk: 100% ``parkour_flat``, minimal rewards, reduced domain randomization."""

    actions: CrabHexFlatWalkActionsCfg = CrabHexFlatWalkActionsCfg()
    rewards: CrabHexFlatWalkRewardsCfg = CrabHexFlatWalkRewardsCfg()
    terminations: CrabHexFlatWalkTerminationsCfg = CrabHexFlatWalkTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Straight flat-walk: fixed world heading 0; P-control corrects slow yaw drift in play/train.
        self.commands.base_velocity.ranges.lin_vel_x = (0.30, 0.65)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_control_stiffness = 1.5
        self.events.push_by_setting_velocity = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_com = None
        tg = getattr(self.scene.terrain, "terrain_generator", None) if self.scene.terrain else None
        if tg is not None:
            tg.curriculum = False
            tg.difficulty_range = (0.1, 0.25)
            for key, sub_terrain in tg.sub_terrains.items():
                if key == "parkour_flat":
                    sub_terrain.proportion = 1.0
                else:
                    sub_terrain.proportion = 0.0


@configclass
class CrabHexFlatWalkEnvCfgPLAY(CrabHexFlatWalkEnvCfg):
    """Flat-walk visualization: follow-cam and command debug."""

    viewer = CRAB_HEX_VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain is not None:
            self.scene.terrain.max_init_terrain_level = None


@configclass
class CrabHexTeacherEnvCfgPLAY(CrabHexTeacherEnvCfg):
    """Visualization / evaluation: follow-cam, parkour debug, longer episodes, structured parkour mix.

    **Default terrain is the easy / flat-heavy mix** so stance checks are not confused with hard parkour.
    Set ``KRABBY_HEX_PLAY_HARD=1`` for the previous high-difficulty play preset (no flat, 0.7–1.0 difficulty).
    """

    viewer = CRAB_HEX_VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain is not None:
            self.scene.terrain.max_init_terrain_level = None
        # Train-easy sets terrain in ``CrabHexTeacherEnvCfg``; do not override here.
        if _crab_hex_train_easy_enabled():
            return
        tg = getattr(self.scene.terrain, "terrain_generator", None) if self.scene.terrain else None
        if tg is not None:
            play_easy_flag = os.environ.get("KRABBY_HEX_PLAY_EASY", "").strip().lower() in ("1", "true", "yes")
            play_hard_flag = os.environ.get("KRABBY_HEX_PLAY_HARD", "").strip().lower() in ("1", "true", "yes")
            # Default easy unless explicitly requesting hard parkour (legacy was hard unless PLAY_EASY).
            easy = play_easy_flag or not play_hard_flag
            if easy:
                tg.difficulty_range = (0.15, 0.55)
                active = [
                    k
                    for k in tg.sub_terrains
                    if k not in ("parkour_flat", "parkour_demo")
                ]
                n_other = len(active)
                share = (0.5 / n_other) if n_other else 0.0
                for key, sub_terrain in tg.sub_terrains.items():
                    if key == "parkour_flat":
                        sub_terrain.proportion = 0.5
                    elif key == "parkour_demo":
                        sub_terrain.proportion = 0.0
                    else:
                        sub_terrain.proportion = share
                    sub_terrain.noise_range = (0.02, 0.02)
            else:
                tg.difficulty_range = (0.7, 1.0)
                for key, sub_terrain in tg.sub_terrains.items():
                    if key == "parkour_flat":
                        sub_terrain.proportion = 0.0
                    else:
                        sub_terrain.proportion = 0.2
                        sub_terrain.noise_range = (0.02, 0.02)


@configclass
class CrabHexStudentEnvCfg(UnitreeGo2StudentParkourEnvCfg):
    scene: CrabHexStudentSceneCfg = CrabHexStudentSceneCfg(num_envs=192, env_spacing=1.0)
    observations: CrabHexStudentObservationsCfg = CrabHexStudentObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: CrabHexStudentRewardsCfg = CrabHexStudentRewardsCfg()
    terminations: CrabHexTerminationsCfg = CrabHexTerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        base_body_cfg = SceneEntityCfg("robot", body_names="body")
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"] = base_body_cfg
        if self.events.randomize_rigid_body_mass is not None:
            self.events.randomize_rigid_body_mass.params["asset_cfg"] = base_body_cfg
        if self.events.randomize_rigid_body_com is not None:
            self.events.randomize_rigid_body_com.params["asset_cfg"] = base_body_cfg
