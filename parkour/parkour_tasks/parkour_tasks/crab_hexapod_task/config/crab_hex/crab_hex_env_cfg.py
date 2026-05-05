from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from parkour_tasks.crab_hexapod_task.config.crab_hex.agents.parkour_mdp_cfg import (
    CommandsCfg,
    CrabHexActionsCfg,
    CrabHexRewardsCfg,
    CrabHexStudentObservationsCfg,
    CrabHexTeacherObservationsCfg,
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
from parkour_tasks.default_cfg import VIEWER
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import (
    UnitreeGo2TeacherParkourEnvCfg,
)


@configclass
class CrabHexTeacherEnvCfg(UnitreeGo2TeacherParkourEnvCfg):
    scene: CrabHexTeacherSceneCfg = CrabHexTeacherSceneCfg(num_envs=2048, env_spacing=1.0)
    observations: CrabHexTeacherObservationsCfg = CrabHexTeacherObservationsCfg()
    actions: CrabHexActionsCfg = CrabHexActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: CrabHexRewardsCfg = CrabHexRewardsCfg()
    terminations: CrabHexTerminationsCfg = CrabHexTerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        dummy = type("DummyContactSensor", (), {"update_period": 0.02})()
        orig = self.scene.contact_forces
        self.scene.contact_forces = dummy
        super().__post_init__()
        self.scene.contact_forces = orig
        base_body_cfg = SceneEntityCfg("robot", body_names="Plate_Bottom")
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"] = base_body_cfg
        if self.events.randomize_rigid_body_mass is not None:
            self.events.randomize_rigid_body_mass.params["asset_cfg"] = base_body_cfg
        if self.events.randomize_rigid_body_com is not None:
            self.events.randomize_rigid_body_com.params["asset_cfg"] = base_body_cfg
        # Go2 teacher pushes the root at t≈8s globally; hexapods tip over. Disable for this task.
        self.events.push_by_setting_velocity = None
        # Avoid ±5% joint noise on reset collapsing the stance.
        if self.events.reset_robot_joints is not None:
            self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # Body +x surge; include negative samples so the policy learns reverse (was forward-only).
        self.commands.base_velocity.ranges.lin_vel_x = (-0.22, 0.35)
        self.commands.base_velocity.ranges.heading = (-0.7, 0.7)
        self.commands.base_velocity.clips.lin_vel_clip = 0.12
        self.commands.base_velocity.clips.ang_vel_clip = 0.22
        # UniformParkourCommand can zero small XY commands when small_commands_to_zero is True (Go2 default).
        self.commands.base_velocity.small_commands_to_zero = False
        # Startup friction randomization + harsh terrain while the policy is random is unstable.
        self.events.physics_material = None
        self.events.randomize_rigid_body_com = None
        if self.events.randomize_rigid_body_mass is not None:
            self.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (0.0, 0.4)
        if self.scene.terrain is not None:
            self.scene.terrain.max_init_terrain_level = 0
        # Slightly smaller dt than Go2 teacher (0.005) for stiff legs / contacts.
        self.sim.dt = 0.004
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        # Hexapod + parkour terrain can exceed default 2**26; PhysX warns if below ~72M.
        self.sim.physx.gpu_collision_stack_size = max(
            self.sim.physx.gpu_collision_stack_size, 2**27
        )


@configclass
class CrabHexTeacherEnvCfgPLAY(CrabHexTeacherEnvCfg):
    """Visualization / evaluation: follow-cam, parkour debug, longer episodes, structured parkour mix."""

    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        if self.scene.terrain is not None:
            self.scene.terrain.max_init_terrain_level = None
        tg = getattr(self.scene.terrain, "terrain_generator", None) if self.scene.terrain else None
        if tg is not None:
            tg.difficulty_range = (0.7, 1.0)
            for key, sub_terrain in tg.sub_terrains.items():
                if key == "parkour_flat":
                    sub_terrain.proportion = 0.0
                else:
                    sub_terrain.proportion = 0.2
                    sub_terrain.noise_range = (0.02, 0.02)


@configclass
class CrabHexStudentEnvCfg(UnitreeGo2StudentParkourEnvCfg):
    scene: CrabHexStudentSceneCfg = CrabHexStudentSceneCfg(num_envs=1024, env_spacing=1.0)
    observations: CrabHexStudentObservationsCfg = CrabHexStudentObservationsCfg()
    actions: CrabHexActionsCfg = CrabHexActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: CrabHexRewardsCfg = CrabHexRewardsCfg()
    terminations: CrabHexTerminationsCfg = CrabHexTerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        dummy = type("DummyContactSensor", (), {"update_period": 0.02})()
        orig = self.scene.contact_forces
        self.scene.contact_forces = dummy
        super().__post_init__()
        self.scene.contact_forces = orig
        base_body_cfg = SceneEntityCfg("robot", body_names="Plate_Bottom")
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"] = base_body_cfg
        if self.events.randomize_rigid_body_mass is not None:
            self.events.randomize_rigid_body_mass.params["asset_cfg"] = base_body_cfg
        if self.events.randomize_rigid_body_com is not None:
            self.events.randomize_rigid_body_com.params["asset_cfg"] = base_body_cfg
        self.events.push_by_setting_velocity = None
        if self.events.reset_robot_joints is not None:
            self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.22, 0.35)
        self.commands.base_velocity.ranges.heading = (-0.7, 0.7)
        self.commands.base_velocity.clips.lin_vel_clip = 0.12
        self.commands.base_velocity.clips.ang_vel_clip = 0.22
        self.commands.base_velocity.small_commands_to_zero = False
        self.events.physics_material = None
        self.events.randomize_rigid_body_com = None
        if self.events.randomize_rigid_body_mass is not None:
            self.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (0.0, 0.4)
        if self.scene.terrain is not None:
            self.scene.terrain.max_init_terrain_level = 0
        self.sim.dt = 0.004
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        if getattr(self.scene, "depth_camera", None) is not None:
            self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        # Same as teacher: GPU collision stack for hex + terrain at scale.
        self.sim.physx.gpu_collision_stack_size = max(
            self.sim.physx.gpu_collision_stack_size, 2**27
        )
