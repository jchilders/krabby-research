from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from parkour_tasks.crab_hexapod_task.config.crab_hex.agents.parkour_mdp_cfg import (
    CommandsCfg,
    CrabHexActionsCfg,
    CrabHexRewardsCfg,
    CrabHexStudentObservationsCfg,
    CrabHexTeacherObservationsCfg,
    EventCfg,
    ParkourEventsCfg,
    TerminationsCfg,
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


@configclass
class CrabHexTeacherEnvCfg(UnitreeGo2TeacherParkourEnvCfg):
    scene: CrabHexTeacherSceneCfg = CrabHexTeacherSceneCfg(num_envs=2048, env_spacing=1.0)
    observations: CrabHexTeacherObservationsCfg = CrabHexTeacherObservationsCfg()
    actions: CrabHexActionsCfg = CrabHexActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: CrabHexRewardsCfg = CrabHexRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
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


@configclass
class CrabHexStudentEnvCfg(UnitreeGo2StudentParkourEnvCfg):
    scene: CrabHexStudentSceneCfg = CrabHexStudentSceneCfg(num_envs=1024, env_spacing=1.0)
    observations: CrabHexStudentObservationsCfg = CrabHexStudentObservationsCfg()
    actions: CrabHexActionsCfg = CrabHexActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: CrabHexRewardsCfg = CrabHexRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
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
