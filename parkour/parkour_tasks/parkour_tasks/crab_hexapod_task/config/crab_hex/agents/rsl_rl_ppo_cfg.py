from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_ppo_cfg import (
    UnitreeGo2ParkourStudentPPORunnerCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_teacher_ppo_cfg import (
    UnitreeGo2ParkourTeacherPPORunnerCfg,
)


@configclass
class CrabHexTeacherPPORunnerCfg(UnitreeGo2ParkourTeacherPPORunnerCfg):
    experiment_name = "crab_hex_teacher"


@configclass
class CrabHexStudentPPORunnerCfg(UnitreeGo2ParkourStudentPPORunnerCfg):
    experiment_name = "crab_hex_student"
