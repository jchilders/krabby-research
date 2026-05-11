from isaaclab.utils import configclass

from parkour_tasks.crab_hexapod_task.config.crab_hex.agents.crab_hex_rl_cfg import (
    CrabHexParkourRslRlActorCfg,
    CrabHexParkourRslRlDepthEncoderCfg,
    CrabHexParkourRslRlEstimatorCfg,
    CrabHexParkourRslRlStateHistEncoderCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlPpoActorCriticCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_ppo_cfg import (
    UnitreeGo2ParkourStudentPPORunnerCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_teacher_ppo_cfg import (
    UnitreeGo2ParkourTeacherPPORunnerCfg,
)


@configclass
class CrabHexTeacherPPORunnerCfg(UnitreeGo2ParkourTeacherPPORunnerCfg):
    experiment_name = "crab_hex_teacher"
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=0.65,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        activation="elu",
        actor=CrabHexParkourRslRlActorCfg(
            class_name="Actor",
            state_history_encoder=CrabHexParkourRslRlStateHistEncoderCfg(
                class_name="StateHistoryEncoder",
            ),
        ),
    )
    estimator = CrabHexParkourRslRlEstimatorCfg(hidden_dims=[128, 64])


@configclass
class CrabHexStudentPPORunnerCfg(UnitreeGo2ParkourStudentPPORunnerCfg):
    experiment_name = "crab_hex_student"
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=0.65,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        activation="elu",
        actor=CrabHexParkourRslRlActorCfg(
            class_name="Actor",
            state_history_encoder=CrabHexParkourRslRlStateHistEncoderCfg(
                class_name="StateHistoryEncoder",
            ),
        ),
    )
    estimator = CrabHexParkourRslRlEstimatorCfg(hidden_dims=[128, 64])
    depth_encoder = CrabHexParkourRslRlDepthEncoderCfg(
        hidden_dims=512,
        learning_rate=1e-3,
        num_steps_per_env=24 * 5,
    )
