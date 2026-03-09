"""Crab Hex (hexapod) task registration."""

import gymnasium as gym

from parkour_tasks.extreme_parkour_task.config.go2 import agents

gym.register(
    id="Isaac-CrabHex-Joystick-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.crab_hex_play_cfg:CrabHexJoystickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:UnitreeGo2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)
