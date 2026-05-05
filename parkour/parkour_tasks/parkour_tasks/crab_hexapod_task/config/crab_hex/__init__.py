"""Gym registrations for crab hex policy training tasks."""

import gymnasium as gym

from . import agents
from .crab_hex_env_cfg import CrabHexStudentEnvCfg, CrabHexTeacherEnvCfg, CrabHexTeacherEnvCfgPLAY

gym.register(
    id="Isaac-Crab-Hex-Teacher-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexTeacherPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Teacher-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexTeacherEnvCfgPLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexTeacherPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Crab-Hex-Student-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CrabHexStudentEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CrabHexStudentPPORunnerCfg,
    },
)
