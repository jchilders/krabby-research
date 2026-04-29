# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Policy layout sizes for ``ExtremeParkourObservations`` (see parkour_isaaclab.envs.mdp.observations).
#
# Krabby crab_hex articulation: ``num_joints = 30``, joint_pos action dim ``18`` (same formulas as env).
#   obs_buf_dim = 13 + 2 * num_joints + action_dim + 4  ->  13 + 60 + 18 + 4 = 95  (= num_prop)
#   priv_latent   = mass(1+3) + friction(1) + stiffness(N) + damping(N)  ->  4 + 1 + 30 + 30 = 65
#   policy flat len = (1 + history_length) * obs_buf_dim + 132 + 9 + priv_latent
#                   = 11 * 95 + 141 + 65 = 1251

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlBaseCfg


@configclass
class CrabHexParkourRslRlBaseCfg(ParkourRslRlBaseCfg):
    """Crab hex: must match ``ExtremeParkourObservations`` tensor layout (see module docstring)."""

    num_prop: int = 95
    num_priv_latent: int = 65
    # num_scan=132, num_hist=10, num_priv_explicit=9 — inherited


@configclass
class CrabHexParkourRslRlStateHistEncoderCfg(CrabHexParkourRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder"
    channel_size: int = 10


@configclass
class CrabHexParkourRslRlEstimatorCfg(CrabHexParkourRslRlBaseCfg):
    class_name: str = "DefaultEstimator"
    train_with_estimated_states: bool = True
    hidden_dims: list[int] = MISSING
    learning_rate: float = 1.0e-4


@configclass
class CrabHexParkourRslRlActorCfg(CrabHexParkourRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: CrabHexParkourRslRlStateHistEncoderCfg = MISSING


@configclass
class CrabHexParkourRslRlDepthEncoderCfg(CrabHexParkourRslRlBaseCfg):
    backbone_class_name: str = "DepthOnlyFCBackbone58x87"
    encoder_class_name: str = "RecurrentDepthBackbone"
    depth_shape: tuple[int, int] = (87, 58)
    hidden_dims: int = 512
    learning_rate: float = 1.0e-3
    num_steps_per_env: int = 24 * 5
