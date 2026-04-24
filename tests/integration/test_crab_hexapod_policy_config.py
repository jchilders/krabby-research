import importlib
import os
import sys
from pathlib import Path

import pytest

PARKOUR_TASKS_SRC = Path(__file__).resolve().parents[2] / "parkour" / "parkour_tasks"
if str(PARKOUR_TASKS_SRC) not in sys.path:
    sys.path.insert(0, str(PARKOUR_TASKS_SRC))


def test_crab_hexapod_package_imports() -> None:
    pytest.importorskip("isaaclab")
    pkg = importlib.import_module("parkour_tasks.crab_hexapod_task")
    assert pkg is not None


def test_crab_hexapod_registered_env_ids() -> None:
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("isaaclab")
    importlib.import_module("parkour_tasks.crab_hexapod_task.config.crab_hex")
    assert "Isaac-Crab-Hex-Teacher-v0" in gym.registry
    assert "Isaac-Crab-Hex-Student-v0" in gym.registry


def test_crab_hexapod_action_joint_regexes() -> None:
    pytest.importorskip("isaaclab")
    mdp_cfg = importlib.import_module(
        "parkour_tasks.crab_hexapod_task.config.crab_hex.agents.parkour_mdp_cfg"
    )
    joint_names = mdp_cfg.CrabHexActionsCfg.joint_pos.joint_names
    assert ".*_HipMount_HipRevoluteJoint" in joint_names
    assert ".*_Hip_FemurPrismatic_PrismaticJoint" in joint_names
    assert ".*_Femur_TibiaPrismatic_PrismaticJoint" in joint_names


def _require_runtime_smoke() -> None:
    if os.environ.get("RUN_CRAB_HEX_RUNTIME_SMOKE", "0") != "1":
        pytest.skip("Set RUN_CRAB_HEX_RUNTIME_SMOKE=1 to run Isaac runtime smoke tests.")


def test_crab_hexapod_teacher_runtime_rollout_smoke() -> None:
    _require_runtime_smoke()
    gym = pytest.importorskip("gymnasium")
    torch = pytest.importorskip("torch")
    pytest.importorskip("isaaclab")
    importlib.import_module("parkour_tasks.crab_hexapod_task.config.crab_hex")
    env = gym.make("Isaac-Crab-Hex-Teacher-v0", num_envs=2, headless=True)
    obs, _ = env.reset()
    assert "policy" in obs
    initial_obs = obs["policy"].clone()
    action_dim = env.action_space.shape[-1]
    for _ in range(25):
        actions = torch.zeros((2, action_dim), device=env.device)
        obs, rewards, *_ = env.step(actions)
    obs_delta = torch.norm(obs["policy"] - initial_obs, dim=1).mean().item()
    assert obs_delta > 0.0
    assert rewards.abs().sum().item() > 0.0
    env.close()


def test_crab_hexapod_student_depth_shape_smoke() -> None:
    _require_runtime_smoke()
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("isaaclab")
    importlib.import_module("parkour_tasks.crab_hexapod_task.config.crab_hex")
    env = gym.make("Isaac-Crab-Hex-Student-v0", num_envs=1, headless=True)
    obs, _ = env.reset()
    assert "depth_camera" in obs
    assert obs["depth_camera"].shape[0] == 1
    assert obs["depth_camera"].shape[1] > 0
    env.close()
