# Crab hexapod task (`crab_hexapod_task`)

Short overview of this package. Full tables, env vars, and Isaac 5 / rsl_rl notes: see **`docs/crab-hexapod-policy-config.md`** at the **repo root** (`krabby-research/`).

Use **`assets/crab_hex.usda`** and point `KRABBY_HEX_USD_PATH` at it while you iterate on the asset. Once the stage is stable and loads cleanly in sim, **export and keep the binary `assets/crab_hex.usd`** (Omniverse export or your USD pipeline) for reproducible runs, Docker, and anything that expects a single `.usd` artifact.

## What was added / changed

- **Package:** `crab_hexapod_task` with `config/crab_hex/` (scene, env, MDP, PPO runner, gym registration for `Isaac-Crab-Hex-Teacher-v0` / `Isaac-Crab-Hex-Student-v0`).
- **Scene / asset:** USD path from `KRABBY_HEX_USD_PATH`; height scanner + depth prim under `KrabbyUno/base_link`; `contact_forces` disabled where needed; student `depth_camera` from `CAMERA_CFG.replace(...)`. Hexapod `crab_hex.usda` uses **zero** PhysX joint friction efforts on prismatic/revolute DOFs to avoid Isaac 5 articulation errors (see canonical doc).
- **MDP:** `ExtremeParkourObservations`, 18-DOF joint names (three actuated patterns × 6 legs), rewards/feet regexes for `Plate_Bottom` / `.*_Tibia` / hip joints.
- **Training stack:** `PPOWithExtractor` standalone + `RolloutStorage` / `TensorDict`; `OnPolicyRunnerWithExtractor` fixes (`add_git_repo_to_log`, `store_code_state` fallback, `multi_gpu_cfg`, `process_env_step`, `log()`, filtered alg keys).
- **Observations:** `_get_priv_latent` tensor shape normalization in `parkour/parkour_isaaclab/envs/mdp/observations.py`.
- **Tests:** `tests/integration/test_crab_hexapod_policy_config.py`; optional `RUN_CRAB_HEX_RUNTIME_SMOKE=1` for a heavier smoke.
- **Repo hygiene:** `.gitignore` includes `parkour/logs/`, `parkour/outputs/`. You can point `KRABBY_HEX_USD_PATH` at **`assets/crab_hex.usda`** during dev; re-export **`assets/crab_hex.usd`** for deployment or Docker defaults.

## How to run

### Train the teacher policy

Run these from **`krabby-research/parkour`** so script paths and Hydra cwd match how the stack expects to be launched.

- **`cd …/parkour`** — working directory for `scripts/rsl_rl/train.py` and local config resolution.
- **`PYTHONPATH=…/parkour_tasks`** — puts the `parkour_tasks` tree on `sys.path` so `crab_hexapod_task` (and gym task registration) import cleanly.
- **`KRABBY_HEX_USD_PATH`** — absolute path to the hexapod stage (**`.usda` or `.usd`**; `.usda` is fine while iterating). Use the **same** value for GUI and headless (`echo "$KRABBY_HEX_USD_PATH"`). See **`docs/crab-hexapod-policy-config.md`** for PhysX friction / stale-binary notes.
- **`isaaclab.sh -p …`** — Isaac Lab’s wrapper: same Python, extensions, and CUDA stack as interactive Isaac (required for `isaaclab` / sim imports).
- **`train.py --task …`** — starts PPO training for the registered Gymnasium task id (`Isaac-Crab-Hex-Teacher-v0` uses privileged / teacher observations).
- **`--headless`** — no GUI; drop it if you want a viewer.

```bash
cd /path/to/krabby-research/parkour
export PYTHONPATH="$(pwd)/parkour_tasks:${PYTHONPATH}"
export KRABBY_HEX_USD_PATH=/path/to/krabby-research/assets/crab_hex.usd

/path/to/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Teacher-v0 --headless
```

### Integration tests

Run from the **repo root** so `tests/` paths and package layout match pytest’s discovery.

- **`isaaclab.sh -p -m pytest …`** — runs pytest inside the Isaac Lab environment (same reason as training: imports and optional sim-backed checks).
- **`RUN_CRAB_HEX_RUNTIME_SMOKE=1`** — opt-in; turns on heavier checks that may spin up more of the runtime (slower, stricter).

```bash
cd /path/to/krabby-research
/path/to/IsaacLab/isaaclab.sh -p -m pytest tests/integration/test_crab_hexapod_policy_config.py -v
# Optional runtime smoke:
RUN_CRAB_HEX_RUNTIME_SMOKE=1 /path/to/IsaacLab/isaaclab.sh -p -m pytest tests/integration/test_crab_hexapod_policy_config.py -v
```
