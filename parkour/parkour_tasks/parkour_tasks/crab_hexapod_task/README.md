# Crab hexapod task (`crab_hexapod_task`)

This package adds a **Krabby hexapod** parkour task on top of Isaac Lab’s extreme parkour stack.  
The goal of this README is that anyone can clone the repo, create a Python env similar to yours, and **train + play** the hexapod policy.

The examples below assume:

- **`$KRABBY_ROOT=/home/sanjay/Projects/krabby`**
- **`krabby-research`** lives at **`$KRABBY_ROOT/krabby-research`**
- **Isaac Lab** lives at **`$KRABBY_ROOT/IsaacLab`**
- Your Isaac Lab conda env is called **`env_isaaclab`**

Adjust paths and the conda env name if your layout is different.

---

## 1. Environment setup (once per machine)

All commands in this README assume:

```bash
conda activate env_isaaclab
```

Then install and point Python at the **krabby-research** copies of `parkour` and `parkour_tasks`:

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby

conda activate env_isaaclab

cd "$KRABBY_ROOT/krabby-research/parkour"
pip install -e .

cd "$KRABBY_ROOT/krabby-research/parkour/parkour_tasks"
pip install -e .

export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH}"
```

Isaac Lab itself is launched via:

```bash
cd "$KRABBY_ROOT/IsaacLab"
./isaaclab.sh -p ...
```

### Hexapod asset path

The crab hexapod USD is under `krabby-research/assets/crab_hex.usda`. Point the env variable at it **before** running any training or play script:

```bash
export KRABBY_HEX_USD_PATH="$KRABBY_ROOT/krabby-research/assets/crab_hex.usda"
```

You can later freeze a binary `.usd` export and point `KRABBY_HEX_USD_PATH` at that file for deployment / Docker, but all training and play examples in this README work with the `.usda`.

---

## 2. Where hexapod configs live (scene, rewards, actions)

- **Gym registrations (teacher + student):**  
  `parkour_tasks/crab_hexapod_task/config/crab_hex/__init__.py`  
  - `Isaac-Crab-Hex-Teacher-v0` → `CrabHexTeacherEnvCfg`  
  - `Isaac-Crab-Hex-Teacher-Play-v0` → `CrabHexTeacherEnvCfgPLAY` (viewer/debug-play variant)  
  - `Isaac-Crab-Hex-Student-v0` → `CrabHexStudentEnvCfg`

- **Scene / robot / sensors:**  
  `parkour_tasks/crab_hexapod_task/config/crab_hex/crab_hex_scene_cfg.py`  
  - Loads the hexapod from `KRABBY_HEX_USD_PATH`  
  - Sets solver iterations, stand pose, hex-specific actuators, height scanner (and depth camera for the student).

- **Env configuration (teacher + student MDP):**  
  `parkour_tasks/crab_hexapod_task/config/crab_hex/crab_hex_env_cfg.py`  
  - `CrabHexTeacherEnvCfg` / `CrabHexStudentEnvCfg` set:
    - Number of envs and spacing  
    - Parkour event / command configuration (velocity command ranges, clips, curriculum)  
    - Simulation dt and GPU collision stack  
  - `CrabHexTeacherEnvCfgPLAY` is the **play/eval** variant:
    - Viewer follow-cam (uses `VIEWER` from `default_cfg.py`)  
    - Longer episodes (~60 s)  
    - Parkour + base velocity debug visualization  
    - Harder terrain band and no `parkour_flat` tiles for clearer behavior.

- **Rewards and actions for the hexapod:**  
  `parkour_tasks/crab_hexapod_task/config/crab_hex/agents/parkour_mdp_cfg.py`
  - `CrabHexRewardsCfg` wires high-level reward terms into the Isaac Lab reward functions in `parkour_isaaclab/envs/mdp/rewards.py`:
    - `reward_collision` (weight **-10.0**): collisions on `Plate_Bottom`, tibias, and femurs  
    - `reward_feet_edge` (weight **-1.0**): feet near terrain edges on the parkour course  
    - `reward_feet_stumble` (weight **-1.0**): stumble signal from tibia contacts  
    - `reward_hip_pos` (weight **-0.5**): regularize hip revolute positions  
    - `reward_tracking_goal_vel` (weight **+1.5**): track parkour goal velocity along the course  
    - `reward_tracking_yaw` (weight **+0.5**): track yaw / heading toward parkour goals
  - `CrabHexTerminationsCfg` mirrors the Go2 reference task: one combined
    `total_terminates` DoneTerm using `terminate_episode`, which OR-combines
    roll/pitch cutoff, time-out, parkour goal reached, and base-too-low.
  - `CrabHexActionsCfg` defines:
    - A **delayed joint position** action over the three actuated DOFs per leg (hip revolute, hip–femur prismatic, femur–tibia prismatic), with tuned scales, history length, and clip ranges.

The **low-level** math for each reward term (errors, masks, exponents, etc.) is implemented in  
`parkour_isaaclab/envs/mdp/rewards.py`; `CrabHexRewardsCfg` just chooses which ones to use, with what weights and bodies/joints.

### Design notes (reward intuition)

- `reward_collision` (**-10.0**): a strong penalty so the policy quickly learns “stay out of trouble” (plate/tibia/femur contacts are bad on the parkour course).
- `reward_feet_edge` (**-1.0**) and `reward_feet_stumble` (**-1.0**): gentler penalties to discourage edge scraping and toe/tibia stumbles without making the task overly brittle.
- `reward_hip_pos` (**-0.5**): a light regularizer on hip revolute angles to keep the crab from drifting into extreme / unsafe joint configurations while still allowing motion freedom.
- `reward_tracking_goal_vel` (**+1.5**): the main “progress” term—it rewards moving with the desired parkour goal motion along the course.
- `reward_tracking_yaw` (**+0.5**): a shaped reward for matching heading toward parkour goals; combined with `reward_tracking_goal_vel`, it makes motion more deliberate.

---

## 3. Training and playing the hexapod

All commands in this section assume:

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
conda activate env_isaaclab
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH}"
export KRABBY_HEX_USD_PATH="$KRABBY_ROOT/krabby-research/assets/crab_hex.usda"
cd "$KRABBY_ROOT/krabby-research/parkour"
```

### 3.1 Train the teacher

The teacher uses privileged observations (terrain, dynamics, etc.) and trains with `scripts/rsl_rl/train.py`.

**Example: 256 envs, 10 000 PPO iterations (≈ 6 h on an RTX 5080–class GPU):**

```bash
cd "$KRABBY_ROOT/IsaacLab"
./isaaclab.sh -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Teacher-v0 \
  --headless \
  --num_envs 256 \
  --seed 1 \
  --max_iterations 10000 \
  2>&1 | tee "$KRABBY_ROOT/krabby-research/parkour/crab_hex_train_256env_10k.log"
```

The script logs runs under:

```text
krabby-research/parkour/logs/rsl_rl/crab_hex_teacher/<TIMESTAMP>/
```

Inside each timestamped folder you will see:

- `model_0.pt`, `model_100.pt`, …, `model_9900.pt`, **`model_9999.pt`** (checkpoints)
- `events.out.tfevents.*` (TensorBoard)
- `params/agent.yaml`, `params/env.yaml` (frozen configs)

You can stop a long run early and still use the last `model_<iter>.pt` that was saved.  
To resume from a specific checkpoint:

```bash
./isaaclab.sh -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Teacher-v0 \
  --headless \
  --num_envs 256 \
  --seed 1 \
  --resume \
  --load_run <TIMESTAMP_DIR_NAME> \
  --checkpoint model_9200.pt \
  --max_iterations 800
```

Here `max_iterations` means “run this many **more** PPO iterations starting from the loaded `iter`,” so `9200 + 800 = 10000`.

### 3.2 Play back the trained teacher (viewer)

To **watch** the trained hexapod on parkour terrain in the Isaac Sim GUI, use `scripts/rsl_rl/play.py` with either the **training** env or the **play** env:

- Training-style env (exact MDP used for training):

```bash
cd "$KRABBY_ROOT/IsaacLab"
./isaaclab.sh -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/play.py" \
  --task Isaac-Crab-Hex-Teacher-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_teacher/<TIMESTAMP>/model_9999.pt"
```

- Play-style env (longer episodes, follow-cam, parkour + command debug, harder terrain mix):

```bash
cd "$KRABBY_ROOT/IsaacLab"
./isaaclab.sh -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/play.py" \
  --task Isaac-Crab-Hex-Teacher-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_teacher/<TIMESTAMP>/model_9999.pt"
```

Replace `<TIMESTAMP>` with the folder name printed in your training log (for example `2026-05-04_18-14-19`).

### 3.3 Train (and later distill) the student

The student uses the same hexapod env but with **student** observations (e.g. depth input) and a **distillation** algorithm. A minimal training command is:

```bash
cd "$KRABBY_ROOT/IsaacLab"
./isaaclab.sh -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Student-v0 \
  --headless \
  --num_envs 1024 \
  --seed 1
```

Student training uses the same log root:

```text
krabby-research/parkour/logs/rsl_rl/crab_hex_student/<TIMESTAMP>/
```

You can play a student checkpoint with:

```bash
./isaaclab.sh -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/play.py" \
  --task Isaac-Crab-Hex-Student-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_student/<TIMESTAMP>/model_XXXX.pt"
```

---

## 4. Reference: existing quadruped (Unitree Go2) rewards and training

The crab hexapod task is built by following the conventions of the **extreme parkour Unitree Go2** task that ships with Isaac Lab.

- **Gym registrations (Go2 teacher / student / eval / play):**  
  `IsaacLab/Isaaclab_Parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/__init__.py`  
  (e.g. `Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0`, `Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0`, etc.)

- **Go2 MDP / rewards / actions:**  
  `IsaacLab/Isaaclab_Parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/parkour_mdp_cfg.py`  
  which in turn uses the same reward functions in  
  `krabby-research/parkour/parkour_isaaclab/envs/mdp/rewards.py`.

### 4.1 Go2 teacher training (extreme parkour)

From inside the Isaac Lab checkout:

```bash
cd "$KRABBY_ROOT/IsaacLab"
conda activate env_isaaclab

./isaaclab.sh -p ./Isaaclab_Parkour/scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
  --headless \
  --num_envs 4096 \
  --seed 1
```

This writes checkpoints under:

```text
Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour_teacher/<TIMESTAMP>/
```

### 4.2 Go2 play (extreme parkour teacher play env)

You can visualize a trained Go2 teacher policy on parkour terrain using the Go2 **PLAY** env:

```bash
cd "$KRABBY_ROOT/IsaacLab"
./isaaclab.sh -p ./Isaaclab_Parkour/scripts/rsl_rl/play.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint ./Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour_teacher/<TIMESTAMP>/model_XXXX.pt
```

The hexapod task mirrors this layout (Gym registrations, env cfgs, reward wiring, and train/play scripts), so anyone familiar with the Go2 extreme parkour examples should find the crab hexapod task immediately recognizable.  
The only extra requirement is to set **`KRABBY_HEX_USD_PATH`** to point at the `crab_hex` USD before training or playing. 
