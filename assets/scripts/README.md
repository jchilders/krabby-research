# Krabby-Uno Isaac Sim scripts

Use **Isaac Sim** with the stage **`/World/KrabbyUno`** from [`crab_hex.usd`](../crab_hex.usd).

## `squat.py`

- **Full-leg squat:** all six legs in sync — **hip yaw**, **hip–femur prismatic**, **femur–tibia prismatic** (same joint paths as `simple_walk.py`).
- Run from the **Script Editor** (open the file or paste its contents). It schedules an **async** coroutine using `await app.next_update_async()` while the timeline plays.
- Do **not** call `omni.kit.app.get_app().update()` in a tight loop.

## `simple_walk.py`

- **Self-contained** — paste the **entire** file into the Script Editor and run (no extra modules on `sys.path`).
- Open-loop **tripod** gait with **zero-mean** commands (stable over longer runs).
- Default: `walk_forward_steps(5)` when run as `__main__`. Call `walk_forward_steps(n)` or `stop_robot()` as needed.
- After editing the script, **paste the updated buffer** again before re-running.

## Quick run

1. Load `assets/crab_hex.usd` in Isaac Sim.
2. Open the Script Editor, paste `squat.py` or `simple_walk.py`, execute.

Or run the file from disk if `__file__` resolves correctly in your Kit setup.
