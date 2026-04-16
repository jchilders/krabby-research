"""
Open-loop tripod walk for Krabby-Uno in Isaac Sim (`assets/crab_hex.usd`).

**Paste this entire file into the Script Editor and run** — it is self-contained
(no other files). The base is not teleported; motion comes from physics and feet.

If nothing starts after Run, Kit may not set `__name__` to `"__main__"`; in that case
execute once: `walk_forward_steps(5)` (or `walk_forward_steps()`).

Uses `await app.next_update_async()` while the timeline plays (do not tight-loop
`app.update()`).
"""

from __future__ import annotations

import asyncio
import math

try:
    import omni
    from pxr import Usd
except ImportError as e:
    raise RuntimeError(
        "This script must run inside Omniverse / Isaac Sim's Python environment."
    ) from e

# --- Inlined from krabby_leg_commands (so copy-paste into Script Editor works) ---

LEGS: tuple[str, ...] = ("FR", "FL", "RR", "RL", "MR", "ML")
KRABBY_ARTICULATION_PREFIX = "/World/KrabbyUno"
_ATTR_ANGULAR = "drive:angular:physics:targetPosition"
_ATTR_LINEAR = "drive:linear:physics:targetPosition"


def _paths_for_leg(leg: str) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    p = KRABBY_ARTICULATION_PREFIX
    return (
        (
            f"{p}/Root_{leg}/{leg}_Hip/{leg}_HipMount_HipRevoluteJoint",
            f"{p}/Root_{leg}/{leg}_HipMount/{leg}_HipMount_HipRevoluteJoint",
        ),
        (
            f"{p}/Root_{leg}/{leg}_FemurPrismatic/{leg}_Hip_FemurPrismatic_PrismaticJoint",
            f"{p}/Root_{leg}/{leg}_Hip/{leg}_Hip_FemurPrismatic_PrismaticJoint",
        ),
        (
            f"{p}/Root_{leg}/{leg}_TibiaPrismatic/{leg}_Femur_TibiaPrismatic_PrismaticJoint",
            f"{p}/Root_{leg}/{leg}_Femur/{leg}_Femur_TibiaPrismatic_PrismaticJoint",
        ),
    )


def _set_first_valid(stage, paths: tuple[str, str], attr_name: str, value: float) -> bool:
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        attr = prim.GetAttribute(attr_name)
        if not attr.IsValid():
            continue
        attr.Set(value)
        return True
    return False


def apply_leg_command(
    stage,
    leg: str,
    hip_mount_yaw_deg: float,
    hip_femur_prismatic_m: float,
    femur_tibia_prismatic_m: float,
) -> bool:
    hip_paths, femur_paths, tibia_paths = _paths_for_leg(leg)
    ok_hip = _set_first_valid(stage, hip_paths, _ATTR_ANGULAR, hip_mount_yaw_deg)
    ok_femur = _set_first_valid(stage, femur_paths, _ATTR_LINEAR, hip_femur_prismatic_m)
    ok_tibia = _set_first_valid(stage, tibia_paths, _ATTR_LINEAR, femur_tibia_prismatic_m)
    return ok_hip and ok_femur and ok_tibia


def zero_leg_commands(stage) -> None:
    for leg in LEGS:
        apply_leg_command(stage, leg, 0.0, 0.0, 0.0)


# --- Walk ---

HEXAPOD_PRIM_PATH = "/World/KrabbyUno"

TRIPOD_A = ("FR", "RL", "ML")
TRIPOD_B = ("FL", "RR", "MR")

# Zero-mean commands per leg over each gait cycle (critical for long runs).
# Old pattern `L*swing + P*stance` has **non-zero average** → slow CoM drift → late falls.
WALK_HZ = 0.15
WALK_YAW_DEG = 4.0
WALK_HIP_LIFT_M = 0.008
WALK_KNEE_LIFT_M = 0.006

DEFAULT_STEPS = 5
SIM_DT_S = 1.0 / 60.0


def _get_stage() -> Usd.Stage:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open. Load `assets/crab_hex.usd` first.")
    return stage


def _validate_root(stage: Usd.Stage) -> None:
    root = stage.GetPrimAtPath(HEXAPOD_PRIM_PATH)
    if not root or not root.IsValid():
        raise RuntimeError(
            f"Missing `{HEXAPOD_PRIM_PATH}`. Open the Krabby-Uno stage before running."
        )


def _apply_tripod_frame(stage: Usd.Stage, t_s: float) -> None:
    omega = 2.0 * math.pi * WALK_HZ
    a = 0.5 * (1.0 + math.sin(omega * t_s))
    b = 1.0 - a

    for leg in LEGS:
        swing = a if leg in TRIPOD_A else b
        # s in [-1, 1]; equals sin(omega * t) for tripod A and -sin for B → **zero mean** over time.
        s = 2.0 * swing - 1.0

        yaw = WALK_YAW_DEG * s
        hip = WALK_HIP_LIFT_M * s
        knee = WALK_KNEE_LIFT_M * s

        apply_leg_command(stage, leg, yaw, hip, knee)


async def walk_forward_async(*, steps: int = DEFAULT_STEPS, dt: float = SIM_DT_S) -> None:
    """Play `steps` full gait cycles."""
    stage = _get_stage()
    _validate_root(stage)

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    period_s = 1.0 / WALK_HZ
    total_time_s = steps * period_s
    t_s = 0.0

    try:
        while t_s < total_time_s:
            _apply_tripod_frame(stage, t_s)
            await app.next_update_async()
            t_s += dt
    finally:
        zero_leg_commands(stage)
        timeline.stop()


def walk_forward_steps(steps: int = DEFAULT_STEPS) -> None:
    asyncio.ensure_future(walk_forward_async(steps=steps))


def stop_robot() -> None:
    stage = _get_stage()
    _validate_root(stage)
    zero_leg_commands(stage)
    omni.timeline.get_timeline_interface().stop()


# Script Editor runs the buffer as __main__. Running `python simple_walk.py` from disk also works.
if __name__ == "__main__":
    walk_forward_steps(DEFAULT_STEPS)
