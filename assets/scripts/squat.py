"""
Krabby-Uno squat helper for `assets/crab_hex.usd`.

Runs in **Isaac Sim** with `/World/KrabbyUno` on stage. Do **not** call
`omni.kit.app.get_app().update()` in a tight loop; use
`await app.next_update_async()` while the timeline plays.

**Normal squat:** all six legs move together. Each leg commands **hip mount yaw**
(degrees), **hip–femur prismatic** (m), and **femur–tibia prismatic** (m) in
phase so the body lowers and rises smoothly (zero-mean over each cycle).

Run from the Script Editor; schedules an async coroutine and returns.
"""

from __future__ import annotations

import asyncio
import math

import omni
from pxr import Usd

HEXAPOD_PRIM_PATH = "/World/KrabbyUno"

LEGS: tuple[str, ...] = ("FR", "FL", "RR", "RL", "MR", "ML")
_PREFIX = "/World/KrabbyUno"
_ATTR_ANGULAR = "drive:angular:physics:targetPosition"
_ATTR_LINEAR = "drive:linear:physics:targetPosition"

# Gait timing / depth (fraction of per-DOF limits used at peak; keep moderate).
SQUAT_FREQUENCY_HZ = 0.35
SQUAT_DEPTH_FRAC = 0.35

# Peak magnitudes at full depth (scaled by SQUAT_DEPTH_FRAC via `s` in [-1, 1]).
SQUAT_YAW_DEG = 5.0
SQUAT_HIP_M = 0.018
SQUAT_KNEE_M = 0.016


def _paths_for_leg(leg: str) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    p = _PREFIX
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


def _set_first_valid(stage: Usd.Stage, paths: tuple[str, str], attr_name: str, value: float) -> bool:
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
    stage: Usd.Stage,
    leg: str,
    hip_mount_yaw_deg: float,
    hip_femur_prismatic_m: float,
    femur_tibia_prismatic_m: float,
) -> bool:
    hip_paths, femur_paths, tibia_paths = _paths_for_leg(leg)
    ok_h = _set_first_valid(stage, hip_paths, _ATTR_ANGULAR, hip_mount_yaw_deg)
    ok_f = _set_first_valid(stage, femur_paths, _ATTR_LINEAR, hip_femur_prismatic_m)
    ok_t = _set_first_valid(stage, tibia_paths, _ATTR_LINEAR, femur_tibia_prismatic_m)
    return ok_h and ok_f and ok_t


def zero_all_legs(stage: Usd.Stage) -> None:
    for leg in LEGS:
        apply_leg_command(stage, leg, 0.0, 0.0, 0.0)


def _get_stage() -> Usd.Stage:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open. Load `assets/crab_hex.usd` first.")
    return stage


def _squat_s_command(depth: float) -> float:
    """Map smooth depth in [0,1] to s in [-1,1] (max squat magnitude at depth=1)."""
    return (2.0 * depth - 1.0) * SQUAT_DEPTH_FRAC


async def squat_reps_async(reps: int = 10) -> None:
    stage = _get_stage()
    root = stage.GetPrimAtPath(HEXAPOD_PRIM_PATH)
    if not root or not root.IsValid():
        raise RuntimeError(f"Missing `{HEXAPOD_PRIM_PATH}` on stage.")

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    omega = 2.0 * math.pi * SQUAT_FREQUENCY_HZ
    period = 1.0 / SQUAT_FREQUENCY_HZ
    total_time = reps * period
    sim_t = 0.0
    dt = 1.0 / 60.0

    try:
        while sim_t < total_time:
            depth = 0.5 * (1.0 - math.cos(omega * sim_t))
            s = _squat_s_command(depth)

            yaw = SQUAT_YAW_DEG * s
            hip = SQUAT_HIP_M * s
            knee = SQUAT_KNEE_M * s

            for leg in LEGS:
                apply_leg_command(stage, leg, yaw, hip, knee)

            await app.next_update_async()
            sim_t += dt
    finally:
        zero_all_legs(stage)
        timeline.stop()


def squat_reps(reps: int = 10) -> None:
    asyncio.ensure_future(squat_reps_async(reps=reps))


if __name__ == "__main__":
    squat_reps(10)
