"""Per-foot contact sensors for crab_hex (Isaac Lab ContactSensor is one parent prim per env)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import ContactSensor

# Scene sensor keys and matching tibia prim names (ContactSensor.body_names last path segment).
CRAB_HEX_TIBIA_CONTACT_KEYS: tuple[str, ...] = (
    "contact_feet_fl",
    "contact_feet_fr",
    "contact_feet_ml",
    "contact_feet_mr",
    "contact_feet_rl",
    "contact_feet_rr",
)

CRAB_HEX_TIBIA_BODY_NAMES: tuple[str, ...] = (
    "FL_Tibia",
    "FR_Tibia",
    "ML_Tibia",
    "MR_Tibia",
    "RL_Tibia",
    "RR_Tibia",
)


def crab_hex_tibia_contacts_ready(scene: InteractiveScene) -> bool:
    s = scene.sensors
    return all(k in s for k in CRAB_HEX_TIBIA_CONTACT_KEYS)


def iter_crab_hex_tibia_contact_sensors(scene: InteractiveScene) -> list[ContactSensor]:
    return [scene.sensors[k] for k in CRAB_HEX_TIBIA_CONTACT_KEYS]
