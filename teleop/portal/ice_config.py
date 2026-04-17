"""Browser ``iceServers`` for ``GET /api/teleop-config`` (operator host only — not installed on the robot)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

# Defaults mirror ``teleop.edge.robot_settings.BUILTIN_STUN_SERVERS``; keep lists aligned per deployment.
BUILTIN_STUN_SERVERS: list[dict[str, Any]] = [
    {"urls": "stun:stun.l.google.com:19302"},
]

# Edit like ``teleop.edge.robot_settings.STUN_TURN_SERVERS`` so browser bootstrap matches your robots.
STUN_TURN_SERVERS: list[dict[str, Any]] = copy.deepcopy(BUILTIN_STUN_SERVERS)


@dataclass
class BrowserIceConfig:
    """Shape compatible with how the portal serves ``iceServers``."""

    stun_turn_servers: list[dict[str, Any]]


def build_browser_ice_config() -> BrowserIceConfig:
    """Assemble ICE list for the viewer from module constants above."""
    ice: list[dict[str, Any]] = []
    for item in (STUN_TURN_SERVERS or [])[:32]:
        if isinstance(item, dict) and "urls" in item:
            ice.append(dict(item))
    if not ice:
        ice = copy.deepcopy(BUILTIN_STUN_SERVERS)
    return BrowserIceConfig(stun_turn_servers=ice)
