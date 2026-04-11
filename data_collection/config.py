"""HAL data collector configuration (dataclasses).

Defaults for production are assembled in ``data_collection/collector_settings.py``.
``from_dict`` / ``load_config`` remain for tests and optional YAML overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


JointsCommandSource = Literal["previous_action"]
"""`/joints/command` is recorded from `HardwareObservations.previous_action` (last
command applied / echoed in the observation pipeline), not from a separate tap on `put_joint_command`."""


@dataclass
class HalEndpoints:
    """ZMQ endpoint strings (same as `HalClientConfig`).

    For in-proc deployment these must match the primary client and server bind URLs exactly.
    """

    observation_endpoint: str
    command_endpoint: str


@dataclass
class RecordingRates:
    """Target maximum record cadence (wall clock). Latest-only HAL semantics may yield fewer samples."""

    images_hz: float = 10.0
    joints_imu_hz: float = 50.0


@dataclass
class TopicEnable:
    """Which ROS topics to write when data is available."""

    camera_front_rgb: bool = True
    camera_front_depth: bool = True
    camera_side_left_rgb: bool = True
    camera_side_right_rgb: bool = True
    camera_side_rgbd_depth: bool = True
    radar_edge: bool = True
    joints_state: bool = True
    joints_command: bool = True
    imu: bool = True


@dataclass
class CatalogTopicMap:
    """Map `rgbd_by_catalog_id` keys to side/radar topics (Jetson catalog ids, e.g. ``side_rgbd``)."""

    side_left_rgb_catalog_id: Optional[str] = "side_rgbd"
    side_right_rgb_catalog_id: Optional[str] = None
    side_rgbd_depth_catalog_id: Optional[str] = "side_rgbd"
    radar_edge_catalog_id: Optional[str] = None


@dataclass
class DataCollectorConfig:
    hal: HalEndpoints
    output_dir: Path
    max_disk_usage_fraction: float = 0.5
    rotation_max_bytes: int = 1_073_741_824
    rotation_max_minutes: float = 30.0
    rates: RecordingRates = field(default_factory=RecordingRates)
    topics: TopicEnable = field(default_factory=TopicEnable)
    catalog_map: CatalogTopicMap = field(default_factory=CatalogTopicMap)
    joint_names: tuple[str, ...] = ()
    joints_command_source: JointsCommandSource = "previous_action"
    polling_timeout_ms: int = 10

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> DataCollectorConfig:
        hal_raw = raw["hal"]
        hal = HalEndpoints(
            observation_endpoint=str(hal_raw["observation_endpoint"]),
            command_endpoint=str(hal_raw["command_endpoint"]),
        )
        rates_raw = raw.get("rates") or {}
        rates = RecordingRates(
            images_hz=float(rates_raw.get("images_hz", 10.0)),
            joints_imu_hz=float(rates_raw.get("joints_imu_hz", 50.0)),
        )
        topics_raw = raw.get("topics") or {}
        topics = TopicEnable(
            camera_front_rgb=bool(topics_raw.get("camera_front_rgb", True)),
            camera_front_depth=bool(topics_raw.get("camera_front_depth", True)),
            camera_side_left_rgb=bool(topics_raw.get("camera_side_left_rgb", True)),
            camera_side_right_rgb=bool(topics_raw.get("camera_side_right_rgb", True)),
            camera_side_rgbd_depth=bool(topics_raw.get("camera_side_rgbd_depth", True)),
            radar_edge=bool(topics_raw.get("radar_edge", True)),
            joints_state=bool(topics_raw.get("joints_state", True)),
            joints_command=bool(topics_raw.get("joints_command", True)),
            imu=bool(topics_raw.get("imu", True)),
        )
        cmap_raw = raw.get("catalog_map") or {}
        catalog_map = CatalogTopicMap(
            side_left_rgb_catalog_id=_optional_str(cmap_raw.get("side_left_rgb_catalog_id", "side_rgbd")),
            side_right_rgb_catalog_id=_optional_str(cmap_raw.get("side_right_rgb_catalog_id")),
            side_rgbd_depth_catalog_id=_optional_str(cmap_raw.get("side_rgbd_depth_catalog_id", "side_rgbd")),
            radar_edge_catalog_id=_optional_str(cmap_raw.get("radar_edge_catalog_id")),
        )
        jsrc = raw.get("joints_command_source", "previous_action")
        if jsrc != "previous_action":
            raise ValueError(
                f"joints_command_source={jsrc!r} unsupported; only 'previous_action' is implemented "
                "(see docs/DATA_COLLECTOR.md)."
            )
        joint_names_raw = raw.get("joint_names") or []
        joint_names = tuple(str(x) for x in joint_names_raw)
        return DataCollectorConfig(
            hal=hal,
            output_dir=Path(raw["output_dir"]).expanduser(),
            max_disk_usage_fraction=float(raw.get("max_disk_usage_fraction", 0.5)),
            rotation_max_bytes=int(raw.get("rotation_max_bytes", 1_073_741_824)),
            rotation_max_minutes=float(raw.get("rotation_max_minutes", 30.0)),
            rates=rates,
            topics=topics,
            catalog_map=catalog_map,
            joint_names=joint_names,
            joints_command_source="previous_action",
            polling_timeout_ms=int(raw.get("polling_timeout_ms", 10)),
        )


def _optional_str(v: Any) -> Optional[str]:
    if v is None or v == "":
        return None
    return str(v)


def load_config(path: Path | str) -> DataCollectorConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {p} must be a YAML mapping")
    return DataCollectorConfig.from_dict(raw)
