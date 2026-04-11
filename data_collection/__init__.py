"""HAL data collection: ROS 2 rosbag2 (mcap) recording from HardwareObservations."""

from __future__ import annotations

__all__ = [
    "build_data_collector_config",
    "load_config",
    "DataCollectorConfig",
    "HalDataCollector",
]

from data_collection.collector_settings import build_data_collector_config
from data_collection.config import DataCollectorConfig, load_config


def __getattr__(name: str):
    if name == "HalDataCollector":
        from data_collection.collector import HalDataCollector as _HalDataCollector

        return _HalDataCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
