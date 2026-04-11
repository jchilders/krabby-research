"""Unit tests for data_collection.serialization."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rosbags.rosbag2")

from rosbags.typesys import get_typestore
from rosbags.typesys.stores import Stores

from data_collection.config import CatalogTopicMap, TopicEnable
from data_collection.serialization import (
    observation_to_writes,
    serialize_joint_state,
)
from hal.client.data_structures.hardware import RgbdCatalogObservation
from tests.helpers import create_dummy_hw_obs


def _ts():
    return get_typestore(Stores.LATEST)


def test_observation_to_writes_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        observation_to_writes(_ts(), object(), TopicEnable(), CatalogTopicMap(), ())


def test_serialize_joint_state_renames_when_wrong_name_count() -> None:
    ts = _ts()
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vel = np.zeros(3, dtype=np.float32)
    raw = serialize_joint_state(ts, 100, "base", ("only_one",), pos, vel)
    msg = ts.deserialize_cdr(raw, "sensor_msgs/msg/JointState")
    assert list(msg.name) == ["joint_0", "joint_1", "joint_2"]


def test_radar_edge_2d_vs_3d_rgb() -> None:
    ts = _ts()
    topics = TopicEnable(
        camera_front_rgb=False,
        camera_front_depth=False,
        camera_side_left_rgb=False,
        camera_side_right_rgb=False,
        camera_side_rgbd_depth=False,
        joints_state=False,
        joints_command=False,
        imu=False,
        radar_edge=True,
    )
    catalog = CatalogTopicMap(radar_edge_catalog_id="radar_front")

    gray2 = np.zeros((4, 5), dtype=np.uint8)
    rgb3 = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb3[:, :, 1] = 200

    for label, entry in (
        ("2d", RgbdCatalogObservation(rgb=gray2, depth=np.ones((4, 5), dtype=np.float32))),
        ("3d", RgbdCatalogObservation(rgb=rgb3, depth=np.ones((4, 5), dtype=np.float32))),
    ):
        obs = create_dummy_hw_obs(camera_height=4, camera_width=5)
        obs.rgbd_by_catalog_id = {"radar_front": entry}
        rows = observation_to_writes(ts, obs, topics, catalog, ())
        radar = [r for r in rows if r[0] == "/radar/edge"]
        assert len(radar) == 1, label
        img = ts.deserialize_cdr(radar[0][2], "sensor_msgs/msg/Image")
        assert img.encoding == "mono8"


def test_side_left_from_catalog_and_legacy_fallback() -> None:
    ts = _ts()
    topics = TopicEnable(
        camera_front_rgb=False,
        camera_front_depth=False,
        camera_side_left_rgb=True,
        camera_side_right_rgb=False,
        camera_side_rgbd_depth=False,
        joints_state=False,
        joints_command=False,
        imu=False,
        radar_edge=False,
    )
    catalog = CatalogTopicMap(side_left_rgb_catalog_id="side_rgbd")
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    dep = np.ones((2, 3), dtype=np.float32)
    obs = create_dummy_hw_obs(camera_height=2, camera_width=3)
    obs.rgbd_by_catalog_id = {"side_rgbd": RgbdCatalogObservation(rgb=rgb, depth=dep)}
    rows = observation_to_writes(ts, obs, topics, catalog, ())
    assert any(r[0] == "/camera/side_left/rgb" for r in rows)

    obs2 = create_dummy_hw_obs(camera_height=2, camera_width=3)
    obs2.side_camera_rgb = rgb
    obs2.side_camera_depth = dep
    rows2 = observation_to_writes(ts, obs2, topics, CatalogTopicMap(side_left_rgb_catalog_id=None), ())
    assert any(r[0] == "/camera/side_left/rgb" for r in rows2)


def test_side_right_and_depth_catalog() -> None:
    ts = _ts()
    topics = TopicEnable(
        camera_front_rgb=False,
        camera_front_depth=False,
        camera_side_left_rgb=False,
        camera_side_right_rgb=True,
        camera_side_rgbd_depth=True,
        joints_state=False,
        joints_command=False,
        imu=False,
        radar_edge=False,
    )
    catalog = CatalogTopicMap(
        side_right_rgb_catalog_id="rightcam",
        side_rgbd_depth_catalog_id="depthcam",
        side_left_rgb_catalog_id=None,
    )
    r_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    d_dep = np.ones((3, 3), dtype=np.float32)
    obs = create_dummy_hw_obs(camera_height=2, camera_width=2)
    obs.rgbd_by_catalog_id = {
        "rightcam": RgbdCatalogObservation(rgb=r_rgb, depth=np.ones((2, 2), dtype=np.float32)),
        "depthcam": RgbdCatalogObservation(rgb=np.zeros((3, 3, 3), dtype=np.uint8), depth=d_dep),
    }
    rows = observation_to_writes(ts, obs, topics, catalog, ())
    topics_found = {r[0] for r in rows}
    assert "/camera/side_right/rgb" in topics_found
    assert "/camera/side_rgbd/depth" in topics_found


def test_split_stamp_via_header_timestamp() -> None:
    ts = _ts()
    obs = create_dummy_hw_obs()
    obs.timestamp_ns = 3_500_000_123
    topics = TopicEnable(
        camera_front_rgb=False,
        camera_front_depth=False,
        camera_side_left_rgb=False,
        camera_side_right_rgb=False,
        camera_side_rgbd_depth=False,
        radar_edge=False,
        joints_state=False,
        joints_command=False,
        imu=True,
    )
    rows = observation_to_writes(ts, obs, topics, CatalogTopicMap(), ())
    imu_row = next(r for r in rows if r[0] == "/imu")
    msg = ts.deserialize_cdr(imu_row[2], "sensor_msgs/msg/Imu")
    assert msg.header.stamp.sec == 3
    assert msg.header.stamp.nanosec == 500_000_123
