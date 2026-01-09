"""Unit tests for HardwareObservations serialization/deserialization."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# Import from installed package
from hal.client.data_structures.hardware import HardwareObservations


def test_serialize_deserialize_with_privileged_latent():
    """Test that privileged_latent is correctly serialized and deserialized."""
    # Create test data
    privileged_latent = np.array([6.921, 1.234, 0.567] + [0.0] * 26, dtype=np.float32)  # 29 values
    scan_features = np.ones(132, dtype=np.float32) * 0.5
    
    # Create HardwareObservations with privileged_latent
    hw_obs = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        rgb_camera_1=np.zeros((480, 640, 3), dtype=np.uint8),
        rgb_camera_2=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_map=np.zeros((480, 640), dtype=np.float32),
        confidence_map=np.ones((480, 640), dtype=np.float32),
        camera_height=480,
        camera_width=640,
        timestamp_ns=1234567890,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        scan_features=scan_features,
        privileged_latent=privileged_latent,
    )
    
    # Serialize
    parts = hw_obs.to_bytes()
    
    # Verify serialization
    assert len(parts) == 14, f"Expected 14 parts (12 base + scan_features + privileged_latent), got {len(parts)}"
    
    # Parse metadata to verify privileged_latent is included
    import json
    metadata = json.loads(parts[0].decode("utf-8"))
    assert "privileged_latent" in metadata, "privileged_latent not in metadata"
    assert metadata["privileged_latent"]["shape"] == [29], f"Expected shape [29], got {metadata['privileged_latent']['shape']}"
    
    # Deserialize
    hw_obs_deserialized = HardwareObservations.from_bytes(parts)
    
    # Verify privileged_latent is preserved
    assert hw_obs_deserialized.privileged_latent is not None, "privileged_latent is None after deserialization"
    assert hw_obs_deserialized.privileged_latent.shape == (29,), f"Expected shape (29,), got {hw_obs_deserialized.privileged_latent.shape}"
    assert np.allclose(
        hw_obs_deserialized.privileged_latent, privileged_latent
    ), f"privileged_latent mismatch: expected {privileged_latent}, got {hw_obs_deserialized.privileged_latent}"
    
    # Verify first value matches
    assert np.isclose(
        hw_obs_deserialized.privileged_latent[0], 6.921
    ), f"First value mismatch: expected 6.921, got {hw_obs_deserialized.privileged_latent[0]}"


def test_serialize_deserialize_without_privileged_latent():
    """Test that serialization works when privileged_latent is None (hardware case)."""
    scan_features = np.ones(132, dtype=np.float32) * 0.5
    
    # Create HardwareObservations without privileged_latent
    hw_obs = HardwareObservations(
        joint_positions=np.zeros(12, dtype=np.float32),
        rgb_camera_1=np.zeros((480, 640, 3), dtype=np.uint8),
        rgb_camera_2=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_map=np.zeros((480, 640), dtype=np.float32),
        confidence_map=np.ones((480, 640), dtype=np.float32),
        camera_height=480,
        camera_width=640,
        timestamp_ns=1234567890,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0, 0, 0, 1], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.zeros(12, dtype=np.float32),
        scan_features=scan_features,
        privileged_latent=None,  # Not available on hardware
    )
    
    # Serialize
    parts = hw_obs.to_bytes()
    
    # Verify serialization (13 parts: 12 base + scan_features, no privileged_latent)
    assert len(parts) == 13, f"Expected 13 parts (12 base + scan_features), got {len(parts)}"
    
    # Parse metadata to verify privileged_latent is NOT included
    import json
    metadata = json.loads(parts[0].decode("utf-8"))
    assert "privileged_latent" not in metadata, "privileged_latent should not be in metadata when None"
    
    # Deserialize
    hw_obs_deserialized = HardwareObservations.from_bytes(parts)
    
    # Verify privileged_latent is None
    assert hw_obs_deserialized.privileged_latent is None, "privileged_latent should be None after deserialization"


def test_serialize_deserialize_roundtrip():
    """Test full roundtrip: serialize -> deserialize -> serialize -> deserialize."""
    privileged_latent = np.array([6.921, 1.234, 0.567] + [0.0] * 26, dtype=np.float32)
    scan_features = np.ones(132, dtype=np.float32) * 0.5
    
    # Create original
    hw_obs_original = HardwareObservations(
        joint_positions=np.random.randn(12).astype(np.float32),
        rgb_camera_1=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        rgb_camera_2=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        depth_map=np.random.randn(480, 640).astype(np.float32),
        confidence_map=np.random.rand(480, 640).astype(np.float32),
        camera_height=480,
        camera_width=640,
        timestamp_ns=1234567890,
        base_ang_vel_b=np.random.randn(3).astype(np.float32),
        base_lin_vel_b=np.random.randn(3).astype(np.float32),
        base_quat_w=np.random.randn(4).astype(np.float32),
        joint_velocities=np.random.randn(12).astype(np.float32),
        contact_forces=np.random.randn(5).astype(np.float32),
        previous_action=np.random.randn(12).astype(np.float32),
        scan_features=scan_features,
        privileged_latent=privileged_latent,
    )
    
    # Roundtrip 1
    parts1 = hw_obs_original.to_bytes()
    hw_obs_round1 = HardwareObservations.from_bytes(parts1)
    
    # Roundtrip 2
    parts2 = hw_obs_round1.to_bytes()
    hw_obs_round2 = HardwareObservations.from_bytes(parts2)
    
    # Verify privileged_latent is preserved through both roundtrips
    assert hw_obs_round2.privileged_latent is not None, "privileged_latent lost in roundtrip"
    assert np.allclose(
        hw_obs_round2.privileged_latent, privileged_latent
    ), f"privileged_latent mismatch after roundtrip: expected {privileged_latent}, got {hw_obs_round2.privileged_latent}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

