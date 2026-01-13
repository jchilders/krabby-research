"""Hardware data structures for Krabby robot.

These structures represent raw hardware sensor data and desired joint positions.
They are designed for zero-copy operations where possible.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HardwareObservations:
    """Hardware observation data.
    
    Contains all raw sensor data from the hardware:
    - Joint positions (12 DOF)
    - Camera data (2x RGB)
    - Depth map
    - Confidence map
    - Camera resolution metadata (self-describing)
    
    Zero-copy guarantees:
    - Arrays are stored as numpy arrays (may be views or copies depending on source)
    - Large arrays (RGB, depth, confidence) should use views when possible
    - Scalar values (timestamp, camera dimensions) are copied
    """
    
    joint_positions: np.ndarray  # Shape: (12,), dtype: float32
    rgb_camera_1: np.ndarray  # Shape: (camera_height, camera_width, 3), dtype: uint8 or float32
    rgb_camera_2: np.ndarray  # Shape: (camera_height, camera_width, 3), dtype: uint8 or float32
    depth_map: np.ndarray  # Shape: (camera_height, camera_width), dtype: float32
    confidence_map: np.ndarray  # Shape: (camera_height, camera_width), dtype: float32
    camera_height: int  # Height of camera images
    camera_width: int  # Width of camera images
    timestamp_ns: int
    
    # Robot state fields (required - HAL server must provide this data)
    base_ang_vel_b: np.ndarray  # Shape: (3,), dtype: float32 - Base angular velocity (body frame)
    base_lin_vel_b: np.ndarray  # Shape: (3,), dtype: float32 - Base linear velocity (body frame)
    base_quat_w: np.ndarray  # Shape: (4,), dtype: float32 - Base quaternion (world frame, x,y,z,w)
    joint_velocities: np.ndarray  # Shape: (12,), dtype: float32 - Joint velocities
    contact_forces: np.ndarray  # Shape: (5,), dtype: float32 - Contact forces (5 values from environment, normalized to [-0.5, 0.5])
    previous_action: np.ndarray  # Shape: (12,), dtype: float32 - Previous joint command
    
    # Environment-specific fields (for matching environment observation manager)
    delta_yaw: float = 0.0  # Target yaw - current yaw (from parkour_event)
    delta_next_yaw: float = 0.0  # Next target yaw - current yaw (from parkour_event)
    terrain_type_flag: float = 1.0  # 1 if not flat, 0 if flat (from environment)
    flat_terrain_flag: float = 0.0  # 1 if flat, 0 if not flat (from environment)
    scan_features: Optional[np.ndarray] = None  # Shape: (132,), dtype: float32 - Scan features from observation manager (measured_heights)
    privileged_latent: Optional[np.ndarray] = None  # Shape: (29,), dtype: float32 - Privileged latent features (available in simulation, None on hardware)
    
    def __post_init__(self) -> None:
        """Validate hardware observations."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        
        if self.camera_height <= 0 or self.camera_width <= 0:
            raise ValueError(f"Camera dimensions must be positive, got {self.camera_height}x{self.camera_width}")
        
        # Validate joint positions
        if self.joint_positions.shape != (12,):
            raise ValueError(
                f"joint_positions shape {self.joint_positions.shape} != (12,)"
            )
        if self.joint_positions.dtype != np.float32:
            # Convert to float32 if needed (creates copy)
            self.joint_positions = self.joint_positions.astype(np.float32)
        
        # Validate camera arrays (check shape consistency with metadata)
        expected_shape_2d = (self.camera_height, self.camera_width)
        expected_shape_3d = (self.camera_height, self.camera_width, 3)
        
        if self.rgb_camera_1.shape != self.rgb_camera_2.shape:
            raise ValueError(
                f"Camera shapes must match: {self.rgb_camera_1.shape} != {self.rgb_camera_2.shape}"
            )
        if self.rgb_camera_1.shape != expected_shape_3d:
            raise ValueError(
                f"RGB camera must be {expected_shape_3d}, got {self.rgb_camera_1.shape}"
            )
        
        # Validate depth and confidence maps
        if self.depth_map.shape != self.confidence_map.shape:
            raise ValueError(
                f"Depth and confidence shapes must match: {self.depth_map.shape} != {self.confidence_map.shape}"
            )
        if self.depth_map.shape != expected_shape_2d:
            raise ValueError(
                f"Depth map must be {expected_shape_2d}, got {self.depth_map.shape}"
            )
        
        # Ensure depth and confidence are float32
        if self.depth_map.dtype != np.float32:
            self.depth_map = self.depth_map.astype(np.float32)
        if self.confidence_map.dtype != np.float32:
            self.confidence_map = self.confidence_map.astype(np.float32)
        
        # Validate optional robot state fields
        if self.base_ang_vel_b.shape != (3,):
            raise ValueError(f"base_ang_vel_b shape {self.base_ang_vel_b.shape} != (3,)")
        if self.base_lin_vel_b.shape != (3,):
            raise ValueError(f"base_lin_vel_b shape {self.base_lin_vel_b.shape} != (3,)")
        if self.base_quat_w.shape != (4,):
            raise ValueError(f"base_quat_w shape {self.base_quat_w.shape} != (4,)")
        if self.joint_velocities.shape != (12,):
            raise ValueError(f"joint_velocities shape {self.joint_velocities.shape} != (12,)")
        if self.contact_forces.shape != (5,):
            raise ValueError(f"contact_forces shape {self.contact_forces.shape} != (5,)")
        if self.previous_action.shape != (12,):
            raise ValueError(f"previous_action shape {self.previous_action.shape} != (12,)")
        
        # Ensure all are float32
        if self.base_ang_vel_b.dtype != np.float32:
            self.base_ang_vel_b = self.base_ang_vel_b.astype(np.float32)
        if self.base_lin_vel_b.dtype != np.float32:
            self.base_lin_vel_b = self.base_lin_vel_b.astype(np.float32)
        if self.base_quat_w.dtype != np.float32:
            self.base_quat_w = self.base_quat_w.astype(np.float32)
        if self.joint_velocities.dtype != np.float32:
            self.joint_velocities = self.joint_velocities.astype(np.float32)
        if self.contact_forces.dtype != np.float32:
            self.contact_forces = self.contact_forces.astype(np.float32)
        if self.previous_action.dtype != np.float32:
            self.previous_action = self.previous_action.astype(np.float32)
    
    def to_bytes(self) -> list[bytes]:
        """Serialize to bytes for ZMQ transport.
        
        Format: multipart message with metadata and arrays:
        - Part 0: metadata JSON (shapes, dtypes, timestamp)
        - Part 1: joint_positions bytes
        - Part 2: rgb_camera_1 bytes
        - Part 3: rgb_camera_2 bytes
        - Part 4: depth_map bytes
        - Part 5: confidence_map bytes
        - Part 6: base_ang_vel_b bytes
        - Part 7: base_lin_vel_b bytes
        - Part 8: base_quat_w bytes
        - Part 9: joint_velocities bytes
        - Part 10: contact_forces bytes
        - Part 11: previous_action bytes
        
        Returns:
            List of bytes for multipart ZMQ message
        """
        # Ensure arrays are contiguous and correct dtype
        joint_pos = np.ascontiguousarray(self.joint_positions, dtype=np.float32)
        rgb1 = np.ascontiguousarray(self.rgb_camera_1, dtype=self.rgb_camera_1.dtype)
        rgb2 = np.ascontiguousarray(self.rgb_camera_2, dtype=self.rgb_camera_2.dtype)
        depth = np.ascontiguousarray(self.depth_map, dtype=np.float32)
        conf = np.ascontiguousarray(self.confidence_map, dtype=np.float32)
        base_ang_vel_b = np.ascontiguousarray(self.base_ang_vel_b, dtype=np.float32)
        base_lin_vel_b = np.ascontiguousarray(self.base_lin_vel_b, dtype=np.float32)
        base_quat_w = np.ascontiguousarray(self.base_quat_w, dtype=np.float32)
        joint_vel = np.ascontiguousarray(self.joint_velocities, dtype=np.float32)
        contact_forces = np.ascontiguousarray(self.contact_forces, dtype=np.float32)
        prev_action = np.ascontiguousarray(self.previous_action, dtype=np.float32)
        
        # Create metadata
        metadata = {
            "joint_positions": {"shape": list(joint_pos.shape), "dtype": str(joint_pos.dtype)},
            "rgb_camera_1": {"shape": list(rgb1.shape), "dtype": str(rgb1.dtype)},
            "rgb_camera_2": {"shape": list(rgb2.shape), "dtype": str(rgb2.dtype)},
            "depth_map": {"shape": list(depth.shape), "dtype": str(depth.dtype)},
            "confidence_map": {"shape": list(conf.shape), "dtype": str(conf.dtype)},
            "base_ang_vel_b": {"shape": list(base_ang_vel_b.shape), "dtype": str(base_ang_vel_b.dtype)},
            "base_lin_vel_b": {"shape": list(base_lin_vel_b.shape), "dtype": str(base_lin_vel_b.dtype)},
            "base_quat_w": {"shape": list(base_quat_w.shape), "dtype": str(base_quat_w.dtype)},
            "joint_velocities": {"shape": list(joint_vel.shape), "dtype": str(joint_vel.dtype)},
            "contact_forces": {"shape": list(contact_forces.shape), "dtype": str(contact_forces.dtype)},
            "previous_action": {"shape": list(prev_action.shape), "dtype": str(prev_action.dtype)},
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "timestamp_ns": self.timestamp_ns,
            "delta_yaw": self.delta_yaw,
            "delta_next_yaw": self.delta_next_yaw,
            "terrain_type_flag": self.terrain_type_flag,
            "flat_terrain_flag": self.flat_terrain_flag,
        }
        
        # Build base message parts
        parts = [
            json.dumps(metadata).encode("utf-8"),
            joint_pos.tobytes(),
            rgb1.tobytes(),
            rgb2.tobytes(),
            depth.tobytes(),
            conf.tobytes(),
            base_ang_vel_b.tobytes(),
            base_lin_vel_b.tobytes(),
            base_quat_w.tobytes(),
            joint_vel.tobytes(),
            contact_forces.tobytes(),
            prev_action.tobytes(),
        ]
        
        # Add scan_features if available (optional field)
        if self.scan_features is not None:
            scan_features = np.ascontiguousarray(self.scan_features, dtype=np.float32)
            metadata["scan_features"] = {"shape": list(scan_features.shape), "dtype": str(scan_features.dtype)}
            parts.append(scan_features.tobytes())
        
        # Add privileged_latent if available (optional field, only in simulation)
        if self.privileged_latent is not None:
            priv_latent = np.ascontiguousarray(self.privileged_latent, dtype=np.float32)
            metadata["privileged_latent"] = {"shape": list(priv_latent.shape), "dtype": str(priv_latent.dtype)}
            parts.append(priv_latent.tobytes())
        
        # Update metadata JSON with optional fields before returning
        parts[0] = json.dumps(metadata).encode("utf-8")
        
        return parts
    
    @classmethod
    def from_bytes(cls, parts: list[bytes]) -> "HardwareObservations":
        """Deserialize from ZMQ multipart message.
        
        Args:
            parts: List of bytes from ZMQ multipart message
                Expected format: [metadata_json, joint_positions, rgb_camera_1, rgb_camera_2, depth_map, confidence_map, base_ang_vel_b, base_lin_vel_b, base_quat_w, joint_velocities, contact_forces, previous_action]
            
        Returns:
            HardwareObservations instance
            
        Raises:
            ValueError: If message format is invalid (wrong number of parts, invalid JSON, etc.)
        """
        # Support multiple formats for backward compatibility:
        # - Legacy format: 6 parts
        # - Standard format: 12 parts (metadata + 11 arrays)
        # - Extended format: 13 parts (metadata + 12 arrays, includes scan_features OR privileged_latent)
        # - Full format: 14 parts (metadata + 13 arrays, includes scan_features + privileged_latent)
        if len(parts) == 6:
            # Old format - provide defaults for missing fields
            return cls._from_bytes_legacy(parts)
        
        # Parse metadata first to deterministically determine which fields are present
        metadata = json.loads(parts[0].decode("utf-8"))
        
        # Determine which optional fields are present based on metadata and part count
        if len(parts) == 12:
            # Standard format without optional fields
            has_scan_features = False
            has_privileged_latent = False
        elif len(parts) == 13:
            # Extended format: exactly one optional field is present
            # Check metadata to determine which one (scan_features comes before privileged_latent in serialization)
            has_scan_features = "scan_features" in metadata
            has_privileged_latent = "privileged_latent" in metadata
            # Validate that exactly one is present
            if has_scan_features == has_privileged_latent:
                raise ValueError(f"Invalid format: 13 parts requires exactly one optional field, but scan_features={has_scan_features}, privileged_latent={has_privileged_latent}")
        elif len(parts) == 14:
            # Full format with both scan_features and privileged_latent
            has_scan_features = "scan_features" in metadata
            has_privileged_latent = "privileged_latent" in metadata
            # Validate that both are present
            if not (has_scan_features and has_privileged_latent):
                raise ValueError(f"Invalid format: 14 parts requires both optional fields, but scan_features={has_scan_features}, privileged_latent={has_privileged_latent}")
        else:
            raise ValueError(f"Expected 6 parts (legacy), 12 parts (standard), 13 parts (extended), or 14 parts (full), got {len(parts)}")
        
        # Deserialize arrays - let numpy raise errors if shapes/dtypes are wrong
        joint_pos = np.frombuffer(parts[1], dtype=np.dtype(metadata["joint_positions"]["dtype"]))
        joint_pos = joint_pos.reshape(tuple(metadata["joint_positions"]["shape"])).astype(np.float32)
        
        rgb1 = np.frombuffer(parts[2], dtype=np.dtype(metadata["rgb_camera_1"]["dtype"]))
        rgb1 = rgb1.reshape(tuple(metadata["rgb_camera_1"]["shape"]))
        
        rgb2 = np.frombuffer(parts[3], dtype=np.dtype(metadata["rgb_camera_2"]["dtype"]))
        rgb2 = rgb2.reshape(tuple(metadata["rgb_camera_2"]["shape"]))
        
        depth = np.frombuffer(parts[4], dtype=np.dtype(metadata["depth_map"]["dtype"]))
        depth = depth.reshape(tuple(metadata["depth_map"]["shape"])).astype(np.float32)
        
        conf = np.frombuffer(parts[5], dtype=np.dtype(metadata["confidence_map"]["dtype"]))
        conf = conf.reshape(tuple(metadata["confidence_map"]["shape"])).astype(np.float32)
        
        base_ang_vel_b = np.frombuffer(parts[6], dtype=np.dtype(metadata["base_ang_vel_b"]["dtype"]))
        base_ang_vel_b = base_ang_vel_b.reshape(tuple(metadata["base_ang_vel_b"]["shape"])).astype(np.float32)
        
        base_lin_vel_b = np.frombuffer(parts[7], dtype=np.dtype(metadata["base_lin_vel_b"]["dtype"]))
        base_lin_vel_b = base_lin_vel_b.reshape(tuple(metadata["base_lin_vel_b"]["shape"])).astype(np.float32)
        
        base_quat_w = np.frombuffer(parts[8], dtype=np.dtype(metadata["base_quat_w"]["dtype"]))
        base_quat_w = base_quat_w.reshape(tuple(metadata["base_quat_w"]["shape"])).astype(np.float32)
        
        joint_vel = np.frombuffer(parts[9], dtype=np.dtype(metadata["joint_velocities"]["dtype"]))
        joint_vel = joint_vel.reshape(tuple(metadata["joint_velocities"]["shape"])).astype(np.float32)
        
        contact_forces = np.frombuffer(parts[10], dtype=np.dtype(metadata["contact_forces"]["dtype"]))
        contact_forces = contact_forces.reshape(tuple(metadata["contact_forces"]["shape"])).astype(np.float32)
        
        prev_action = np.frombuffer(parts[11], dtype=np.dtype(metadata["previous_action"]["dtype"]))
        prev_action = prev_action.reshape(tuple(metadata["previous_action"]["shape"])).astype(np.float32)
        
        # Extract camera dimensions from metadata or infer from array shapes
        camera_height = metadata.get("camera_height")
        camera_width = metadata.get("camera_width")
        if camera_height is None or camera_width is None:
            # Fallback: extract from rgb_camera_1 shape if not in metadata
            if len(rgb1.shape) >= 2:
                camera_height = rgb1.shape[0]
                camera_width = rgb1.shape[1]
            else:
                raise ValueError("Cannot determine camera dimensions from metadata or array shapes")
        
        return cls(
            joint_positions=joint_pos,
            rgb_camera_1=rgb1,
            rgb_camera_2=rgb2,
            depth_map=depth,
            confidence_map=conf,
            camera_height=camera_height,
            camera_width=camera_width,
            timestamp_ns=metadata.get("timestamp_ns", 0),
            base_ang_vel_b=base_ang_vel_b,
            base_lin_vel_b=base_lin_vel_b,
            base_quat_w=base_quat_w,
            joint_velocities=joint_vel,
            contact_forces=contact_forces,
            previous_action=prev_action,
            delta_yaw=metadata.get("delta_yaw", 0.0),
            delta_next_yaw=metadata.get("delta_next_yaw", 0.0),
            terrain_type_flag=metadata.get("terrain_type_flag", 1.0),
            flat_terrain_flag=metadata.get("flat_terrain_flag", 0.0),
            scan_features=cls._deserialize_scan_features(parts, metadata) if has_scan_features else None,
            privileged_latent=cls._deserialize_privileged_latent(parts, metadata) if has_privileged_latent else None,
        )
    
    @classmethod
    def _deserialize_scan_features(cls, parts: list[bytes], metadata: dict) -> Optional[np.ndarray]:
        """Deserialize scan_features from message parts if available."""
        if "scan_features" not in metadata or len(parts) < 13:
            return None
        scan_features = np.frombuffer(parts[12], dtype=np.dtype(metadata["scan_features"]["dtype"]))
        scan_features = scan_features.reshape(tuple(metadata["scan_features"]["shape"])).astype(np.float32)
        return scan_features
    
    @classmethod
    def _deserialize_privileged_latent(cls, parts: list[bytes], metadata: dict) -> Optional[np.ndarray]:
        """Deserialize privileged_latent from message parts if available."""
        if "privileged_latent" not in metadata:
            return None
        
        # Determine index: scan_features is always at index 12 (if present), privileged_latent is at index 13
        # Serialization order: base (12 parts) + scan_features (part 12) + privileged_latent (part 13)
        # So if privileged_latent is present, it's always at index 13 (after scan_features)
        priv_latent_idx = 13
        
        if len(parts) <= priv_latent_idx:
            raise ValueError(f"Metadata indicates privileged_latent should be present, but not enough parts: len={len(parts)}, need > {priv_latent_idx}")
        
        priv_latent = np.frombuffer(parts[priv_latent_idx], dtype=np.dtype(metadata["privileged_latent"]["dtype"]))
        priv_latent = priv_latent.reshape(tuple(metadata["privileged_latent"]["shape"])).astype(np.float32)
        return priv_latent
    
    @classmethod
    def _from_bytes_legacy(cls, parts: list[bytes]) -> "HardwareObservations":
        """Deserialize from legacy format (6 parts) for backward compatibility."""
        if len(parts) != 6:
            raise ValueError(f"Expected 6 parts (legacy format), got {len(parts)}")
        
        # Parse metadata
        metadata = json.loads(parts[0].decode("utf-8"))
        
        # Deserialize arrays
        joint_pos = np.frombuffer(parts[1], dtype=np.dtype(metadata["joint_positions"]["dtype"]))
        joint_pos = joint_pos.reshape(tuple(metadata["joint_positions"]["shape"])).astype(np.float32)
        
        rgb1 = np.frombuffer(parts[2], dtype=np.dtype(metadata["rgb_camera_1"]["dtype"]))
        rgb1 = rgb1.reshape(tuple(metadata["rgb_camera_1"]["shape"]))
        
        rgb2 = np.frombuffer(parts[3], dtype=np.dtype(metadata["rgb_camera_2"]["dtype"]))
        rgb2 = rgb2.reshape(tuple(metadata["rgb_camera_2"]["shape"]))
        
        depth = np.frombuffer(parts[4], dtype=np.dtype(metadata["depth_map"]["dtype"]))
        depth = depth.reshape(tuple(metadata["depth_map"]["shape"])).astype(np.float32)
        
        conf = np.frombuffer(parts[5], dtype=np.dtype(metadata["confidence_map"]["dtype"]))
        conf = conf.reshape(tuple(metadata["confidence_map"]["shape"])).astype(np.float32)
        
        # Extract camera dimensions
        camera_height = metadata.get("camera_height")
        camera_width = metadata.get("camera_width")
        if camera_height is None or camera_width is None:
            if len(rgb1.shape) >= 2:
                camera_height = rgb1.shape[0]
                camera_width = rgb1.shape[1]
            else:
                raise ValueError("Cannot determine camera dimensions from metadata or array shapes")
        
        # Provide defaults for missing fields
        return cls(
            joint_positions=joint_pos,
            rgb_camera_1=rgb1,
            rgb_camera_2=rgb2,
            depth_map=depth,
            confidence_map=conf,
            camera_height=camera_height,
            camera_width=camera_width,
            timestamp_ns=metadata.get("timestamp_ns", 0),
            base_ang_vel_b=np.zeros(3, dtype=np.float32),
            base_lin_vel_b=np.zeros(3, dtype=np.float32),
            base_quat_w=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            joint_velocities=np.zeros(12, dtype=np.float32),
            contact_forces=np.zeros(5, dtype=np.float32),
            previous_action=np.zeros(12, dtype=np.float32),
            delta_yaw=metadata.get("delta_yaw", 0.0),
            delta_next_yaw=metadata.get("delta_next_yaw", 0.0),
            terrain_type_flag=metadata.get("terrain_type_flag", 1.0),
            flat_terrain_flag=metadata.get("flat_terrain_flag", 0.0),
        )
    


@dataclass
class JointCommand:
    """Joint command structure.
    
    Contains 18 target joint positions for hardware control (6 legs × 3 DOF per leg: hip_yaw, hip_pitch, knee).
    
    Joint Mapping:
        Changed from 12 joints to 18 joints to support hexapod robot configuration.
        - Previous: 12 joints for 4-legged robot (4 legs × 3 DOF = 12 joints)
        - Current: 18 joints for 6-legged hexapod (6 legs × 3 DOF = 18 joints)
        Each leg has 3 degrees of freedom: hip_yaw, hip_pitch, and knee.
    
    Zero-copy guarantees:
    - joint_positions array may be a view if source is compatible
    - timestamp is always copied (scalar)
    """
    
    joint_positions: np.ndarray  # Shape: (18,), dtype: float32
    timestamp_ns: int  # Timestamp when command was created
    observation_timestamp_ns: int  # Timestamp of the observation this command responds to.
        # Used for tracking command-observation relationships and measuring round-trip latency.
    
    def __post_init__(self) -> None:
        """Validate desired joint positions."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        
        if self.joint_positions.shape != (18,):
            raise ValueError(
                f"joint_positions shape {self.joint_positions.shape} != (18,)"
            )
        if self.joint_positions.dtype != np.float32:
            # Convert to float32 if needed (creates copy)
            self.joint_positions = self.joint_positions.astype(np.float32)
        
        # Runtime type validation: Validate values (check for NaN/Inf)
        if not np.isfinite(self.joint_positions).all():
            nan_count = np.isnan(self.joint_positions).sum()
            inf_count = np.isinf(self.joint_positions).sum()
            raise ValueError(f"Invalid joint_positions values: {nan_count} NaN, {inf_count} Inf")
    
    def to_bytes(self) -> list[bytes]:
        """Serialize to bytes for ZMQ transport.
        
        Format: multipart message with metadata and array:
        - Part 0: metadata JSON (shape, dtype, timestamp)
        - Part 1: joint_positions bytes
        
        Returns:
            List of bytes for multipart ZMQ message
        """
        # Ensure array is contiguous and correct dtype
        joint_pos = np.ascontiguousarray(self.joint_positions, dtype=np.float32)
        
        # Create metadata
        metadata = {
            "joint_positions": {"shape": list(joint_pos.shape), "dtype": str(joint_pos.dtype)},
            "timestamp_ns": self.timestamp_ns,
            "observation_timestamp_ns": self.observation_timestamp_ns,
        }
        
        return [
            json.dumps(metadata).encode("utf-8"),
            joint_pos.tobytes(),
        ]
    
    @classmethod
    def from_bytes(cls, parts: list[bytes]) -> "JointCommand":
        """Deserialize from ZMQ multipart message.
        
        Args:
            parts: List of bytes from ZMQ multipart message
                Expected format: [metadata_json, joint_positions]
            
        Returns:
            JointCommand instance
            
        Raises:
            ValueError: If message format is invalid (wrong number of parts, invalid JSON, etc.)
        """
        if len(parts) != 2:
            raise ValueError(f"Expected 2 parts (metadata + array), got {len(parts)}")
        
        # Parse metadata - fail fast on invalid JSON
        metadata = json.loads(parts[0].decode("utf-8"))
        
        # Deserialize array - let numpy raise errors if shapes/dtypes are wrong
        joint_pos = np.frombuffer(parts[1], dtype=np.dtype(metadata["joint_positions"]["dtype"]))
        joint_pos = joint_pos.reshape(tuple(metadata["joint_positions"]["shape"])).astype(np.float32)
        
        return cls(
            joint_positions=joint_pos,
            timestamp_ns=metadata.get("timestamp_ns", 0),
            observation_timestamp_ns=metadata.get("observation_timestamp_ns", 0),
        )
    

