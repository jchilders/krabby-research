"""Parkour-specific model types.

These types represent Parkour policy model inputs and outputs, including
observations in the exact training format and inference responses.

Observation layout comes from robot + model definitions (ObservationDimensions).
Use model_definition.get_observation_dimensions(robot_definition) to obtain dims.

Zero-Copy Guarantees:
- Large arrays (observation tensors) use views when possible
- Array slicing operations (get_proprioceptive, get_scan, etc.) return views, not copies
- Only copies when dtype conversion is required or arrays are not contiguous
- Scalar values (timestamps) are always copied (necessary)
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import torch

from compute.parkour.model_definition import ObservationDimensions
from hal.client.observation.types import NavigationCommand


@dataclass
class ParkourObservation:
    """Parkour observation as structured parts; convert to array only at model input.

    Layout is [num_prop, num_scan, num_vision, num_priv_explicit, num_priv_latent, history_dim]
    with sizes from observation_dimensions (robot + model definitions).
    When num_vision is 0, vision is an empty list. Observations are stored as
    separate parts; use to_array() right before passing into the model.

    This is the base observation class. For teacher/student separation, use
    TeacherObservation or StudentObservation instead.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        observation_dimensions: Layout dimensions from robot + model definitions
        proprioceptive: Proprioceptive features (num_prop,)
        scan: Scan/depth features (num_scan,)
        vision: List of vision feature arrays (one per camera/source); concatenated in
            order for to_array(). Empty when num_vision is 0.
        priv_explicit: Privileged explicit (num_priv_explicit,)
        priv_latent: Privileged latent (num_priv_latent,)
        history: History features (history_dim,)
    """

    timestamp_ns: int
    observation_dimensions: ObservationDimensions
    proprioceptive: np.ndarray
    scan: np.ndarray
    vision: List[np.ndarray]  # empty when num_vision == 0; one or more arrays otherwise
    priv_explicit: np.ndarray
    priv_latent: np.ndarray
    history: np.ndarray

    def __post_init__(self) -> None:
        """Validate observation parts."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        d = self.observation_dimensions
        if d.num_vision == 0:
            if len(self.vision) != 0:
                raise ValueError("vision must be empty list when num_vision is 0")
        else:
            if not isinstance(self.vision, list) or len(self.vision) == 0:
                raise ValueError("vision must be a non-empty list when num_vision > 0")
            total = 0
            for i, arr in enumerate(self.vision):
                if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                    raise ValueError(f"vision[{i}] must be 1D numpy array")
                if arr.dtype != np.float32:
                    self.vision[i] = np.asarray(arr, dtype=np.float32)
                total += self.vision[i].size
            if total != d.num_vision:
                raise ValueError(
                    f"vision arrays total size {total} != num_vision {d.num_vision}"
                )
        for name, arr, expected in [
            ("proprioceptive", self.proprioceptive, (d.num_prop,)),
            ("scan", self.scan, (d.num_scan,)),
            ("priv_explicit", self.priv_explicit, (d.num_priv_explicit,)),
            ("priv_latent", self.priv_latent, (d.num_priv_latent,)),
            ("history", self.history, (d.history_dim,)),
        ]:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} must be a numpy array")
            if arr.shape != expected:
                raise ValueError(f"{name} shape {arr.shape} != expected {expected}")
            if arr.dtype != np.float32:
                setattr(
                    self,
                    name,
                    arr.astype(np.float32, copy=(arr.dtype != np.float64)),
                )

    def to_array(self) -> np.ndarray:
        """Build flat observation array in training format (for model input only).

        Call this right before passing the observation into the model.
        Order: [proprioceptive, scan, (vision concatenated if non-empty), priv_explicit, priv_latent, history].
        """
        parts = [self.proprioceptive, self.scan]
        if self.vision:
            parts.append(np.concatenate(self.vision))
        parts.extend([self.priv_explicit, self.priv_latent, self.history])
        return np.concatenate(parts)

    def get_proprioceptive(self) -> np.ndarray:
        """Get proprioceptive features as a view."""
        return self.proprioceptive

    def get_scan(self) -> np.ndarray:
        """Get scan/depth features as a view."""
        return self.scan

    def get_vision(self) -> List[np.ndarray]:
        """Get vision features as a list of arrays (one per camera/source). Empty when no vision."""
        return self.vision

    def get_priv_explicit(self) -> np.ndarray:
        """Get privileged explicit features as a view."""
        return self.priv_explicit

    def get_priv_latent(self) -> np.ndarray:
        """Get privileged latent features as a view."""
        return self.priv_latent

    def get_history(self) -> np.ndarray:
        """Get history features as a view."""
        return self.history

    @classmethod
    def from_parts(
        cls,
        observation_dimensions: ObservationDimensions,
        proprioceptive: np.ndarray,
        scan: np.ndarray,
        priv_explicit: np.ndarray,
        priv_latent: np.ndarray,
        history: np.ndarray,
        timestamp_ns: int,
        vision: Union[None, np.ndarray, List[np.ndarray], tuple] = None,
    ) -> "ParkourObservation":
        """Create ParkourObservation from component parts (no concatenation).

        Args:
            observation_dimensions: Layout from robot + model definitions
            proprioceptive: Proprioceptive features (shape: (num_prop,), float32)
            scan: Scan/depth features (shape: (num_scan,), float32)
            priv_explicit: Privileged explicit (shape: (num_priv_explicit,), float32)
            priv_latent: Privileged latent (shape: (num_priv_latent,), float32)
            history: History features (shape: (history_dim,), float32)
            timestamp_ns: Timestamp in nanoseconds
            vision: Vision features: None or [] when num_vision is 0; a single array
                (shape (num_vision,)), or a list/tuple of arrays (one per camera/source)
                whose concatenation has total size num_vision.
        """
        d = observation_dimensions
        if proprioceptive.shape != (d.num_prop,):
            raise ValueError(f"proprioceptive shape {proprioceptive.shape} != ({d.num_prop},)")
        if scan.shape != (d.num_scan,):
            raise ValueError(f"scan shape {scan.shape} != ({d.num_scan},)")
        if priv_explicit.shape != (d.num_priv_explicit,):
            raise ValueError(f"priv_explicit shape {priv_explicit.shape} != ({d.num_priv_explicit},)")
        if priv_latent.shape != (d.num_priv_latent,):
            raise ValueError(f"priv_latent shape {priv_latent.shape} != ({d.num_priv_latent},)")
        if history.shape != (d.history_dim,):
            raise ValueError(f"history shape {history.shape} != ({d.history_dim},)")

        if d.num_vision == 0:
            if vision is not None and vision != []:
                raise ValueError("vision must be None or [] when num_vision is 0")
            vision_list: List[np.ndarray] = []
        else:
            if vision is None or (isinstance(vision, (list, tuple)) and len(vision) == 0):
                raise ValueError("vision must be provided when num_vision > 0")
            if isinstance(vision, np.ndarray):
                vision_list = [np.asarray(vision, dtype=np.float32)]
                if vision_list[0].shape != (d.num_vision,):
                    raise ValueError(f"vision shape {vision_list[0].shape} != ({d.num_vision},)")
            else:
                vision_list = [np.asarray(arr, dtype=np.float32) for arr in vision]
                total = sum(arr.size for arr in vision_list)
                if total != d.num_vision:
                    raise ValueError(
                        f"vision arrays total size {total} != num_vision {d.num_vision}"
                    )

        proprioceptive = np.asarray(proprioceptive, dtype=np.float32)
        scan = np.asarray(scan, dtype=np.float32)
        priv_explicit = np.asarray(priv_explicit, dtype=np.float32)
        priv_latent = np.asarray(priv_latent, dtype=np.float32)
        history = np.asarray(history, dtype=np.float32)

        return cls(
            timestamp_ns=timestamp_ns,
            observation_dimensions=observation_dimensions,
            proprioceptive=proprioceptive,
            scan=scan,
            vision=vision_list,
            priv_explicit=priv_explicit,
            priv_latent=priv_latent,
            history=history,
        )

    @classmethod
    def from_array(
        cls,
        observation_dimensions: ObservationDimensions,
        observation: np.ndarray,
        timestamp_ns: int,
    ) -> "ParkourObservation":
        """Create ParkourObservation from a flat array (e.g. for tests or deserialization).

        Slices the array into parts; no copy of the underlying data when contiguous.
        Order: [proprioceptive, scan, camera, priv_explicit, priv_latent, history].
        """
        d = observation_dimensions
        if observation.shape != (d.obs_dim,):
            raise ValueError(
                f"Observation shape {observation.shape} != expected ({d.obs_dim},)"
            )
        observation = np.asarray(observation, dtype=np.float32)
        proprioceptive = observation[: d.num_prop]
        scan = observation[d.num_prop : d.num_prop + d.num_scan]
        start_vis = d.num_prop + d.num_scan
        if d.num_vision > 0:
            vision_list = [observation[start_vis : start_vis + d.num_vision].copy()]
            start_pe = start_vis + d.num_vision
        else:
            vision_list = []
            start_pe = start_vis
        priv_explicit = observation[start_pe : start_pe + d.num_priv_explicit]
        start_pl = start_pe + d.num_priv_explicit
        priv_latent = observation[start_pl : start_pl + d.num_priv_latent]
        history = observation[-d.history_dim :]
        return cls(
            timestamp_ns=timestamp_ns,
            observation_dimensions=observation_dimensions,
            proprioceptive=proprioceptive.copy(),
            scan=scan.copy(),
            vision=vision_list,
            priv_explicit=priv_explicit.copy(),
            priv_latent=priv_latent.copy(),
            history=history.copy(),
        )


@dataclass
class TeacherObservation(ParkourObservation):
    """Teacher model observation (full sensor suite).
    
    Contains the complete observation with all sensors available during training.
    This is the full observation format used by the teacher model.
    
    Zero-copy guarantees:
    - Inherits zero-copy support from ParkourObservation
    - All sensor data is included in the observation array
    """
    pass


@dataclass
class StudentObservation(ParkourObservation):
    """Student model observation (subset of sensors).
    
    Contains a subset of sensors available to the student model.
    The student model typically has access to fewer sensors than the teacher
    to encourage learning robust representations.
    
    The observation array may have a different dimension than the teacher
    observation, depending on which sensors are excluded.
    
    Zero-copy guarantees:
    - Inherits zero-copy support from ParkourObservation
    - Only includes sensors available to student model
    """
    pass


@dataclass
class ParkourModelIO:
    """Combined input model aggregating all telemetry for policy inference.

    Uses zero-copy observation format matching training exactly.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        nav_cmd: Navigation command (for reference, not in observation)
        observation: Complete observation in training format
    """

    timestamp_ns: int
    nav_cmd: Optional[NavigationCommand] = None
    observation: Optional[ParkourObservation] = None

    def __post_init__(self) -> None:
        """Validate ParkourModelIO."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")

    def is_complete(self) -> bool:
        """Check if all required components are present."""
        return self.nav_cmd is not None and self.observation is not None

    def is_synchronized(self, max_age_ns: int = 10_000_000) -> bool:
        """Check if all components have timestamps within max_age_ns of each other.

        Args:
            max_age_ns: Maximum age difference in nanoseconds (default 10ms)

        Returns:
            True if all components are synchronized, False otherwise
        """
        if not self.is_complete():
            return False

        timestamps = [
            self.nav_cmd.timestamp_ns,
            self.observation.timestamp_ns,
        ]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        return (max_ts - min_ts) <= max_age_ns

    def get_observation_array(self) -> np.ndarray:
        """Build observation array in training format (for model input).

        Converts structured observation parts to flat array right before model.
        """
        if self.observation is None:
            raise ValueError("Observation not set")
        return self.observation.to_array()


@dataclass
class InferenceResponse:
    """Policy inference response matching inference output format exactly.

    The action tensor is in the exact format returned by act_inference:
    - Type: torch.Tensor
    - Shape: (ACTION_DIM,) or (batch, ACTION_DIM)
    - Dtype: torch.float32
    - Device: Same as model device (cuda/cpu)
    - Zero-copy: tensor is used directly without conversion

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        action: Action tensor directly from inference (torch.Tensor)
        success: Whether inference succeeded
        error_message: Error message if success=False (optional)
        timing_breakdown: List of timing measurements as (label, time_ms) tuples
    """

    timestamp_ns: int
    action: Optional[torch.Tensor] = None  # torch.Tensor from act_inference
    success: bool = True
    error_message: Optional[str] = None
    timing_breakdown: list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate inference response."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not self.success and self.error_message is None:
            self.error_message = "Inference failed (no error message provided)"
        if self.action is not None:
            if not isinstance(self.action, torch.Tensor):
                raise ValueError("action must be torch.Tensor")
            if self.action.dtype != torch.float32:
                # Only convert if necessary - prefer views
                self.action = self.action.to(torch.float32)
            if self.action.ndim not in (1, 2):
                raise ValueError(f"action must be 1D or 2D tensor, got shape {self.action.shape}")

    def validate_action_dim(self, action_dim: int) -> None:
        """Validate that action matches expected action dimension.

        Args:
            action_dim: Expected action dimension
        """
        if self.action is None:
            raise ValueError("action is None")
        # Handle both (action_dim,) and (batch, action_dim) shapes
        if self.action.ndim == 1:
            if len(self.action) != action_dim:
                raise ValueError(f"action length {len(self.action)} != action_dim {action_dim}")
        elif self.action.ndim == 2:
            if self.action.shape[1] != action_dim:
                raise ValueError(f"action shape {self.action.shape} != (batch, {action_dim})")
        else:
            raise ValueError(f"action must be 1D or 2D tensor, got shape {self.action.shape}")

    def get_action(self) -> torch.Tensor:
        """Get action tensor as a view (zero-copy).

        Returns:
            Action tensor matching inference output format
        """
        if self.action is None:
            raise ValueError("action not set")
        return self.action

    def get_action_numpy(self) -> np.ndarray:
        """Get action as numpy array (creates copy if on GPU, view if on CPU).

        This is for ZMQ serialization. Only call when needed.

        Returns:
            Action array as numpy (shape: ACTION_DIM, dtype: float32)
        """
        if self.action is None:
            raise ValueError("action not set")
        # Squeeze batch dimension if present
        action = self.action.squeeze(0) if self.action.ndim == 2 else self.action
        # Convert to numpy - shares memory if on CPU, copies if on GPU
        if action.is_cuda:
            return action.cpu().numpy()
        else:
            return action.numpy()

    @classmethod
    def create_success(
        cls,
        action: torch.Tensor,
        timing_breakdown: list[tuple[str, float]],
    ) -> "InferenceResponse":
        """Create successful inference response with action tensor.

        Args:
            action: Action tensor from inference (torch.Tensor from act_inference)
            timing_breakdown: List of (label, time_ms) tuples for timing breakdown

        Returns:
            InferenceResponse with action tensor (zero-copy)
        """
        return cls(
            timestamp_ns=time.time_ns(),
            action=action,  # Direct reference, no copy
            success=True,
            timing_breakdown=timing_breakdown,
        )

    @classmethod
    def create_failure(
        cls,
        error_message: str,
    ) -> "InferenceResponse":
        """Create failed inference response.

        Args:
            error_message: Error message describing the failure

        Returns:
            InferenceResponse with success=False
        """
        return cls(
            timestamp_ns=time.time_ns(),
            action=None,
            success=False,
            error_message=error_message,
        )
