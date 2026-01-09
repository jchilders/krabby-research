"""Mapper from Parkour model output to hardware format.

This mapper converts model inference output (joint locomotion embedding)
into hardware joint position commands. It uses zero-copy operations where
possible.
"""

import logging

import numpy as np
import torch

from compute.parkour.parkour_types import InferenceResponse
from hal.client.data_structures.hardware import JointCommand

logger = logging.getLogger(__name__)

# Real Krabby has 18 DOF (joints), but we use 12 for now to match model/environment
KRABBY_JOINT_COUNT = 12


class ParkourLocomotionToHWMapper:
    """Maps Parkour model output to hardware format.
    
    Converts model navigation/locomotion output to hardware joint positions.
    This is a 1:1 mapping since model and hardware both use 12 DOF.
    
    Zero-copy guarantees:
    - If model outputs 12 joints directly, can use view
    - Timestamp is always copied (scalar)
    
    Note: The model outputs ACTION_DIM joints (12 for quadruped).
    Hardware has 12 joints, so this is a 1:1 mapping.
    """
    
    def __init__(self, model_action_dim: int = 12):
        """Initialize the mapper.
        
        Args:
            model_action_dim: Action dimension from the model (typically 12)
        """
        self.model_action_dim = model_action_dim
    
    def map(self, model_output: InferenceResponse, observation_timestamp_ns: int) -> JointCommand:
        """Map model output to hardware joint positions.
        
        Args:
            model_output: Model inference response containing action tensor
            observation_timestamp_ns: Timestamp of the observation this command responds to
            
        Returns:
            JointCommand for hardware control
            
        Raises:
            ValueError: If model output is invalid or failed
        """
        if not model_output.success:
            raise ValueError(f"Model inference failed: {model_output.error_message}")
        
        if model_output.action is None:
            raise ValueError("Model output action is None")
        
        # Get action tensor (zero-copy view)
        action_tensor = model_output.get_action()
        
        # Convert to numpy if needed (creates copy if on GPU, view if on CPU)
        if isinstance(action_tensor, torch.Tensor):
            if action_tensor.is_cuda:
                action_array = action_tensor.cpu().numpy()
            else:
                action_array = action_tensor.numpy()
        else:
            action_array = np.asarray(action_tensor, dtype=np.float32)
        
        # Handle batch dimension
        if action_array.ndim == 2:
            action_array = action_array[0]  # Take first batch element
        elif action_array.ndim != 1:
            raise ValueError(f"Action must be 1D or 2D, got shape {action_array.shape}")
        
        # Ensure float32
        if action_array.dtype != np.float32:
            action_array = action_array.astype(np.float32)
        
        # Map to 12 joints (1:1 mapping)
        joint_positions = self._map_to_krabby_joints(action_array)
        
        return JointCommand(
            joint_positions=joint_positions,
            timestamp_ns=model_output.timestamp_ns,
            observation_timestamp_ns=observation_timestamp_ns,
        )
    
    def _map_to_krabby_joints(self, model_action: np.ndarray) -> np.ndarray:
        """Map model action to Krabby 12-joint positions.
        
        Args:
            model_action: Model action array (shape: ACTION_DIM,)
            
        Returns:
            Joint positions array (shape: 12,)
        """
        if len(model_action) != self.model_action_dim:
            raise ValueError(
                f"Model action dimension {len(model_action)} != expected {self.model_action_dim}"
            )
        
        # Model outputs 12 joints directly, use as-is (1:1 mapping)
        if self.model_action_dim == KRABBY_JOINT_COUNT:
            # Can use view if compatible, otherwise copy
            if model_action.shape == (KRABBY_JOINT_COUNT,) and model_action.dtype == np.float32:
                # Try to use view if contiguous
                if model_action.flags["C_CONTIGUOUS"]:
                    return model_action
                else:
                    return np.ascontiguousarray(model_action, dtype=np.float32)
            else:
                return np.asarray(model_action, dtype=np.float32)
        
        # Should not reach here if model_action_dim == 12 and KRABBY_JOINT_COUNT == 12
        raise ValueError(
            f"Model action dimension {self.model_action_dim} != hardware joint count {KRABBY_JOINT_COUNT}"
        )

