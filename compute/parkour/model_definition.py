"""Model observation definition for parkour policy checkpoints.

Model-specific observation dimensions are determined by the model architecture
and training definition, not by robot hardware. Observation dimension for a
policy (obs_dim) is computed from both robot proprio size and model dims here.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from hal.server.robot_definition import RobotDefinition


@dataclass(frozen=True)
class ObservationDimensions:
    """Observation layout dimensions from a robot + model definition pair.

    Single source of truth for array sizes and slice indices into the
    observation vector. Use model_definition.get_observation_dimensions(robot_definition)
    to build this.
    """

    num_prop: int
    observation_joint_count: int
    num_scan: int
    num_priv_explicit: int
    num_priv_latent: int
    num_hist: int
    history_dim: int
    obs_dim: int


@dataclass(frozen=True)
class ModelObservationDefinition:
    """Model-specific observation dimension definition.

    These values are determined by the model architecture and training definition,
    not by the robot hardware definition. All fields are required (no defaults).
    """

    num_scan: int
    num_priv_explicit: int
    num_priv_latent: int
    num_hist: int
    action_dim: int

    def get_observation_dimensions(self, robot_definition: RobotDefinition) -> ObservationDimensions:
        """Observation dimensions for this model and the given robot."""
        num_prop = robot_definition.get_num_prop()
        history_dim = self.num_hist * num_prop
        obs_dim = (
            num_prop
            + self.num_scan
            + self.num_priv_explicit
            + self.num_priv_latent
            + history_dim
        )
        return ObservationDimensions(
            num_prop=num_prop,
            observation_joint_count=robot_definition.get_observation_joint_count(),
            num_scan=self.num_scan,
            num_priv_explicit=self.num_priv_explicit,
            num_priv_latent=self.num_priv_latent,
            num_hist=self.num_hist,
            history_dim=history_dim,
            obs_dim=obs_dim,
        )

    def get_observation_dimensions_for_checkpoint(
        self, checkpoint_path: str | Path, robot_definition: RobotDefinition
    ) -> ObservationDimensions:
        """Observation dimensions that match a saved checkpoint (for loading).

        Infers num_prop from the checkpoint's state_dict so the policy is built
        with the same layout the checkpoint was trained with. Use this when
        loading a checkpoint for inference instead of get_observation_dimensions(robot_definition).
        """
        num_prop = infer_num_prop_from_checkpoint(checkpoint_path)
        history_dim = self.num_hist * num_prop
        obs_dim = (
            num_prop
            + self.num_scan
            + self.num_priv_explicit
            + self.num_priv_latent
            + history_dim
        )
        return ObservationDimensions(
            num_prop=num_prop,
            observation_joint_count=robot_definition.get_observation_joint_count(),
            num_scan=self.num_scan,
            num_priv_explicit=self.num_priv_explicit,
            num_priv_latent=self.num_priv_latent,
            num_hist=self.num_hist,
            history_dim=history_dim,
            obs_dim=obs_dim,
        )

    def validate(self) -> None:
        """Validate model definition values."""
        if self.num_scan <= 0:
            raise ValueError(f"num_scan must be > 0, got {self.num_scan}")
        if self.num_priv_explicit <= 0:
            raise ValueError(
                f"num_priv_explicit must be > 0, got {self.num_priv_explicit}"
            )
        if self.num_priv_latent <= 0:
            raise ValueError(
                f"num_priv_latent must be > 0, got {self.num_priv_latent}"
            )
        if self.num_hist <= 0:
            raise ValueError(f"num_hist must be > 0, got {self.num_hist}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.action_dim}")


def infer_num_prop_from_checkpoint(checkpoint_path: str | Path) -> int:
    """Infer proprioceptive dimension from a checkpoint's state_dict.

    Reads actor.history_encoder.encoder.0.weight shape (_, num_prop).
    Use when loading a checkpoint for inference so the policy is built with
    the same dimensions the checkpoint was trained with.

    Raises:
        FileNotFoundError: If checkpoint file does not exist
        KeyError: If state_dict does not contain the expected key
        ValueError: If shape cannot be inferred
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = loaded.get("model_state_dict", loaded)
    key = "actor.history_encoder.encoder.0.weight"
    if key not in state_dict:
        raise KeyError(f"Checkpoint missing '{key}'; cannot infer num_prop")
    weight = state_dict[key]
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight at '{key}', got shape {weight.shape}")
    return int(weight.shape[1])


PARKOUR_MODEL_OBSERVATION_DEFINITION = ModelObservationDefinition(
    num_scan=132,
    num_priv_explicit=9,
    num_priv_latent=29,
    num_hist=10,
    action_dim=12,
)
