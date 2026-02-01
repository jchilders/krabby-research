"""Model observation definition for parkour policy checkpoints.

Model-specific observation dimensions are determined by the model architecture
and training definition, not by robot hardware.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelObservationDefinition:
    """Model-specific observation dimension definition.

    These values are determined by the model architecture and training definition,
    not by the robot hardware definition.
    """

    num_scan: int = 132  # Scan/depth features
    num_priv_explicit: int = 9  # Privileged explicit features
    num_priv_latent: int = 29  # Privileged latent features
    num_hist: int = 10  # History buffer length
    action_dim: int = 12  # Model output action dimension

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


DEFAULT_MODEL_OBSERVATION_DEFINITION = ModelObservationDefinition(
    num_scan=132,
    num_priv_explicit=9,
    num_priv_latent=29,
    num_hist=10,
    action_dim=12,
)


def load_model_definition(checkpoint_path: Path) -> ModelObservationDefinition:
    """Load model observation definition from checkpoint directory.

    Expects a Python file at checkpoint_dir/params/model_definition.py that
    defines a variable ``model_observation_definition`` (an instance of
    ModelObservationDefinition).

    Args:
        checkpoint_path: Path to model checkpoint file.

    Returns:
        ModelObservationDefinition with values from the definition file.

    Raises:
        FileNotFoundError: If definition file not found.
        AttributeError: If the definition file does not define model_observation_definition.
    """
    checkpoint_dir = checkpoint_path.parent
    definition_file = checkpoint_dir / "params" / "model_definition.py"

    if not definition_file.exists():
        raise FileNotFoundError(f"Model definition not found at {definition_file}.")

    spec = importlib.util.spec_from_file_location(
        "model_definition", definition_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_def = getattr(module, "model_observation_definition", None)
    if not isinstance(model_def, ModelObservationDefinition):
        raise AttributeError(
            f"{definition_file} must define 'model_observation_definition' "
            "as a ModelObservationDefinition instance"
        )
    model_def.validate()
    return model_def
