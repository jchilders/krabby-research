"""Model observation definition for parkour policy checkpoints.

Model-specific observation dimensions are determined by the model architecture
and training definition, not by robot hardware. Observation dimension for a
policy (obs_dim) is computed from both robot proprio size and model dims here.
"""

from dataclasses import dataclass

from hal.server.robot_definition import RobotDefinition


@dataclass(frozen=True)
class ObservationDimensions:
    """Observation layout dimensions from a robot + model definition pair.

    Single source of truth for array sizes and slice indices into the
    observation vector. Use model_definition.get_observation_dimensions(robot_definition)
    to build this.

    Flat layout order: [proprioceptive, scan, vision, priv_explicit, priv_latent, history].
    ``scan`` is the concatenation of front depth features then optional side depth features:
    total width is ``num_scan == num_scan_front + num_side_scan`` (e.g. 132 or 264).
    vision_dims: per-source sizes (e.g. (64, 64) for two cameras); total vision dim is sum(vision_dims).
    """

    num_prop: int
    observation_joint_count: int
    num_scan_front: int
    num_side_scan: int
    num_scan: int  # total scan slice = num_scan_front + num_side_scan
    vision_dims: tuple[int, ...]  # per-source dims; () when no vision
    num_priv_explicit: int
    num_priv_latent: int
    num_hist: int
    history_dim: int
    obs_dim: int

    @property
    def num_vision(self) -> int:
        """Total vision dimension (sum of vision_dims)."""
        return sum(self.vision_dims)


@dataclass(frozen=True)
class ModelObservationDefinition:
    """Model-specific observation dimension definition.

    These values are determined by the model architecture and training definition,
    not by the robot hardware definition. vision_dims () means no vision.
    """

    num_scan: int
    num_priv_explicit: int
    num_priv_latent: int
    num_hist: int
    action_dim: int
    vision_dims: tuple[int, ...] = ()  # per-source dims; () when no vision
    num_side_scan: int = 0  # extra depth-feature dims (e.g. second ZED); appended after num_scan in flat obs

    def get_observation_dimensions(self, robot_definition: RobotDefinition) -> ObservationDimensions:
        """Observation dimensions for this model and the given robot."""
        num_prop = robot_definition.get_num_prop()
        history_dim = self.num_hist * num_prop
        vision_total = sum(self.vision_dims)
        num_scan_front = self.num_scan
        num_side_scan = self.num_side_scan
        total_scan = num_scan_front + num_side_scan
        obs_dim = (
            num_prop
            + total_scan
            + vision_total
            + self.num_priv_explicit
            + self.num_priv_latent
            + history_dim
        )
        return ObservationDimensions(
            num_prop=num_prop,
            observation_joint_count=robot_definition.get_observation_joint_count(),
            num_scan_front=num_scan_front,
            num_side_scan=num_side_scan,
            num_scan=total_scan,
            vision_dims=self.vision_dims,
            num_priv_explicit=self.num_priv_explicit,
            num_priv_latent=self.num_priv_latent,
            num_hist=self.num_hist,
            history_dim=history_dim,
            obs_dim=obs_dim,
        )

    def validate(self) -> None:
        """Validate model definition values."""
        if any(d < 0 for d in self.vision_dims):
            raise ValueError(f"vision_dims must be non-negative, got {self.vision_dims}")
        if self.num_scan <= 0:
            raise ValueError(f"num_scan must be > 0, got {self.num_scan}")
        if self.num_side_scan < 0:
            raise ValueError(f"num_side_scan must be >= 0, got {self.num_side_scan}")
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


PARKOUR_MODEL_OBSERVATION_DEFINITION = ModelObservationDefinition(
    num_scan=132,
    num_priv_explicit=9,
    num_priv_latent=29,
    num_hist=10,
    action_dim=12,
    vision_dims=(),
    num_side_scan=0,
)

# Front (132) + side (132) depth scan features; requires a checkpoint trained with obs_dim matching.
PARKOUR_MODEL_OBSERVATION_DEFINITION_DUAL_SCAN = ModelObservationDefinition(
    num_scan=132,
    num_priv_explicit=9,
    num_priv_latent=29,
    num_hist=10,
    action_dim=12,
    vision_dims=(),
    num_side_scan=132,
)
