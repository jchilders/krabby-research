"""Unit tests for compute.parkour.model_definition."""

import pytest

from compute.parkour.model_definition import (
    PARKOUR_MODEL_OBSERVATION_DEFINITION,
    ModelObservationDefinition,
)
from hal.server.robot_definition import (
    ObservationScalingDefinition,
    RobotDefinition,
)

_TEST_ROBOT = RobotDefinition(
    name="test_robot",
    legs=("FL", "FR", "RL", "RR"),
    joint_types=("hip_yaw", "hip_pitch", "knee"),
    observation_scaling=ObservationScalingDefinition(
        base_ang_vel=0.25,
        joint_vel=0.05,
        base_lin_vel=2.0,
    ),
)


def test_model_definition_get_observation_dimensions():
    """ModelObservationDefinition.get_observation_dimensions returns ObservationDimensions."""
    model = PARKOUR_MODEL_OBSERVATION_DEFINITION
    dims = model.get_observation_dimensions(_TEST_ROBOT)
    assert dims.num_prop == _TEST_ROBOT.get_num_prop()
    num_prop = _TEST_ROBOT.get_num_prop()
    history_dim = model.num_hist * num_prop
    expected_obs_dim = (
        num_prop
        + model.num_scan
        + model.num_priv_explicit
        + model.num_priv_latent
        + history_dim
    )
    assert dims.obs_dim == expected_obs_dim
    assert dims.observation_joint_count == 12


def test_model_observation_definition_validate_success():
    """ModelObservationDefinition.validate() passes for valid values."""
    defn = ModelObservationDefinition(
        num_scan=132,
        num_priv_explicit=9,
        num_priv_latent=29,
        num_hist=10,
        action_dim=12,
    )
    defn.validate()


def test_model_observation_definition_validate_raises_for_invalid():
    """ModelObservationDefinition.validate() raises for non-positive values."""
    with pytest.raises(ValueError, match="num_scan must be > 0"):
        ModelObservationDefinition(
            num_scan=0,
            num_priv_explicit=9,
            num_priv_latent=29,
            num_hist=10,
            action_dim=12,
        ).validate()
    with pytest.raises(ValueError, match="action_dim must be > 0"):
        ModelObservationDefinition(
            num_scan=132,
            num_priv_explicit=9,
            num_priv_latent=29,
            num_hist=10,
            action_dim=-1,
        ).validate()
