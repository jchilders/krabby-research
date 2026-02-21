"""Unit tests for unified observation format validation.

These tests verify that the observation format matches the layout from
robot + model definitions. Dimensions come from ObservationDimensions
(model_definition.get_observation_dimensions(robot_definition)).
"""

import numpy as np
import pytest

from hal.client.observation.types import NavigationCommand
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from compute.parkour.parkour_types import ParkourObservation, ParkourModelIO
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


@pytest.fixture
def observation_dimensions():
    return PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(_TEST_ROBOT)


class TestObservationDimensions:
    """Test observation dimensions from definitions."""

    def test_dimension_sum_matches_total(self, observation_dimensions):
        """Sum of component dimensions equals obs_dim."""
        d = observation_dimensions
        total = (
            d.num_prop + d.num_scan + d.num_vision
            + d.num_priv_explicit + d.num_priv_latent + d.history_dim
        )
        assert total == d.obs_dim

    def test_history_dimension_calculation(self, observation_dimensions):
        """history_dim == num_hist * num_prop."""
        d = observation_dimensions
        assert d.history_dim == d.num_hist * d.num_prop


class TestObservationStructure:
    """Test observation structure and component positioning."""

    def test_observation_component_positions(self, observation_dimensions):
        """Each component is positioned correctly."""
        d = observation_dimensions
        obs = ParkourObservation.from_parts(
            d,
            proprioceptive=np.full(d.num_prop, 1.0, dtype=np.float32),
            scan=np.full(d.num_scan, 2.0, dtype=np.float32),
            priv_explicit=np.full(d.num_priv_explicit, 3.0, dtype=np.float32),
            priv_latent=np.full(d.num_priv_latent, 4.0, dtype=np.float32),
            history=np.full(d.history_dim, 5.0, dtype=np.float32),
            timestamp_ns=1,
            vision=[] if d.num_vision == 0 else [np.zeros(d.num_vision, dtype=np.float32)],
        )
        arr = obs.to_array()
        assert np.allclose(arr[: d.num_prop], 1.0)
        assert obs.get_proprioceptive().shape == (d.num_prop,)
        assert np.allclose(arr[d.num_prop : d.num_prop + d.num_scan], 2.0)
        assert obs.get_scan().shape == (d.num_scan,)
        start = d.num_prop + d.num_scan + d.num_vision
        assert np.allclose(arr[start : start + d.num_priv_explicit], 3.0)
        assert obs.get_priv_explicit().shape == (d.num_priv_explicit,)
        start += d.num_priv_explicit
        assert np.allclose(arr[start : start + d.num_priv_latent], 4.0)
        assert obs.get_priv_latent().shape == (d.num_priv_latent,)
        assert np.allclose(arr[-d.history_dim :], 5.0)
        assert obs.get_history().shape == (d.history_dim,)

    def test_observation_component_ordering(self, observation_dimensions):
        """Components are in order: prop, scan, camera, priv_explicit, priv_latent, history."""
        d = observation_dimensions
        obs_array = np.arange(d.obs_dim, dtype=np.float32)
        obs = ParkourObservation.from_array(d, obs_array, timestamp_ns=1)
        prop = obs.get_proprioceptive()
        assert prop[0] == 0.0
        assert prop[-1] == d.num_prop - 1
        scan = obs.get_scan()
        assert scan[0] == float(d.num_prop)
        assert scan[-1] == d.num_prop + d.num_scan - 1
        if d.num_vision > 0:
            v = obs.get_vision()
            assert len(v) >= 1 and sum(a.size for a in v) == d.num_vision
        else:
            assert obs.get_vision() == []
        assert obs.get_history()[-1] == d.obs_dim - 1

    def test_observation_no_gaps_or_overlaps(self, observation_dimensions):
        """Components have no gaps or overlaps."""
        d = observation_dimensions
        obs = ParkourObservation.from_parts(
            d,
            proprioceptive=np.zeros(d.num_prop, dtype=np.float32),
            scan=np.zeros(d.num_scan, dtype=np.float32),
            priv_explicit=np.zeros(d.num_priv_explicit, dtype=np.float32),
            priv_latent=np.zeros(d.num_priv_latent, dtype=np.float32),
            history=np.zeros(d.history_dim, dtype=np.float32),
            timestamp_ns=1,
            vision=[] if d.num_vision == 0 else [np.zeros(d.num_vision, dtype=np.float32)],
        )
        obs.get_proprioceptive()[:] = 1.0
        obs.get_scan()[:] = 2.0
        obs.get_priv_explicit()[:] = 3.0
        obs.get_priv_latent()[:] = 4.0
        obs.get_history()[:] = 5.0
        arr = obs.to_array()
        assert np.all(arr > 0)
        assert len(np.unique(arr)) == 5


class TestObservationDataType:
    """Test observation data type requirements."""

    def test_observation_must_be_float32(self, observation_dimensions):
        """Observation parts must be float32 (or converted)."""
        d = observation_dimensions
        obs_array = np.zeros(d.obs_dim, dtype=np.float32)
        obs = ParkourObservation.from_array(d, obs_array, timestamp_ns=1)
        assert obs.to_array().dtype == np.float32
        obs_array_float64 = np.zeros(d.obs_dim, dtype=np.float64)
        obs2 = ParkourObservation.from_array(d, obs_array_float64, timestamp_ns=1)
        assert obs2.to_array().dtype == np.float32

    def test_observation_shape_validation(self, observation_dimensions):
        """Observation shape is validated against obs_dim."""
        d = observation_dimensions
        obs_array = np.zeros(d.obs_dim, dtype=np.float32)
        obs = ParkourObservation.from_array(d, obs_array, timestamp_ns=1)
        assert obs.to_array().shape == (d.obs_dim,)
        with pytest.raises(ValueError, match="Observation shape"):
            wrong = np.zeros(d.obs_dim + 1, dtype=np.float32)
            ParkourObservation.from_array(d, wrong, timestamp_ns=1)


class TestObservationFormatConsistency:
    """Test observation format consistency."""

    def test_parkour_model_io_observation_format(self, observation_dimensions):
        """ParkourModelIO provides observation in correct format (to_array at model boundary)."""
        d = observation_dimensions
        nav_cmd = NavigationCommand.create_now()
        observation = ParkourObservation.from_parts(
            d,
            proprioceptive=np.zeros(d.num_prop, dtype=np.float32),
            scan=np.zeros(d.num_scan, dtype=np.float32),
            priv_explicit=np.zeros(d.num_priv_explicit, dtype=np.float32),
            priv_latent=np.zeros(d.num_priv_latent, dtype=np.float32),
            history=np.zeros(d.history_dim, dtype=np.float32),
            timestamp_ns=1,
            vision=[] if d.num_vision == 0 else [np.zeros(d.num_vision, dtype=np.float32)],
        )
        model_io = ParkourModelIO(
            timestamp_ns=1,
            nav_cmd=nav_cmd,
            observation=observation,
        )
        retrieved = model_io.get_observation_array()
        assert retrieved.shape == (d.obs_dim,)
        assert retrieved.dtype == np.float32
        assert retrieved.flags["C_CONTIGUOUS"]

    def test_observation_from_parts_format(self, observation_dimensions):
        """from_parts() creates observation with correct format."""
        d = observation_dimensions
        prop = np.zeros(d.num_prop, dtype=np.float32)
        scan = np.zeros(d.num_scan, dtype=np.float32)
        vision = [np.zeros(d.num_vision, dtype=np.float32)] if d.num_vision > 0 else []
        priv_explicit = np.zeros(d.num_priv_explicit, dtype=np.float32)
        priv_latent = np.zeros(d.num_priv_latent, dtype=np.float32)
        history = np.zeros(d.history_dim, dtype=np.float32)
        obs = ParkourObservation.from_parts(
            d,
            proprioceptive=prop,
            scan=scan,
            priv_explicit=priv_explicit,
            priv_latent=priv_latent,
            history=history,
            timestamp_ns=1,
            vision=vision if vision else [],
        )
        assert obs.to_array().shape == (d.obs_dim,)
        assert obs.to_array().dtype == np.float32
        assert np.array_equal(obs.get_proprioceptive(), prop)
        assert np.array_equal(obs.get_scan(), scan)
        if d.num_vision == 0:
            assert obs.get_vision() == []
        else:
            assert len(obs.get_vision()) == 1 and np.array_equal(obs.get_vision()[0], vision[0])
        assert np.array_equal(obs.get_priv_explicit(), priv_explicit)
        assert np.array_equal(obs.get_priv_latent(), priv_latent)
        assert np.array_equal(obs.get_history(), history)

    def test_observation_format_matches_training_spec(self, observation_dimensions):
        """Observation layout matches definition."""
        d = observation_dimensions
        obs = ParkourObservation.from_parts(
            d,
            proprioceptive=np.zeros(d.num_prop, dtype=np.float32),
            scan=np.zeros(d.num_scan, dtype=np.float32),
            priv_explicit=np.zeros(d.num_priv_explicit, dtype=np.float32),
            priv_latent=np.zeros(d.num_priv_latent, dtype=np.float32),
            history=np.zeros(d.history_dim, dtype=np.float32),
            timestamp_ns=1,
            vision=[] if d.num_vision == 0 else [np.zeros(d.num_vision, dtype=np.float32)],
        )
        assert len(obs.get_proprioceptive()) == d.num_prop
        assert len(obs.get_scan()) == d.num_scan
        if d.num_vision > 0:
            v = obs.get_vision()
            assert len(v) >= 1 and sum(a.size for a in v) == d.num_vision
        else:
            assert obs.get_vision() == []
        assert len(obs.get_priv_explicit()) == d.num_priv_explicit
        assert len(obs.get_priv_latent()) == d.num_priv_latent
        assert len(obs.get_history()) == d.history_dim
        total = (
            d.num_prop + d.num_scan + d.num_vision
            + d.num_priv_explicit + d.num_priv_latent + d.history_dim
        )
        assert total == d.obs_dim
