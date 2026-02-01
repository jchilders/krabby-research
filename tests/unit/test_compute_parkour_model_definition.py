"""Unit tests for compute.parkour.model_definition."""

import pytest

from compute.parkour.model_definition import (
    ModelObservationDefinition,
    load_model_definition,
)


def test_load_model_definition_raises_when_file_missing(tmp_path):
    """load_model_definition raises FileNotFoundError when params/model_definition.py is missing."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_model_definition(checkpoint)

    assert "params/model_definition.py" in str(exc_info.value)


def test_load_model_definition_raises_when_checkpoint_dir_has_no_params(tmp_path):
    """load_model_definition raises FileNotFoundError when params dir does not exist."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()

    with pytest.raises(FileNotFoundError):
        load_model_definition(checkpoint)


def test_load_model_definition_returns_definition_from_python_file(tmp_path):
    """load_model_definition returns ModelObservationDefinition from params/model_definition.py."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    definition_py = params_dir / "model_definition.py"
    definition_py.write_text('''from compute.parkour.model_definition import ModelObservationDefinition

model_observation_definition = ModelObservationDefinition(
    num_scan=132,
    num_priv_explicit=9,
    num_priv_latent=29,
    num_hist=10,
    action_dim=12,
)
''')

    result = load_model_definition(checkpoint)

    assert isinstance(result, ModelObservationDefinition)
    assert result.num_scan == 132
    assert result.num_priv_explicit == 9
    assert result.num_priv_latent == 29
    assert result.num_hist == 10
    assert result.action_dim == 12


def test_load_model_definition_custom_action_dim(tmp_path):
    """load_model_definition loads custom action_dim from Python file."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    (params_dir / "model_definition.py").write_text('''from compute.parkour.model_definition import ModelObservationDefinition

model_observation_definition = ModelObservationDefinition(
    num_scan=132,
    num_priv_explicit=9,
    num_priv_latent=29,
    num_hist=10,
    action_dim=18,
)
''')

    result = load_model_definition(checkpoint)

    assert result.action_dim == 18


def test_load_model_definition_raises_when_variable_missing(tmp_path):
    """load_model_definition raises AttributeError when model_observation_definition not defined."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    (params_dir / "model_definition.py").write_text("# empty file, no model_observation_definition\n")

    with pytest.raises(AttributeError) as exc_info:
        load_model_definition(checkpoint)

    assert "model_observation_definition" in str(exc_info.value)


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
        ModelObservationDefinition(num_scan=0).validate()
    with pytest.raises(ValueError, match="action_dim must be > 0"):
        ModelObservationDefinition(action_dim=-1).validate()
