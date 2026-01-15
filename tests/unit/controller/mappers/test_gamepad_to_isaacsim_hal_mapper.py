"""Unit tests for GamepadToIsaacSimHALMapper.

How to Run the Tests:
====================

From the krabby-research directory, run:

    # Run all tests in this file
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_isaacsim_hal_mapper.py -v

    # Run a specific test class
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_isaacsim_hal_mapper.py::TestMapperInitialization -v

    # Run a specific test
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_isaacsim_hal_mapper.py::TestMapperInitialization::test_default_initialization -v

    # Run with coverage
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_isaacsim_hal_mapper.py --cov=controller.mappers --cov-report=html

    # Run with verbose output and show print statements
    python -m pytest tests/unit/controller/mappers/test_gamepad_to_isaacsim_hal_mapper.py -v -s


Prerequisites:
    - pytest: pip install pytest
    - pytest-cov (optional, for coverage): pip install pytest-cov

Note: These tests use mocking and do not require actual gamepad hardware or HAL server.

Test Coverage:
==============

These tests verify:
- Initialization with default and custom scaling factors
- Single leg mapping with axis values
- Multiple leg mapping
- Incremental control (state persistence)
- No legs selected handling
- Joint position clamping
- Timestamp handling
- Reset functionality
- Error handling
- Edge cases
"""

import time
from unittest.mock import patch

import numpy as np
import pytest

from controller.input.state import ControllerState, GamepadControlData, LegIdentifier
from controller.mappers.gamepad_to_isaacsim_hal_mapper import (
    DEFAULT_HIP_UP_DOWN_SCALE,
    DEFAULT_HIP_YAW_SCALE,
    DEFAULT_KNEE_OUT_IN_SCALE,
    GamepadToIsaacSimHALMapper,
    LEG_TO_JOINT_INDICES,
)
from hal.client.data_structures.hardware import JointCommand


class TestMapperInitialization:
    """Test mapper initialization."""

    def test_default_initialization(self):
        """Test mapper initialization with default scaling factors."""
        mapper = GamepadToIsaacSimHALMapper()
        
        assert mapper.hip_up_down_scale == DEFAULT_HIP_UP_DOWN_SCALE
        assert mapper.knee_out_in_scale == DEFAULT_KNEE_OUT_IN_SCALE
        assert mapper.hip_yaw_scale == DEFAULT_HIP_YAW_SCALE
        assert mapper._last_joint_positions.shape == (18,)
        assert mapper._last_joint_positions.dtype == np.float32
        assert np.allclose(mapper._last_joint_positions, 0.0)

    def test_custom_initialization(self):
        """Test mapper initialization with custom scaling factors."""
        mapper = GamepadToIsaacSimHALMapper(
            hip_up_down_scale=0.5,
            knee_out_in_scale=0.4,
            hip_yaw_scale=0.3,
        )
        
        assert mapper.hip_up_down_scale == 0.5
        assert mapper.knee_out_in_scale == 0.4
        assert mapper.hip_yaw_scale == 0.3
        assert mapper._last_joint_positions.shape == (18,)
        assert np.allclose(mapper._last_joint_positions, 0.0)


class TestMapperSingleLegMapping:
    """Test mapping single leg control data."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToIsaacSimHALMapper()

    def test_map_front_left_leg(self):
        """Test mapping Front Left leg with axis values."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.5,
            knee_out_in=-0.3,
            hip_yaw=0.2,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        assert isinstance(joint_cmd, JointCommand)
        assert joint_cmd.joint_positions.shape == (18,)
        assert joint_cmd.joint_positions.dtype == np.float32
        
        # FL leg indices: 0 (hip_yaw), 1 (hip_pitch), 2 (knee)
        hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        
        # Check that only FL leg joints are modified (others remain at 0)
        assert joint_cmd.joint_positions[hip_yaw_idx] == pytest.approx(0.2 * DEFAULT_HIP_YAW_SCALE)
        assert joint_cmd.joint_positions[hip_pitch_idx] == pytest.approx(0.5 * DEFAULT_HIP_UP_DOWN_SCALE)
        assert joint_cmd.joint_positions[knee_idx] == pytest.approx(-0.3 * DEFAULT_KNEE_OUT_IN_SCALE)
        
        # Other joints should remain at zero
        other_indices = [i for i in range(18) if i not in [hip_yaw_idx, hip_pitch_idx, knee_idx]]
        assert np.allclose(joint_cmd.joint_positions[other_indices], 0.0)

    def test_map_front_right_leg(self):
        """Test mapping Front Right leg."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_RIGHT},
            hip_up_down=0.4,
            knee_out_in=0.6,
            hip_yaw=-0.1,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        # FR leg indices: 3 (hip_yaw), 4 (hip_pitch), 5 (knee)
        hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_RIGHT]
        
        assert joint_cmd.joint_positions[hip_yaw_idx] == pytest.approx(-0.1 * DEFAULT_HIP_YAW_SCALE)
        assert joint_cmd.joint_positions[hip_pitch_idx] == pytest.approx(0.4 * DEFAULT_HIP_UP_DOWN_SCALE)
        assert joint_cmd.joint_positions[knee_idx] == pytest.approx(0.6 * DEFAULT_KNEE_OUT_IN_SCALE)

    def test_map_all_single_legs(self):
        """Test mapping each leg individually."""
        state = ControllerState()
        legs = [
            LegIdentifier.FRONT_LEFT,
            LegIdentifier.FRONT_RIGHT,
            LegIdentifier.MIDDLE_LEFT,
            LegIdentifier.MIDDLE_RIGHT,
            LegIdentifier.REAR_LEFT,
            LegIdentifier.REAR_RIGHT,
        ]
        
        for leg in legs:
            control_data = GamepadControlData(
                selected_legs={leg},
                hip_up_down=0.5,
                knee_out_in=0.5,
                hip_yaw=0.5,
                raw_state=state,
            )
            
            joint_cmd = self.mapper.map(control_data)
            hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[leg]
            
            # All should have the same scaled values
            expected = 0.5 * DEFAULT_HIP_UP_DOWN_SCALE
            assert joint_cmd.joint_positions[hip_pitch_idx] == pytest.approx(expected)
            assert joint_cmd.joint_positions[knee_idx] == pytest.approx(0.5 * DEFAULT_KNEE_OUT_IN_SCALE)
            assert joint_cmd.joint_positions[hip_yaw_idx] == pytest.approx(0.5 * DEFAULT_HIP_YAW_SCALE)
            
            # Reset for next iteration
            self.mapper.reset()


class TestMapperMultipleLegsMapping:
    """Test mapping multiple legs simultaneously."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToIsaacSimHALMapper()

    def test_map_two_legs(self):
        """Test mapping two legs at once."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT, LegIdentifier.FRONT_RIGHT},
            hip_up_down=0.5,
            knee_out_in=-0.3,
            hip_yaw=0.2,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        # Check FL leg
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        assert joint_cmd.joint_positions[fl_hip_yaw] == pytest.approx(0.2 * DEFAULT_HIP_YAW_SCALE)
        assert joint_cmd.joint_positions[fl_hip_pitch] == pytest.approx(0.5 * DEFAULT_HIP_UP_DOWN_SCALE)
        assert joint_cmd.joint_positions[fl_knee] == pytest.approx(-0.3 * DEFAULT_KNEE_OUT_IN_SCALE)
        
        # Check FR leg
        fr_hip_yaw, fr_hip_pitch, fr_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_RIGHT]
        assert joint_cmd.joint_positions[fr_hip_yaw] == pytest.approx(0.2 * DEFAULT_HIP_YAW_SCALE)
        assert joint_cmd.joint_positions[fr_hip_pitch] == pytest.approx(0.5 * DEFAULT_HIP_UP_DOWN_SCALE)
        assert joint_cmd.joint_positions[fr_knee] == pytest.approx(-0.3 * DEFAULT_KNEE_OUT_IN_SCALE)

    def test_map_tripod_left(self):
        """Test mapping left tripod (FL, RL, MR)."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={
                LegIdentifier.FRONT_LEFT,
                LegIdentifier.REAR_LEFT,
                LegIdentifier.MIDDLE_RIGHT,
            },
            hip_up_down=0.3,
            knee_out_in=0.4,
            hip_yaw=-0.2,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        # Check all three legs have the same values
        for leg in [LegIdentifier.FRONT_LEFT, LegIdentifier.REAR_LEFT, LegIdentifier.MIDDLE_RIGHT]:
            hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[leg]
            assert joint_cmd.joint_positions[hip_yaw_idx] == pytest.approx(-0.2 * DEFAULT_HIP_YAW_SCALE)
            assert joint_cmd.joint_positions[hip_pitch_idx] == pytest.approx(0.3 * DEFAULT_HIP_UP_DOWN_SCALE)
            assert joint_cmd.joint_positions[knee_idx] == pytest.approx(0.4 * DEFAULT_KNEE_OUT_IN_SCALE)

    def test_map_all_legs(self):
        """Test mapping all six legs simultaneously."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={
                LegIdentifier.FRONT_LEFT,
                LegIdentifier.FRONT_RIGHT,
                LegIdentifier.MIDDLE_LEFT,
                LegIdentifier.MIDDLE_RIGHT,
                LegIdentifier.REAR_LEFT,
                LegIdentifier.REAR_RIGHT,
            },
            hip_up_down=0.1,
            knee_out_in=0.2,
            hip_yaw=0.3,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        # All legs should have the same values
        for leg in LegIdentifier:
            hip_yaw_idx, hip_pitch_idx, knee_idx = LEG_TO_JOINT_INDICES[leg]
            assert joint_cmd.joint_positions[hip_yaw_idx] == pytest.approx(0.3 * DEFAULT_HIP_YAW_SCALE)
            assert joint_cmd.joint_positions[hip_pitch_idx] == pytest.approx(0.1 * DEFAULT_HIP_UP_DOWN_SCALE)
            assert joint_cmd.joint_positions[knee_idx] == pytest.approx(0.2 * DEFAULT_KNEE_OUT_IN_SCALE)


class TestMapperIncrementalControl:
    """Test incremental control (state persistence)."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToIsaacSimHALMapper()

    def test_incremental_control_accumulation(self):
        """Test that joint positions accumulate across multiple calls."""
        state = ControllerState()
        
        # First call: move FL leg
        control_data1 = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.5,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        joint_cmd1 = self.mapper.map(control_data1)
        
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        expected_hip_pitch_1 = 0.5 * DEFAULT_HIP_UP_DOWN_SCALE
        assert joint_cmd1.joint_positions[fl_hip_pitch] == pytest.approx(expected_hip_pitch_1)
        
        # Second call: move FL leg again
        control_data2 = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.3,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        joint_cmd2 = self.mapper.map(control_data2)
        
        # Should accumulate: 0.5 + 0.3 = 0.8
        expected_hip_pitch_2 = (0.5 + 0.3) * DEFAULT_HIP_UP_DOWN_SCALE
        assert joint_cmd2.joint_positions[fl_hip_pitch] == pytest.approx(expected_hip_pitch_2)

    def test_incremental_control_different_legs(self):
        """Test that different legs maintain independent state."""
        state = ControllerState()
        
        # Move FL leg
        control_data1 = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.5,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        joint_cmd1 = self.mapper.map(control_data1)
        
        # Move FR leg
        control_data2 = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_RIGHT},
            hip_up_down=0.3,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        joint_cmd2 = self.mapper.map(control_data2)
        
        # FL should still have its value
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        assert joint_cmd2.joint_positions[fl_hip_pitch] == pytest.approx(0.5 * DEFAULT_HIP_UP_DOWN_SCALE)
        
        # FR should have its value
        fr_hip_yaw, fr_hip_pitch, fr_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_RIGHT]
        assert joint_cmd2.joint_positions[fr_hip_pitch] == pytest.approx(0.3 * DEFAULT_HIP_UP_DOWN_SCALE)


class TestMapperNoLegsSelected:
    """Test behavior when no legs are selected."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToIsaacSimHALMapper()

    def test_no_legs_selected(self):
        """Test that no legs selected maintains current positions."""
        state = ControllerState()
        
        # First, move a leg to establish state
        control_data1 = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.5,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        joint_cmd1 = self.mapper.map(control_data1)
        
        # Then, no legs selected
        control_data2 = GamepadControlData(
            selected_legs=set(),
            hip_up_down=0.0,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        joint_cmd2 = self.mapper.map(control_data2)
        
        # Positions should remain the same
        assert np.allclose(joint_cmd1.joint_positions, joint_cmd2.joint_positions)

    def test_no_legs_selected_initial(self):
        """Test no legs selected when mapper is fresh (all zeros)."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs=set(),
            hip_up_down=0.0,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        # All positions should be zero
        assert np.allclose(joint_cmd.joint_positions, 0.0)


class TestMapperJointClamping:
    """Test joint position clamping."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToIsaacSimHALMapper()

    def test_joint_clamping_positive(self):
        """Test that joint positions are clamped to [-2.0, 2.0]."""
        state = ControllerState()
        
        # Use very large values that would exceed limits
        # With default scale of 0.3, we need values > 6.67 to exceed 2.0
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=10.0,  # Would be 3.0 without clamping
            knee_out_in=10.0,
            hip_yaw=10.0,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        # All values should be clamped to [-2.0, 2.0]
        assert np.all(joint_cmd.joint_positions >= -2.0)
        assert np.all(joint_cmd.joint_positions <= 2.0)
        
        # The specific leg should be at the limit
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        assert joint_cmd.joint_positions[fl_hip_pitch] == pytest.approx(2.0)
        assert joint_cmd.joint_positions[fl_knee] == pytest.approx(2.0)
        assert joint_cmd.joint_positions[fl_hip_yaw] == pytest.approx(2.0)

    def test_joint_clamping_negative(self):
        """Test negative clamping."""
        state = ControllerState()
        
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=-10.0,
            knee_out_in=-10.0,
            hip_yaw=-10.0,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        # All values should be clamped
        assert np.all(joint_cmd.joint_positions >= -2.0)
        assert np.all(joint_cmd.joint_positions <= 2.0)
        
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        assert joint_cmd.joint_positions[fl_hip_pitch] == pytest.approx(-2.0)
        assert joint_cmd.joint_positions[fl_knee] == pytest.approx(-2.0)
        assert joint_cmd.joint_positions[fl_hip_yaw] == pytest.approx(-2.0)


class TestMapperTimestamps:
    """Test timestamp handling."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToIsaacSimHALMapper()

    def test_timestamp_generation(self):
        """Test that timestamps are generated correctly."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.5,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        
        before_ns = time.time_ns()
        joint_cmd = self.mapper.map(control_data)
        after_ns = time.time_ns()
        
        # Timestamp should be within the time window
        assert before_ns <= joint_cmd.timestamp_ns <= after_ns
        assert joint_cmd.timestamp_ns > 0
        assert joint_cmd.observation_timestamp_ns > 0

    def test_observation_timestamp_provided(self):
        """Test that provided observation timestamp is used."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.5,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        
        observation_ts = 1234567890123456789
        joint_cmd = self.mapper.map(control_data, observation_timestamp_ns=observation_ts)
        
        assert joint_cmd.observation_timestamp_ns == observation_ts
        assert joint_cmd.timestamp_ns != observation_ts  # Should be different (current time)

    def test_observation_timestamp_none(self):
        """Test that None observation timestamp uses current time."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.5,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        
        before_ns = time.time_ns()
        joint_cmd = self.mapper.map(control_data, observation_timestamp_ns=None)
        after_ns = time.time_ns()
        
        # Both timestamps should be within the time window
        assert before_ns <= joint_cmd.timestamp_ns <= after_ns
        assert before_ns <= joint_cmd.observation_timestamp_ns <= after_ns


class TestMapperReset:
    """Test reset functionality."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToIsaacSimHALMapper()

    def test_reset_clears_state(self):
        """Test that reset clears joint positions to zero."""
        state = ControllerState()
        
        # Move a leg
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.5,
            knee_out_in=0.3,
            hip_yaw=0.2,
            raw_state=state,
        )
        self.mapper.map(control_data)
        
        # Verify state is non-zero
        assert not np.allclose(self.mapper._last_joint_positions, 0.0)
        
        # Reset
        self.mapper.reset()
        
        # State should be zero
        assert np.allclose(self.mapper._last_joint_positions, 0.0)
        assert self.mapper._last_joint_positions.shape == (18,)
        assert self.mapper._last_joint_positions.dtype == np.float32

    def test_reset_after_multiple_calls(self):
        """Test reset after multiple mapping calls."""
        state = ControllerState()
        
        # Multiple calls
        for _ in range(5):
            control_data = GamepadControlData(
                selected_legs={LegIdentifier.FRONT_LEFT},
                hip_up_down=0.1,
                knee_out_in=0.0,
                hip_yaw=0.0,
                raw_state=state,
            )
            self.mapper.map(control_data)
        
        # Reset
        self.mapper.reset()
        
        # Next call should start from zero
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.5,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        joint_cmd = self.mapper.map(control_data)
        
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        assert joint_cmd.joint_positions[fl_hip_pitch] == pytest.approx(0.5 * DEFAULT_HIP_UP_DOWN_SCALE)


class TestMapperErrorHandling:
    """Test error handling."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToIsaacSimHALMapper()

    def test_invalid_control_data_type(self):
        """Test that invalid control_data type raises ValueError."""
        with pytest.raises(ValueError, match="control_data must be GamepadControlData"):
            self.mapper.map("invalid")  # type: ignore

    def test_invalid_control_data_none(self):
        """Test that None control_data raises ValueError."""
        with pytest.raises(ValueError, match="control_data must be GamepadControlData"):
            self.mapper.map(None)  # type: ignore


class TestMapperCustomScaling:
    """Test mapper with custom scaling factors."""

    def test_custom_scaling_factors(self):
        """Test that custom scaling factors are applied correctly."""
        mapper = GamepadToIsaacSimHALMapper(
            hip_up_down_scale=0.5,
            knee_out_in_scale=0.4,
            hip_yaw_scale=0.3,
        )
        
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=1.0,
            knee_out_in=1.0,
            hip_yaw=1.0,
            raw_state=state,
        )
        
        joint_cmd = mapper.map(control_data)
        
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        assert joint_cmd.joint_positions[fl_hip_pitch] == pytest.approx(0.5)
        assert joint_cmd.joint_positions[fl_knee] == pytest.approx(0.4)
        assert joint_cmd.joint_positions[fl_hip_yaw] == pytest.approx(0.3)


class TestMapperEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Create a fresh mapper for each test."""
        self.mapper = GamepadToIsaacSimHALMapper()

    def test_zero_axis_values(self):
        """Test mapping with zero axis values."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.0,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        # Joint positions should remain at zero
        assert np.allclose(joint_cmd.joint_positions, 0.0)

    def test_very_small_axis_values(self):
        """Test mapping with very small axis values."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.001,
            knee_out_in=-0.001,
            hip_yaw=0.0001,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        assert joint_cmd.joint_positions[fl_hip_pitch] == pytest.approx(0.001 * DEFAULT_HIP_UP_DOWN_SCALE)
        assert joint_cmd.joint_positions[fl_knee] == pytest.approx(-0.001 * DEFAULT_KNEE_OUT_IN_SCALE)
        assert joint_cmd.joint_positions[fl_hip_yaw] == pytest.approx(0.0001 * DEFAULT_HIP_YAW_SCALE)

    def test_max_normalized_values(self):
        """Test mapping with maximum normalized values (±1.0)."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=1.0,
            knee_out_in=-1.0,
            hip_yaw=1.0,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        fl_hip_yaw, fl_hip_pitch, fl_knee = LEG_TO_JOINT_INDICES[LegIdentifier.FRONT_LEFT]
        assert joint_cmd.joint_positions[fl_hip_pitch] == pytest.approx(DEFAULT_HIP_UP_DOWN_SCALE)
        assert joint_cmd.joint_positions[fl_knee] == pytest.approx(-DEFAULT_KNEE_OUT_IN_SCALE)
        assert joint_cmd.joint_positions[fl_hip_yaw] == pytest.approx(DEFAULT_HIP_YAW_SCALE)

    def test_joint_command_zero_copy_guarantee(self):
        """Test that joint positions array is a new array (not a view)."""
        state = ControllerState()
        control_data = GamepadControlData(
            selected_legs={LegIdentifier.FRONT_LEFT},
            hip_up_down=0.5,
            knee_out_in=0.0,
            hip_yaw=0.0,
            raw_state=state,
        )
        
        joint_cmd = self.mapper.map(control_data)
        
        # Modify the returned array
        joint_cmd.joint_positions[0] = 999.0
        
        # Internal state should not be affected (zero-copy guarantee)
        assert self.mapper._last_joint_positions[0] != 999.0
