"""Unit tests for InputController - this will be updated later once leg mapping logic moves to the mappers and is not part of InputController."""
import time
from unittest.mock import Mock, patch

from controller.input.input_controller import InputController
from controller.input.state import ControllerState, GamepadControlData, LegIdentifier


class TestInputControllerSingleton:
    """Test singleton pattern."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        if InputController._instance is not None:
            try:
                InputController._instance.stop()
            except:
                pass
        InputController._instance = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if InputController._instance is not None:
            try:
                InputController._instance.stop()
            except:
                pass
        InputController._instance = None
    
    def test_singleton_returns_same_instance(self):
        """Test that get_instance returns the same instance."""
        controller1 = InputController.get_instance()
        controller2 = InputController.get_instance()
        assert controller1 is controller2


class TestInputControllerState:
    """Test state management."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        InputController._instance = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if InputController._instance is not None:
            InputController._instance.stop()
        InputController._instance = None
    
    def test_default_state(self):
        """Test default controller state."""
        controller = InputController.get_instance()
        state = controller.get_state()
        assert state.LT is False
        assert state.LB is False
        assert state.LX == 0.0
        assert state.LY == 0.0
        assert state.RX == 0.0
        assert state.RY == 0.0
    
    def test_state_update_pygame(self):
        """Test state update from pygame joystick."""
        controller = InputController.get_instance()
        
        # Mock pygame joystick
        mock_joystick = Mock()
        mock_joystick.get_numbuttons.return_value = 11
        mock_joystick.get_numaxes.return_value = 6
        mock_joystick.get_button.side_effect = lambda i: {
            7: True,   # LS
            9: True,   # LB
        }.get(i, False)
        mock_joystick.get_axis.side_effect = lambda i: {
            0: 0.5,    # LX
            1: -0.3,   # LY
            4: 0.2,    # LT (above threshold)
        }.get(i, 0.0)
        
        controller._update_state_pygame(mock_joystick)
        state = controller.get_state()
        
        assert state.LS is True
        assert state.LB is True
        assert state.LT is True
        assert abs(state.LX - 0.5) < 0.01
        assert abs(state.LY - (-0.3)) < 0.01


class TestInputControllerLegSelection:
    """Test leg selection logic."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        InputController._instance = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if InputController._instance is not None:
            InputController._instance.stop()
        InputController._instance = None
    
    def test_single_leg_selection(self):
        """Test single leg selection."""
        controller = InputController.get_instance()
        
        # Front Left (LT without LB)
        state = ControllerState(LT=True, LB=False)
        control_data = controller._process_controls(state)
        assert LegIdentifier.FRONT_LEFT in control_data.selected_legs
        assert len(control_data.selected_legs) == 1
        
        # Front Right (RT without RB)
        state = ControllerState(RT=True, RB=False)
        control_data = controller._process_controls(state)
        assert LegIdentifier.FRONT_RIGHT in control_data.selected_legs
        assert len(control_data.selected_legs) == 1
    
    def test_combo_leg_selection(self):
        """Test combo leg selection (tripod)."""
        controller = InputController.get_instance()
        
        # Left tripod (LT + LB)
        state = ControllerState(LT=True, LB=True)
        control_data = controller._process_controls(state)
        assert LegIdentifier.FRONT_LEFT in control_data.selected_legs
        assert LegIdentifier.REAR_LEFT in control_data.selected_legs
        assert LegIdentifier.MIDDLE_RIGHT in control_data.selected_legs
        assert len(control_data.selected_legs) == 3
        
        # Right tripod (RT + RB)
        state = ControllerState(RT=True, RB=True)
        control_data = controller._process_controls(state)
        assert LegIdentifier.FRONT_RIGHT in control_data.selected_legs
        assert LegIdentifier.REAR_RIGHT in control_data.selected_legs
        assert LegIdentifier.MIDDLE_LEFT in control_data.selected_legs
        assert len(control_data.selected_legs) == 3


class TestInputControllerAxisMapping:
    """Test axis mapping."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        InputController._instance = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if InputController._instance is not None:
            InputController._instance.stop()
        InputController._instance = None
    
    def test_axis_mapping(self):
        """Test axis mapping to control data."""
        controller = InputController.get_instance()
        
        state = ControllerState(LX=0.5, LY=-0.3, RY=0.7)
        control_data = controller._process_controls(state)
        
        # LY is inverted for hip_up_down
        assert control_data.hip_up_down == 0.3
        # LX maps directly to knee_out_in
        assert control_data.knee_out_in == 0.5
        # RY maps directly to hip_yaw
        assert control_data.hip_yaw == 0.7


class TestInputControllerLifecycle:
    """Test start/stop lifecycle."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        InputController._instance = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if InputController._instance is not None:
            InputController._instance.stop()
        InputController._instance = None
    
    @patch('controller.input.input_controller.pygame')
    def test_start_stop(self, mock_pygame):
        """Test starting and stopping the controller."""
        # Mock pygame initialization
        mock_pygame.get_init.return_value = False
        mock_pygame.joystick.get_init.return_value = False
        mock_pygame.joystick.get_count.return_value = 1
        
        mock_joystick = Mock()
        mock_joystick.get_name.return_value = "Test Controller"
        mock_pygame.joystick.Joystick.return_value = mock_joystick
        
        controller = InputController.get_instance()
        
        # Start controller
        controller.start(update_rate_hz=50.0)
        assert controller._running is True
        assert controller._thread is not None
        
        # Wait a bit
        time.sleep(0.1)
        
        # Stop controller
        controller.stop()
        assert controller._running is False
