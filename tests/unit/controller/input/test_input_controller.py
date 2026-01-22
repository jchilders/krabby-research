"""Unit tests for InputController."""

from controller.input.input_controller import InputController
from controller.input.state import ControllerState


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
        assert state.LS is False
        assert state.RT is False
        assert state.RB is False
        assert state.RS is False
        assert state.LX == 0.0
        assert state.LY == 0.0
        assert state.RX == 0.0
        assert state.RY == 0.0




class TestInputControllerThreadSafety:
    """Test thread safety of state access."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        InputController._instance = None
    
    def teardown_method(self):
        """Clean up after each test."""
        if InputController._instance is not None:
            InputController._instance.stop()
        InputController._instance = None
    
    def test_get_state_returns_copy(self):
        """Test that get_state returns a copy, not the internal state."""
        controller = InputController.get_instance()
        
        state1 = controller.get_state()
        state2 = controller.get_state()
        
        # Should be different objects
        assert state1 is not state2
        
        # But should have same values
        assert state1.LT == state2.LT
        assert state1.LY == state2.LY
