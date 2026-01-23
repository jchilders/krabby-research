"""InputController singleton for gamepad/joystick input handling using Pygame.

Testing with Pro Controller:
-------------------------------------
1. Ensure your Pro Controller is paired via Bluetooth:
   - Press and hold the sync button on the Pro Controller until lights flash
   - On Mac: System Preferences > Bluetooth > Connect to "Pro Controller"

2. Install pygame if not already installed:
   pip install pygame

3. Use the monitor command:
   python -m controller.input --monitor

Pro Controller Button/Axis Mapping:
------------------------------------------------------------------------------
- B7 = Left Stick click (LS)
- B8 = Right Stick click (RS)
- B9 = Left Bumper (LB)
- B10 = Right Bumper (RB)
- A0 = Left Stick X (LX)
- A1 = Left Stick Y (LY)
- A2 = Right Stick X (RX)
- A3 = Right Stick Y (RY)
- A4 = ZL (Left Trigger) - analog axis
- A5 = ZR (Right Trigger) - analog axis
"""
import logging
import threading
import time
from typing import Callable, Optional

try:
    import pygame
except ImportError:
    raise ImportError(
        "pygame library not installed. Install with: pip install pygame"
    )

from controller.input.state import ControllerState

logger = logging.getLogger(__name__)


class InputController:
    """Singleton controller for reading and storing gamepad input state.
    
    This class provides a thread-safe interface for reading gamepad events
    and storing them in a normalized ControllerState dataclass.
    Uses pygame.
    
    Usage:
        controller = InputController.get_instance()
        controller.start(device_id=0, update_rate_hz=50)
        # ... use controller.get_state() or register callbacks
        controller.stop()
    """
    
    _instance: Optional["InputController"] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize InputController (private, use get_instance())."""
        # Check if an instance already exists and it's not this instance
        if InputController._instance is not None and InputController._instance is not self:
            raise RuntimeError(
                "InputController is a singleton. Use get_instance() instead."
            )
        
        self._state = ControllerState()
        self._state_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._device_id: Optional[int] = None
        self._update_rate_hz = 50.0   # once every 20ms
        self._callbacks: list[Callable[[ControllerState], None]] = []
        self._callback_lock = threading.Lock()
        self._joystick: Optional[pygame.joystick.Joystick] = None
        
    @classmethod
    def get_instance(cls) -> "InputController":
        """Get the singleton instance of InputController."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__()
        return cls._instance
    
    def start(
        self,
        device_id: Optional[int] = None,
        update_rate_hz: float = 50.0,
    ) -> None:
        """Start the input controller event loop.
        
        Args:
            device_id: Optional device ID to use. If None, uses first available gamepad.
            update_rate_hz: Target update rate for processing controls (default: 50.0).
        """
        if self._running:
            logger.warning("InputController is already running")
            return
        
        if update_rate_hz <= 0:
            raise ValueError(
                f"update_rate_hz must be greater than 0, got {update_rate_hz}"
            )
        
        self._update_rate_hz = update_rate_hz
        self._device_id = device_id if device_id is not None else 0
        
        # Reset state
        with self._state_lock:
            self._state = ControllerState()
        
        self._running = True
        self._thread = threading.Thread(
            target=self._event_loop,
            daemon=True,
            name="InputController"
        )
        self._thread.start()
        logger.info(
            f"InputController started (device_id={self._device_id}, "
            f"update_rate={update_rate_hz} Hz)"
        )
    
    def stop(self) -> None:
        """Stop the input controller event loop."""
        if not self._running:
            return
        
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("InputController thread did not stop cleanly")
        
        # Clean up pygame joystick
        if self._joystick is not None:
            try:
                self._joystick.quit()
            except Exception:
                pass
            self._joystick = None
        
        logger.info("InputController stopped")
    
    def get_state(self) -> ControllerState:
        """Get the current normalized controller state (thread-safe).
        
        Returns:
            A copy of the current ControllerState.
        """
        with self._state_lock:
            return ControllerState(
                LT=self._state.LT,
                LB=self._state.LB,
                LS=self._state.LS,
                RS=self._state.RS,
                RT=self._state.RT,
                RB=self._state.RB,
                LX=self._state.LX,
                LY=self._state.LY,
                RX=self._state.RX, # Right stick X - not mapped to anything currently
                RY=self._state.RY, 
            )
    
    def register_callback(
        self, callback: Callable[[ControllerState], None]
    ) -> None:
        """Register a callback to be called when controller state is updated.
        
        Args:
            callback: Function that takes ControllerState as argument.
        """
        with self._callback_lock:
            self._callbacks.append(callback)
    
    def unregister_callback(
        self, callback: Callable[[ControllerState], None]
    ) -> None:
        """Unregister a callback.
        
        Args:
            callback: Callback function to remove.
        """
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _event_loop(self) -> None:
        """Main event loop running in background thread.
        
        This loop handles:
        1. Reading gamepad events via pygame
        2. Processing controls at fixed rate (50-100 Hz)
        
        This method checks if pygame is already initialized and only initializes
        joystick subsystem if needed.
        """
        sleep_time = 1.0 / self._update_rate_hz # e.g. if update_rate_hz is 50, sleep_time is 0.02 seconds
        
        # Check if pygame is already initialized 
        pygame_was_initialized = pygame.get_init()
        joystick_was_initialized = pygame.joystick.get_init()
        
        # Initialize pygame only if not already initialized
        try:
            if not pygame_was_initialized:
                pygame.init()
            if not joystick_was_initialized:
                pygame.joystick.init()
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}", exc_info=True)
            self._running = False
            return
        
        # Initialize joystick
        joystick_count = pygame.joystick.get_count()
        logger.debug(f"Found {joystick_count} joystick(s)")
        
        if joystick_count == 0:
            logger.error("No joystick/gamepad found. Make sure your controller is connected.")
            logger.error("Try running with --list to verify the controller is detected.")
            self._running = False
            if not pygame_was_initialized:
                pygame.quit()
            return
        
        if self._device_id >= pygame.joystick.get_count():
            logger.warning(
                f"Device ID {self._device_id} not available. "
                f"Using device 0 instead."
            )
            self._device_id = 0
        
        try:
            self._joystick = pygame.joystick.Joystick(self._device_id)
            self._joystick.init()
            logger.info(f"Using pygame joystick: {self._joystick.get_name()}")
        except Exception as e:
            logger.error(f"Failed to initialize joystick {self._device_id}: {e}", exc_info=True)
            self._running = False
            if not pygame_was_initialized:
                pygame.quit()
            return
        
        try:
            # Main control processing loop at fixed rate
            while self._running:
                start_time = time.time()
            
                
                # Update state from pygame joystick
                self._update_state_pygame(self._joystick)
                
                # Get current state and notify callbacks
                state = self.get_state()
                self._notify_callbacks(state)
                
                # Sleep to maintain target rate
                elapsed = time.time() - start_time
                sleep_duration = max(0, sleep_time - elapsed)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                    
        except Exception as e:
            logger.error(f"InputController event loop error: {e}", exc_info=True)
            raise e
        finally:
            # Clean up joystick
            if self._joystick is not None:
                try:
                    self._joystick.quit()
                except Exception:
                    pass
            if not pygame_was_initialized:
                pygame.quit()
            self._running = False
    
    def _update_state_pygame(self, joystick: pygame.joystick.Joystick) -> None:
        """Update controller state from pygame joystick.
        
        Args:
            joystick: pygame.joystick.Joystick instance.
        
        Note: Pro Controller button/axis indices may vary. If buttons don't work
        correctly, you may need to adjust these indices. See the file header
        documentation for how to find the correct mapping.
        """
        with self._state_lock:
            # Reset all button states first
            self._state.LT = False
            self._state.LB = False
            self._state.LS = False
            self._state.RT = False
            self._state.RB = False
            self._state.RS = False
            
            # Pro Controller button mapping (Pro Controller on any platform via pygame)
            # B7 = Left Stick click (LS)
            # B8 = Right Stick click (RS)
            # B9 = Left Bumper (LB)
            # B10 = Right Bumper (RB)
            # A4 = ZL (Left Trigger) - analog axis
            # A5 = ZR (Right Trigger) - analog axis
            
            # Stick buttons (clicking the sticks)
            if joystick.get_numbuttons() > 7:
                self._state.LS = bool(joystick.get_button(7))  # Left stick press
            if joystick.get_numbuttons() > 8:
                self._state.RS = bool(joystick.get_button(8))  # Right stick press
            
            # Bumpers (L and R buttons)
            if joystick.get_numbuttons() > 9:
                self._state.LB = bool(joystick.get_button(9))  # L button (left bumper)
            if joystick.get_numbuttons() > 10:
                self._state.RB = bool(joystick.get_button(10))  # R button (right bumper)
            
            # Triggers (ZL and ZR) - analog axes, thresholded as boolean
            if joystick.get_numaxes() > 4:
                trigger_left_val = joystick.get_axis(4)
                self._state.LT = (trigger_left_val > 0.1)  # ZL (Left Trigger)
            if joystick.get_numaxes() > 5:
                trigger_right_val = joystick.get_axis(5)
                self._state.RT = (trigger_right_val > 0.1)  # ZR (Right Trigger)
            
            # Sticks (Clipped to -1.0 to 1.0 range. Pygame already returns values in this range. Clipping is just to ensure the values are in the correct range.)
            if joystick.get_numaxes() > 0:
                self._state.LX = max(-1.0, min(1.0, joystick.get_axis(0)))  # Left stick X
            if joystick.get_numaxes() > 1:
                self._state.LY = max(-1.0, min(1.0, joystick.get_axis(1)))  # Left stick Y
            if joystick.get_numaxes() > 2:
                self._state.RX = max(-1.0, min(1.0, joystick.get_axis(2)))  # Right stick X
            if joystick.get_numaxes() > 3:
                self._state.RY = max(-1.0, min(1.0, joystick.get_axis(3)))  # Right stick Y
            logger.debug(f"Left stick X: {self._state.LX}, Left stick Y: {self._state.LY}, Right stick X: {self._state.RX}, Right stick Y: {self._state.RY}")
            logger.debug(f"Left trigger: {self._state.LT}, Right trigger: {self._state.RT}")
            logger.debug(f"Left bumper: {self._state.LB}, Right bumper: {self._state.RB}")
            logger.debug(f"Left stick press: {self._state.LS}, Right stick press: {self._state.RS}")
    
    def _notify_callbacks(self, state: ControllerState) -> None:
        """Notify all registered callbacks with new controller state.
        
        Args:
            state: Current controller state to send to callbacks.
        """
        with self._callback_lock:
            callbacks = list(self._callbacks)  # Copy to avoid lock during iteration
        
        for callback in callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Error in InputController callback: {e}", exc_info=True)
    
    @staticmethod
    def list_devices() -> list[dict]:
        """List available gamepad/input devices using pygame.
        
        Returns:
            List of device dictionaries with 'name' and 'path' keys.
        """
        try:
            pygame.init()
            pygame.joystick.init()
            gamepads = []
            for i in range(pygame.joystick.get_count()):
                joystick = pygame.joystick.Joystick(i)
                joystick.init()
                gamepads.append({
                    "name": joystick.get_name(),
                    "path": f"pygame_joystick_{i}",
                })
                joystick.quit()
            pygame.quit()
            return gamepads
        except Exception as e:
            logger.error(f"Error listing devices: {e}", exc_info=True)
            return []

