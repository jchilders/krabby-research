"""InputController singleton for gamepad/joystick input handling."""
import logging
import threading
import time
from typing import Callable, Optional

try:
    from inputs import get_gamepad, devices
except ImportError:
    raise ImportError(
        "inputs library not installed. Install with: pip install inputs"
    )

from controller.input.state import ControllerState, GamepadControlData, LegIdentifier

logger = logging.getLogger(__name__)


class InputController:
    """Singleton controller for reading and processing gamepad input.
    
    This class provides a thread-safe interface for reading gamepad events
    and processing them into normalized controller state and control data.
    
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
        self._callbacks: list[Callable[[GamepadControlData], None]] = []
        self._callback_lock = threading.Lock()
        
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
        
        self._update_rate_hz = update_rate_hz
        self._device_id = device_id
        
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
            f"InputController started (device_id={device_id}, "
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
    
    def get_control_data(self) -> GamepadControlData:
        """Get processed control data with leg selection and axis mappings.
        
        Returns:
            GamepadControlData with selected legs and axis values.
        """
        state = self.get_state()
        return self._process_controls(state)
    
    def register_callback(
        self, callback: Callable[[GamepadControlData], None]
    ) -> None:
        """Register a callback to be called when control data is updated.
        
        Args:
            callback: Function that takes GamepadControlData as argument.
        """
        with self._callback_lock:
            self._callbacks.append(callback)
    
    def unregister_callback(
        self, callback: Callable[[GamepadControlData], None]
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
        
        This loop handles two concerns:
        1. Reading gamepad events (blocking, but in separate thread)
        2. Processing controls at fixed rate (50-100 Hz)
        
        We use a separate thread for event reading to avoid blocking
        the control processing loop.
        """
        sleep_time = 1.0 / self._update_rate_hz if self._update_rate_hz > 0 else 0.02
        
        # Start event reading in a sub-thread to avoid blocking control processing
        event_thread_running = threading.Event()
        event_thread_running.set()
        
        def read_events():
            """Read gamepad events in a separate thread."""
            try:
                while self._running and event_thread_running.is_set():
                    try:
                        # get_gamepad() blocks until events are available
                        # This is fine since we're in a separate thread
                        events = get_gamepad()
                        for event in events:
                            if not self._running:
                                break
                            self._update_state(event)
                    except Exception as e:
                        if self._running:
                            logger.debug(f"Error reading gamepad events: {e}")
                        time.sleep(0.01)  # Brief sleep on error
            except Exception as e:
                logger.error(f"Event reading thread error: {e}", exc_info=True)
        
        event_thread = threading.Thread(
            target=read_events,
            daemon=True,
            name="InputController-EventReader"
        )
        event_thread.start()
        
        try:
            # Main control processing loop at fixed rate
            while self._running:
                start_time = time.time()
                
                # Process controls and notify callbacks
                control_data = self.get_control_data()
                self._notify_callbacks(control_data)
                
                # Sleep to maintain target rate
                elapsed = time.time() - start_time
                sleep_duration = max(0, sleep_time - elapsed)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                    
        except Exception as e:
            logger.error(f"InputController event loop error: {e}", exc_info=True)
        finally:
            event_thread_running.clear()
            # Wait briefly for event thread to finish
            event_thread.join(timeout=1.0)
            self._running = False
    
    def _update_state(self, event) -> None:
        """Update controller state from a gamepad event.
        
        Args:
            event: Event from inputs library.
        """
        code = event.code
        val = event.state
        
        with self._state_lock:
            # Map event codes to state fields
            # Triggers (analog, thresholded at 10)
            if code == "ABS_Z":  # Left Trigger analog
                self._state.LT = (val > 10)
            elif code == "ABS_RZ":  # Right Trigger analog
                self._state.RT = (val > 10)
            # Bumpers
            elif code == "BTN_TL":  # Left Bumper
                self._state.LB = (val == 1)
            elif code == "BTN_TR":  # Right Bumper
                self._state.RB = (val == 1)
            # Stick buttons
            elif code == "BTN_THUMBL":  # Left Stick button
                self._state.LS = (val == 1)
            elif code == "BTN_THUMBR":  # Right Stick button
                self._state.RS = (val == 1)
            # Sticks (normalize to [-1.0, 1.0])
            elif code == "ABS_X":  # Left stick X
                # Normalize from typical range [-32768, 32767] to [-1.0, 1.0]
                self._state.LX = max(-1.0, min(1.0, val / 32768.0))
            elif code == "ABS_Y":  # Left stick Y
                self._state.LY = max(-1.0, min(1.0, val / 32768.0))
            elif code == "ABS_RX":  # Right stick X
                self._state.RX = max(-1.0, min(1.0, val / 32768.0))
            elif code == "ABS_RY":  # Right stick Y
                self._state.RY = max(-1.0, min(1.0, val / 32768.0))
    
    def _process_controls(self, state: ControllerState) -> GamepadControlData:
        """Process controller state into leg selection and axis mappings.
        
        Implements the leg selection logic:
        - LT (without LB): Select Front Left (FL)
        - LB (without LT): Select Rear Left (RL)
        - LS: Select Left Middle (ML)
        - RS: Select Right Middle (MR)
        - RT (without RB): Select Front Right (FR)
        - RB (without RT): Select Rear Right (RR)
        - LT + LB: Select FL, RL, MR (tripod combo left)
        - RT + RB: Select FR, RR, ML (tripod combo right)
        
        Axis mappings:
        - Left stick Y: Hip up/down (inverted)
        - Left stick X: Knee out/in
        - Right stick Y: Hip yaw forward/back
        
        Args:
            state: Current controller state.
            
        Returns:
            GamepadControlData with selected legs and axis values.
        """
        # Determine leg selections
        select_FL = state.LT and not state.LB
        select_RL = state.LB and not state.LT
        select_ML = state.LS
        select_MR = state.RS
        select_FR = state.RT and not state.RB
        select_RR = state.RB and not state.RT
        
        # Combo triggers
        combo_left = state.LT and state.LB  # FL/RL/MR
        combo_right = state.RT and state.RB  # FR/RR/ML
        
        # Build selected legs set
        legs = set()
        if combo_left:
            legs |= {LegIdentifier.FRONT_LEFT, LegIdentifier.REAR_LEFT, LegIdentifier.MIDDLE_RIGHT}
        if combo_right:
            legs |= {LegIdentifier.FRONT_RIGHT, LegIdentifier.REAR_RIGHT, LegIdentifier.MIDDLE_LEFT}
        if not combo_left and not combo_right:
            if select_FL:
                legs.add(LegIdentifier.FRONT_LEFT)
            if select_RL:
                legs.add(LegIdentifier.REAR_LEFT)
            if select_ML:
                legs.add(LegIdentifier.MIDDLE_LEFT)
            if select_MR:
                legs.add(LegIdentifier.MIDDLE_RIGHT)
            if select_FR:
                legs.add(LegIdentifier.FRONT_RIGHT)
            if select_RR:
                legs.add(LegIdentifier.REAR_RIGHT)
        
        # Map axes
        hip_up_down = -state.LY  # Invert Y axis (up = positive)
        knee_out_in = state.LX
        hip_yaw = state.RY
        
        return GamepadControlData(
            selected_legs=legs,
            hip_up_down=hip_up_down,
            knee_out_in=knee_out_in,
            hip_yaw=hip_yaw,
            raw_state=state,
        )
    
    def _notify_callbacks(self, control_data: GamepadControlData) -> None:
        """Notify all registered callbacks with new control data.
        
        Args:
            control_data: Current control data to send to callbacks.
        """
        with self._callback_lock:
            callbacks = list(self._callbacks)  # Copy to avoid lock during iteration
        
        for callback in callbacks:
            try:
                callback(control_data)
            except Exception as e:
                logger.error(f"Error in InputController callback: {e}", exc_info=True)
    
    @staticmethod
    def list_devices() -> list[dict]:
        """List available gamepad/input devices.
        
        Returns:
            List of device dictionaries with 'name' and 'path' keys.
        """
        try:
            gamepads = []
            for device in devices:
                if device.device_type == "gamepad":
                    gamepads.append({
                        "name": device.name,
                        "path": device.path,
                    })
            return gamepads
        except Exception as e:
            logger.error(f"Error listing devices: {e}", exc_info=True)
            return []
