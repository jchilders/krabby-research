"""InputController singleton for gamepad input using Pygame SDL2 Game Controller API.

Uses pygame's SDL2 Game Controller API (pygame._sdl2.controller) for logical axis/button
mapping, so behavior is consistent across macOS, Linux (e.g. Jetson Orin), and Windows.
SDL's controller mapping database normalizes different physical controllers to the same
logical layout (left stick, right stick, triggers, shoulder buttons, stick clicks).

Testing:
--------
1. Install pygame: pip install pygame
2. Use the monitor command: python -m controller.input --monitor
3. List controller-capable devices: python -m controller.input --list
"""
import logging
import threading
import time
from typing import Callable, Optional

try:
    import pygame
    import pygame._sdl2.controller as sdl2_controller
except ImportError as e:
    if "_sdl2" in str(e) or "controller" in str(e):
        raise ImportError(
            "pygame SDL2 controller module not available. Install pygame 2.6+ with SDL2."
        ) from e
    raise ImportError(
        "pygame library not installed. Install with: pip install pygame"
    ) from e

from controller.input.state import ControllerState

logger = logging.getLogger(__name__)

# SDL2 controller get_axis() returns int: sticks -32768..32767, triggers 0..32768
_AXIS_SCALE = 32768.0


class InputController:
    """Singleton controller for reading and storing gamepad input state.

    This class provides a thread-safe interface for reading gamepad events
    and storing them in a normalized ControllerState dataclass.
    Uses pygame SDL2 Game Controller API for cross-platform logical mapping.

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
        self._controller: Optional[sdl2_controller.Controller] = None

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

        # Clean up SDL2 controller
        if self._controller is not None:
            try:
                self._controller.quit()
            except Exception:
                pass
            self._controller = None

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
                RX=self._state.RX,
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

        Uses pygame SDL2 Game Controller API. Initializes pygame and controller
        subsystem, opens the selected controller-capable device, and polls state
        at fixed rate.
        """
        sleep_time = 1.0 / self._update_rate_hz

        pygame_was_initialized = pygame.get_init()
        joystick_was_initialized = pygame.joystick.get_init()

        try:
            if not pygame_was_initialized:
                pygame.init()
            if not joystick_was_initialized:
                pygame.joystick.init()
            if not sdl2_controller.get_init():
                sdl2_controller.init()
        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}", exc_info=True)
            self._running = False
            return

        device_count = sdl2_controller.get_count()
        logger.debug(f"Found {device_count} joystick(s)")

        if device_count == 0:
            logger.error(
                "No controller-capable device found. Make sure your controller is connected."
            )
            logger.error("Try running with --list to verify the controller is detected.")
            self._running = False
            if not pygame_was_initialized:
                pygame.quit()
            return

        # Resolve device_id: use first controller-capable device if selected one is not
        if self._device_id >= device_count:
            logger.warning(
                f"Device ID {self._device_id} not available. Using device 0 instead."
            )
            self._device_id = 0

        if not sdl2_controller.is_controller(self._device_id):
            # Try device 0 if it is a controller
            if sdl2_controller.is_controller(0):
                logger.warning(
                    f"Device {self._device_id} is not a supported game controller. "
                    "Using device 0 instead."
                )
                self._device_id = 0
            else:
                logger.error(
                    f"Device {self._device_id} is not a supported game controller. "
                    "Use --list to see controller-capable devices."
                )
                self._running = False
                if not pygame_was_initialized:
                    pygame.quit()
                return

        try:
            self._controller = sdl2_controller.Controller(self._device_id)
            name = sdl2_controller.name_forindex(self._device_id) or "Unknown"
            logger.info(f"Using SDL2 controller: {name}")
        except Exception as e:
            logger.error(
                f"Failed to initialize controller {self._device_id}: {e}",
                exc_info=True,
            )
            self._running = False
            if not pygame_was_initialized:
                pygame.quit()
            return

        try:
            while self._running:
                start_time = time.time()

                self._update_state_controller(self._controller)

                state = self.get_state()
                self._notify_callbacks(state)

                elapsed = time.time() - start_time
                sleep_duration = max(0, sleep_time - elapsed)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

        except Exception as e:
            logger.error(f"InputController event loop error: {e}", exc_info=True)
            raise
        finally:
            if self._controller is not None:
                try:
                    self._controller.quit()
                except Exception:
                    pass
                self._controller = None
            if not pygame_was_initialized:
                pygame.quit()
            self._running = False

    def _update_state_controller(
        self, controller: sdl2_controller.Controller
    ) -> None:
        """Update controller state from SDL2 Game Controller.

        Uses logical axis/button constants; values are normalized to [-1, 1]
        for sticks and thresholded for triggers.
        """
        with self._state_lock:
            self._state.LT = False
            self._state.LB = False
            self._state.LS = False
            self._state.RT = False
            self._state.RB = False
            self._state.RS = False

            # Buttons (logical names)
            self._state.LB = bool(controller.get_button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER))
            self._state.RB = bool(controller.get_button(pygame.CONTROLLER_BUTTON_RIGHTSHOULDER))
            self._state.LS = bool(controller.get_button(pygame.CONTROLLER_BUTTON_LEFTSTICK))
            self._state.RS = bool(controller.get_button(pygame.CONTROLLER_BUTTON_RIGHTSTICK))

            # Triggers (axes 0..32768, normalize then threshold)
            trigger_left = controller.get_axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT)
            trigger_right = controller.get_axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)
            self._state.LT = (trigger_left / _AXIS_SCALE) > 0.1
            self._state.RT = (trigger_right / _AXIS_SCALE) > 0.1

            # Sticks (axes -32768..32767, normalize to [-1, 1])
            self._state.LX = max(-1.0, min(1.0, controller.get_axis(pygame.CONTROLLER_AXIS_LEFTX) / _AXIS_SCALE))
            self._state.LY = max(-1.0, min(1.0, controller.get_axis(pygame.CONTROLLER_AXIS_LEFTY) / _AXIS_SCALE))
            self._state.RX = max(-1.0, min(1.0, controller.get_axis(pygame.CONTROLLER_AXIS_RIGHTX) / _AXIS_SCALE))
            self._state.RY = max(-1.0, min(1.0, controller.get_axis(pygame.CONTROLLER_AXIS_RIGHTY) / _AXIS_SCALE))

            logger.debug(
                f"Left stick X: {self._state.LX}, Left stick Y: {self._state.LY}, "
                f"Right stick X: {self._state.RX}, Right stick Y: {self._state.RY}"
            )
            logger.debug(
                f"Left trigger: {self._state.LT}, Right trigger: {self._state.RT}"
            )
            logger.debug(
                f"Left bumper: {self._state.LB}, Right bumper: {self._state.RB}"
            )
            logger.debug(
                f"Left stick press: {self._state.LS}, Right stick press: {self._state.RS}"
            )

    def _notify_callbacks(self, state: ControllerState) -> None:
        """Notify all registered callbacks with new controller state."""
        with self._callback_lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Error in InputController callback: {e}", exc_info=True)

    @staticmethod
    def list_devices() -> list[dict]:
        """List controller-capable gamepad devices (SDL2 Game Controller API).

        Returns:
            List of device dicts with 'name' and 'path' for devices that
            is_controller(i) is True.
        """
        try:
            pygame.init()
            pygame.joystick.init()
            if not sdl2_controller.get_init():
                sdl2_controller.init()
            gamepads = []
            for i in range(sdl2_controller.get_count()):
                if sdl2_controller.is_controller(i):
                    name = sdl2_controller.name_forindex(i)
                    gamepads.append({
                        "name": name if name is not None else f"Controller {i}",
                        "path": f"pygame_controller_{i}",
                        "device_id": i,  # joystick index for Controller(index)
                    })
            pygame.quit()
            return gamepads
        except Exception as e:
            logger.error(f"Error listing devices: {e}", exc_info=True)
            return []
