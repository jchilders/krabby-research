"""Control loop for wiring controller components together.

This module provides a ControlLoop class that wires singleton components
(InputController, HALClient, mappers, SDKs) based on configuration.

"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from controller.input import InputController
from controller.input.state import ControllerState
from controller.mappers.gamepad_to_isaacsim_hal_mapper import GamepadToIsaacSimHALMapper
from controller.mappers.gamepad_to_krabby_hal_mapper import GamepadToKrabbyHALMapper
from hal.client import HalClient
from hal.client.config import HalClientConfig

logger = logging.getLogger(__name__)


class ControlMode(str, Enum):
    """Control loop mode enumeration. 
    More modes may be added later."""
    INPUT_CONTROLLER_ISAACSIM = "input_controller_isaacsim"
    MODEL_CONTROLLER_KRABBY = "model_controller_krabby"    
    INPUT_CONTROLLER_KRABBY = "input_controller_krabby"


@dataclass
class ControlLoopConfig:
    """Configuration for ControlLoop.
    
    Attributes:
        mode: Control mode (INPUT_CONTROLLER_ISAACSIM or MODEL_CONTROLLER_KRABBY or INPUT_CONTROLLER_KRABBY)
        input_controller_device_id: Optional device ID for InputController
        input_controller_update_rate_hz: Update rate for InputController (default: 50.0)
        hal_client_config: HAL client configuration
        mapper_hip_up_down_scale: Scaling factor for hip up/down axis
        mapper_knee_out_in_scale: Scaling factor for knee out/in axis
        mapper_hip_yaw_scale: Scaling factor for hip yaw axis
    """
    mode: ControlMode
    input_controller_device_id: Optional[int] = None
    input_controller_update_rate_hz: float = 50.0
    hal_client_config: Optional[HalClientConfig] = None
    mapper_hip_up_down_scale: float = 0.3
    mapper_knee_out_in_scale: float = 0.3
    mapper_hip_yaw_scale: float = 0.2


class ControlLoop:
    """Control loop that wires controller components together.
    
    Manages lifecycle of InputController, HALClient, mappers, and SDKs
    based on configuration. Provides clean thread lifecycle management.
    
    Usage:
        config = ControlLoopConfig(
            mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
            hal_client_config=HalClientConfig(...),
        )
        loop = ControlLoop(config)
        loop.start()
        # ... run control loop ...
        loop.stop()
    """
    
    def __init__(self, config: ControlLoopConfig):
        """Initialize control loop.
        
        Args:
            config: Control loop configuration
        """
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Component references
        self._input_controller: Optional[InputController] = None
        self._hal_client: Optional[HalClient] = None
        self._gamepad_to_isaacsim_hal_mapper: Optional[GamepadToIsaacSimHALMapper] = None
        self._gamepad_to_krabby_hal_mapper: Optional[GamepadToKrabbyHALMapper] = None
    
    def start(self) -> None:
        """Start the control loop.
        
        Initializes and starts all components based on configuration.
        
        Raises:
            RuntimeError: If control loop is already running or configuration is invalid
        """
        if self._running:
            logger.warning("ControlLoop is already running")
            return
        
        logger.info(f"Starting ControlLoop in mode: {self.config.mode}")
        
        # Initialize components based on mode
        if self.config.mode == ControlMode.INPUT_CONTROLLER_ISAACSIM:
            self._start_input_controller_isaacsim_mode()
        elif self.config.mode == ControlMode.MODEL_CONTROLLER_KRABBY:
            # TODO: Implement MODEL_CONTROLLER_KRABBY mode. 
            raise NotImplementedError("MODEL_CONTROLLER_KRABBY mode not yet implemented")
        elif self.config.mode == ControlMode.INPUT_CONTROLLER_KRABBY:
            self._start_input_controller_krabby_mode()
        else:
            raise ValueError(f"Unknown control mode: {self.config.mode}")
        
        self._running = True
        logger.info("ControlLoop started successfully")
    
    def stop(self) -> None:
        """Stop the control loop.
        
        Stops all components and cleans up resources.
        """
        if not self._running:
            return
        
        logger.info("Stopping ControlLoop")
        self._running = False
        
        # Stop input controller
        if self._input_controller is not None:
            self._input_controller.stop()
            self._input_controller = None
        
        # Close HAL client
        if self._hal_client is not None:
            self._hal_client.close()
            self._hal_client = None
        
        logger.info("ControlLoop stopped")
    
    def _start_input_controller_isaacsim_mode(self) -> None:
        """Start components for INPUT_CONTROLLER_ISAACSIM mode."""
        # Get InputController singleton
        self._input_controller = InputController.get_instance()
        
        # Initialize HAL client
        if self.config.hal_client_config is None:
            raise ValueError("hal_client_config is required for INPUT_CONTROLLER_ISAACSIM mode")
        
        self._hal_client = HalClient(
            config=self.config.hal_client_config,
            context=None,
        )
        self._hal_client.initialize()
        
        # Initialize mapper
        self._gamepad_to_isaacsim_hal_mapper = GamepadToIsaacSimHALMapper(
            hip_up_down_scale=self.config.mapper_hip_up_down_scale,
            knee_out_in_scale=self.config.mapper_knee_out_in_scale,
            hip_yaw_scale=self.config.mapper_hip_yaw_scale,
        )
        
        # Register callback to send commands when gamepad state changes
        self._input_controller.register_callback(self._on_gamepad_state)
        
        # Start input controller
        self._input_controller.start(
            device_id=self.config.input_controller_device_id,
            update_rate_hz=self.config.input_controller_update_rate_hz,
        )
        
        logger.info("INPUT_CONTROLLER_ISAACSIM mode initialized")
    
    def _start_input_controller_krabby_mode(self) -> None:
        """Start components for INPUT_CONTROLLER_KRABBY mode."""
        # Get InputController singleton
        self._input_controller = InputController.get_instance()
        
        # Initialize HAL client
        if self.config.hal_client_config is None:
            raise ValueError("hal_client_config is required for INPUT_CONTROLLER_KRABBY mode")
        
        self._hal_client = HalClient(
            config=self.config.hal_client_config,
            context=None,
        )
        self._hal_client.initialize()
        
        # Initialize mapper
        self._gamepad_to_krabby_hal_mapper = GamepadToKrabbyHALMapper(
            hip_up_down_scale=self.config.mapper_hip_up_down_scale,
            knee_out_in_scale=self.config.mapper_knee_out_in_scale,
            hip_yaw_scale=self.config.mapper_hip_yaw_scale,
        )
        
        # Register callback to send commands when gamepad state changes
        self._input_controller.register_callback(self._on_gamepad_state_krabby)
        
        # Start input controller
        self._input_controller.start(
            device_id=self.config.input_controller_device_id,
            update_rate_hz=self.config.input_controller_update_rate_hz,
        )
        
        logger.info("INPUT_CONTROLLER_KRABBY mode initialized")
    
    def _on_gamepad_state(self, state: ControllerState) -> None:
        """Callback for gamepad state updates.
        
        Maps controller state to joint command and sends via HAL client.
        
        Args:
            state: Controller state
        """
        if not self._running:
            return
        
        try:
            # Map controller state to joint command
            # TODO: For now, we don't have observation timestamp, so use None
            # In a real implementation, we'd track the last observation timestamp
            joint_cmd = self._gamepad_to_isaacsim_hal_mapper.map(state, observation_timestamp_ns=None)
            
            # Send command via HAL client
            self._hal_client.put_joint_command(joint_cmd)
            
            logger.debug(
                f"Sent joint command: "
                f"joint_range=[{joint_cmd.joint_positions.min():.3f}, {joint_cmd.joint_positions.max():.3f}]"
            )
        except Exception as e:
            logger.error(f"Error processing gamepad state: {e}", exc_info=True)
    
    def _on_gamepad_state_krabby(self, state: ControllerState) -> None:
        """Callback for gamepad state updates in Krabby mode.
        
        Maps controller state to joint command and sends via HAL client.
        
        Args:
            state: Controller state
        """
        if not self._running:
            return
        
        try:
            # Map controller state to joint command
            joint_cmd = self._gamepad_to_krabby_hal_mapper.map(state, observation_timestamp_ns=None)
            
            # Send command via HAL client
            self._hal_client.put_joint_command(joint_cmd)
            
            logger.debug(
                f"Sent joint command: "
                f"joint_range=[{joint_cmd.joint_positions.min():.3f}, {joint_cmd.joint_positions.max():.3f}]"
            )
        except Exception as e:
            logger.error(f"Error processing gamepad state: {e}", exc_info=True)
    
    def is_running(self) -> bool:
        """Check if control loop is running.
        
        Returns:
            True if running, False otherwise
        """
        return self._running
