#!/usr/bin/env python3
"""Full end-to-end test: Gamepad → ControlLoop → HAL → IsaacSimMCUSDK.
It uses a mock IsaacSim environment to test the complete end-to-end flow and a

This script tests the complete end-to-end flow:
1. InputController reads gamepad events
2. ControlLoop receives GamepadControlData via callback
3. GamepadToIsaacSimHALMapper converts to JointCommand
4. HALClient sends command to HAL server
5. IsaacSimHalServer receives command and calls IsaacSimMCUSDK.apply_command()
6. IsaacSimMCUSDK logs the command in Isaac's preferred joint format

Usage:
    python controller/scripts/demo/test_gamepad_to_isaacsim_hal.py

Requirements:
    - A gamepad/joystick connected (Bluetooth or USB)
    - The inputs library installed: pip install inputs
    - Note: On macOS, you may need to use pygame instead (see controller/input/pygametemp/)
        - if you want to use pygame on MacOS to test, replace the InputController import in the control_loop.py file with the pygame version
        -   In control_loop.py, Use "from controller.input.pygametemp.input_controller_test_pygame import InputController" instead of "from controller.input import InputController"
        - To make it easy to remove pygame in future, pygame references have only been kept in the pygametemp folder and not in the main input folder. It is also kept here as this is a demo script.

Note: 
1. When the test is run, it keeps on logging the joint commands in Isaac's format continuously. 
To see specific action, you can press a specific gamepad button(like LB, etc) and then move the joystick(like left stick, right stick, etc) to see the corresponding joint commands in Isaac's format.
2. TODO As of 1/15/2026, this is only tested on MacOS. It is not tested on Ubuntu/Linux yet. Again, as mentioned above, do not forget to set the pygame version in control_loop.py if you want to use pygame on MacOS to test.
            
"""

import os
import sys

# Add project root to Python path so we can import controller, hal, etc.
# Script is in controller/scripts/demo/, so go up 3 levels to reach project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "../../..")
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import signal
import time
import threading
from unittest.mock import MagicMock
from typing import Optional

import numpy as np
import torch

from controller.control_loop import ControlLoop, ControlLoopConfig, ControlMode
from hal.client.config import HalClientConfig
from hal.server import HalServerConfig
from hal.server.isaac import IsaacSimHalServer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable DEBUG for IsaacSimMCUSDK to see joint command logs
logging.getLogger('hal.server.isaac.isaacsim_mcusdk').setLevel(logging.DEBUG)
# Reduce ControlLoop logging to WARNING to avoid spam (it logs every command at INFO)
# We'll see the actual commands in IsaacSimMCUSDK logs when they change
logging.getLogger('controller.control_loop').setLevel(logging.WARNING)
# Keep InputController at INFO to see if gamepad is detected (but it's less verbose)
logging.getLogger('controller.input').setLevel(logging.INFO)
# Reduce other loggers to WARNING to avoid spam
logging.getLogger('controller.mappers.gamepad_to_isaacsim_hal_mapper').setLevel(logging.WARNING)
logging.getLogger('hal.server.server').setLevel(logging.WARNING)  # Reduce HAL server ZMQ spam
logging.getLogger('hal.client').setLevel(logging.WARNING)  # Reduce HAL client spam

logger = logging.getLogger(__name__)


def create_mock_isaac_env():
    """Create a mock IsaacSim environment for testing.
    
    Returns:
        Mock environment with required attributes for IsaacSimHalServer.
    """
    # Create mock contact sensor
    mock_contact_sensor = MagicMock()
    mock_contact_sensor.cfg = MagicMock()
    mock_contact_sensor.cfg.body_ids = [0, 1, 2, 3]
    mock_contact_sensor.data = MagicMock()
    # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
    mock_contact_sensor.data.net_forces_w_history = torch.zeros((1, 10, 4, 3), dtype=torch.float32)
    
    # Create mock robot (18 joints for hexapod: 6 legs × 3 DOF)
    mock_robot = MagicMock()
    mock_robot.data = MagicMock()
    mock_robot.data.joint_pos = torch.zeros((1, 18), dtype=torch.float32)  # 18 joints for hexapod
    mock_robot.data.root_ang_vel_b = torch.zeros((1, 3), dtype=torch.float32)
    mock_robot.data.root_lin_vel_b = torch.zeros((1, 3), dtype=torch.float32)
    mock_robot.data.root_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)  # Identity quaternion
    mock_robot.data.joint_vel = torch.zeros((1, 18), dtype=torch.float32)
    
    # Create mock scene
    mock_scene = MagicMock()
    mock_scene.__getitem__ = MagicMock(return_value=mock_robot)  # For env.scene["robot"]
    mock_scene.__contains__ = MagicMock(return_value=True)  # For "robot" in env.scene
    mock_scene.keys = MagicMock(return_value=["robot"])  # For scene.keys() - returns entity names
    mock_scene.sensors = {'contact_forces': mock_contact_sensor}  # Set sensors dict directly
    
    # Create mock environment
    env = MagicMock()
    env.scene = mock_scene
    env.unwrapped = env
    env.unwrapped.num_envs = 1
    env.device = torch.device("cpu")
    
    # Mock observation manager
    env.observation_manager = MagicMock()
    from compute.parkour.parkour_types import OBS_DIM
    obs_tensor = torch.ones(OBS_DIM, dtype=torch.float32) * 0.1
    env.observation_manager.compute = MagicMock(return_value={"policy": obs_tensor})
    
    # Mock action manager
    mock_action_term = MagicMock()
    mock_action_term.action_history_buf = torch.zeros((1, 10, 18), dtype=torch.float32)  # (num_envs, history, action_dim)
    mock_action_manager = MagicMock()
    mock_action_manager.get_term = MagicMock(return_value=mock_action_term)
    env.action_manager = mock_action_manager
    
    # Mock env.step()
    def mock_step(action):
        return ({}, torch.zeros(1), torch.zeros(1, dtype=torch.bool), 
                torch.zeros(1, dtype=torch.bool), {})
    env.step = MagicMock(side_effect=mock_step)
    
    return env

def initialize_pygame_if_needed():
    """Initialize Pygame on macOS if needed for InputController."""
    import platform
    if platform.system() != "Darwin":
        return
    
    try:
        import pygame
        if not pygame.get_init():
            logger.info("Initializing pygame in main thread (macOS requirement)...")
            pygame.init()
            logger.info("Pygame core initialized")
        else:
            logger.info("Pygame core already initialized")
        
        if not pygame.joystick.get_init():
            pygame.joystick.init()
            logger.info("Pygame joystick subsystem initialized")
        else:
            logger.info("Pygame joystick subsystem already initialized")
            
        logger.info("Pygame fully initialized in main thread - safe for background threads to use")
    except Exception as e:
        logger.error(f"Failed to initialize pygame in main thread: {e}", exc_info=True)
        logger.error("This will likely cause crashes on macOS. Exiting.")
        sys.exit(1)


def pump_pygame_events_if_needed(last_pump_time):
    """Pump Pygame events on macOS to update joystick state."""
    import platform
    if platform.system() != "Darwin":
        return
    
    try:
        import pygame
        if pygame.get_init():
            current_time = time.time()
            if current_time - last_pump_time[0] > 0.016:  # ~60Hz max
                pygame.event.pump()
                last_pump_time[0] = current_time
    except Exception:
        pass  # Silently ignore errors


def log_joystick_info(input_controller):
    """Log joystick information if available."""
    if not hasattr(input_controller, '_joystick') or input_controller._joystick is None:
        return
    
    try:
        import pygame
        joystick = input_controller._joystick
        logger.info(f"Joystick info: name={joystick.get_name()}, num_buttons={joystick.get_numbuttons()}, num_axes={joystick.get_numaxes()}")
        if joystick.get_numbuttons() > 0:
            logger.info(f"Sample button states: B7={joystick.get_button(7) if joystick.get_numbuttons() > 7 else 'N/A'}, "
                       f"B9={joystick.get_button(9) if joystick.get_numbuttons() > 9 else 'N/A'}, "
                       f"B10={joystick.get_button(10) if joystick.get_numbuttons() > 10 else 'N/A'}")
        if joystick.get_numaxes() > 0:
            logger.info(f"Sample axis values: A0={joystick.get_axis(0):.2f}, A1={joystick.get_axis(1):.2f}, "
                       f"A4={joystick.get_axis(4) if joystick.get_numaxes() > 4 else 'N/A':.2f}, "
                       f"A5={joystick.get_axis(5) if joystick.get_numaxes() > 5 else 'N/A':.2f}")
    except Exception as e:
        logger.warning(f"Could not read joystick directly: {e}")


def main():
    """Main test function."""
    logger.info("=" * 80)
    logger.info("Full End to End Test: Gamepad → ControlLoop → HAL → IsaacSimMCUSDK")
    logger.info("=" * 80)
    
    # Initialize Pygame if needed
    initialize_pygame_if_needed()
    
    # Create HAL server config (using network endpoints for easier setup)
    hal_server_config = HalServerConfig(
        observation_bind="tcp://127.0.0.1:5555",
        command_bind="tcp://127.0.0.1:5556",
    )
    
    # Create mock IsaacSim environment
    logger.info("Creating mock IsaacSim environment...")
    mock_env = create_mock_isaac_env()
    
    # Create and initialize HAL server
    logger.info("Initializing HAL server...")
    hal_server = IsaacSimHalServer(hal_server_config, env=mock_env)
    hal_server.initialize()
    hal_server.set_debug(True)
    logger.info("HAL server initialized and listening on tcp://127.0.0.1:5555 (obs) and tcp://127.0.0.1:5556 (cmd)")
    
    # Create HAL client config
    hal_client_config = HalClientConfig(
        observation_endpoint="tcp://127.0.0.1:5555",
        command_endpoint="tcp://127.0.0.1:5556",
    )
    
    # Create ControlLoop config
    logger.info("Creating ControlLoop configuration...")
    control_loop_config = ControlLoopConfig(
        mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
        input_controller_device_id=None,  # Use first available gamepad
        input_controller_update_rate_hz=50.0,
        hal_client_config=hal_client_config,
        hal_server_config=hal_server_config,
        mapper_hip_up_down_scale=0.3,
        mapper_knee_out_in_scale=0.3,
        mapper_hip_yaw_scale=0.2,
    )
    
    # Create and start ControlLoop
    logger.info("Starting ControlLoop...")
    control_loop = ControlLoop(control_loop_config)
    control_loop.start()
    logger.info("ControlLoop started - press buttons on your gamepad!")
    
    # Give it a moment to initialize, then get InputController reference
    time.sleep(0.5)
    
    # Get InputController reference AFTER start()
    input_controller = control_loop._input_controller
    if input_controller is None:
        logger.error("InputController is None after start! Check ControlLoop initialization.")
        return
    
    logger.info(f"InputController instance: {input_controller}")
    
    # Log initial state and joystick info
    try:
        state = input_controller.get_state()
        control_data = input_controller.get_control_data()
        logger.info(f"Initial state - LT:{state.LT}, LB:{state.LB}, LS:{state.LS}, RS:{state.RS}, RT:{state.RT}, RB:{state.RB}")
        logger.info(f"Initial sticks - LX:{state.LX:.2f}, LY:{state.LY:.2f}, RX:{state.RX:.2f}, RY:{state.RY:.2f}")
        logger.info(f"Initial control data: hip_up_down={control_data.hip_up_down:.2f}, knee_out_in={control_data.knee_out_in:.2f}, hip_yaw={control_data.hip_yaw:.2f}")
        log_joystick_info(input_controller)
    except Exception as e:
        logger.warning(f"Could not get initial InputController state: {e}")
    
    # Check if gamepad is detected (works for both pygame and non-pygame versions)
    gamepad_detected = False
    if hasattr(input_controller, '_joystick'):
        # Pygame version - check _joystick attribute
        if input_controller._joystick is not None:
            gamepad_detected = True
    else:
        # Non-pygame version - check if inputs library can detect devices
        try:
            from inputs import devices
            gamepads = [device for device in devices if device.name]
            # Filter for gamepad-like devices (keyboards and mice are not gamepads)
            gamepads = [d for d in gamepads if 'keyboard' not in d.name.lower() and 'mouse' not in d.name.lower()]
            if gamepads:
                gamepad_detected = True
                logger.info(f"Found {len(gamepads)} gamepad device(s) via inputs library: {[d.name for d in gamepads]}")
        except Exception as e:
            logger.debug(f"Could not check for gamepad devices via inputs library: {e}")
            # Continue anyway - the event loop will handle errors
    
    if not gamepad_detected:
        logger.error("No gamepad detected. Please connect a gamepad and try again. On macOS, you may need to use pygame instead (see controller/input/pygametemp/ and comments at the top of this file).")
        logger.error("Exiting script - gamepad is required for this test.")
        return
    
    logger.info("Watch for IsaacSimMCUSDK debug logs showing joint commands in Isaac's format")
    logger.info("\nGamepad controls:")
    logger.info("  - LT/LB/LS/RS/RT/RB: Select legs")
    logger.info("  - Left stick Y: Hip up/down")
    logger.info("  - Left stick X: Knee out/in")
    logger.info("  - Right stick Y: Hip yaw")
    logger.info("\nPress Ctrl+C to stop")
    
    # Set up signal handler for graceful shutdown
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        logger.info("\nReceived interrupt signal, stopping...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("\nPolling for commands... (press gamepad buttons to see logs)")
    
    # Background thread to process commands using HAL server's interface
    # 
    # Why this is needed:
    # - The HAL server's apply_command() method blocks until a command is received (or 1s timeout)
    # - This is designed for production use in hal/server/isaac/main.py where the main loop:
    #   1. Calls hal_server.set_observation() to publish observations
    #   2. Calls action = hal_server.apply_command() which blocks until command received
    #   3. Calls env.step(action) to step the simulation
    # - For this test script, we don't have a simulation loop, so we use a background thread
    #   to call apply_command() without blocking the main thread (which needs to pump pygame events)
    #
    # Production case (hal/server/isaac/main.py):
    # - The main simulation loop synchronously calls apply_command() which blocks until command received
    # - This ensures observation → command → step synchronization
    # - No background thread needed because the main loop IS the command processing loop
    #
    # Test script case:
    # - On macOS: We need non-blocking behavior to pump pygame events in the main thread
    #   (pygame requires main-thread event pumping for joystick state updates)
    # - On Ubuntu/Linux: The inputs library handles events in its own background thread,
    #   so no main-thread event pumping is needed
    # - Background thread handles command processing using the server's existing interface
    #   Note: The background thread is only strictly necessary on macOS, but we use it on
    #   both platforms for code simplicity. On Ubuntu, we could call apply_command() directly
    #   in the main loop since there's nothing else the main thread needs to do.
        
    last_pump_time = [0.0]
    
    def command_processing_loop():
        """Background thread that uses HAL server's apply_command() method.
        
        This uses the server's existing interface which:
        - Polls for commands internally
        - Uses the server's internal SDK to convert commands
        - Logs commands via the SDK (which is what we want to see in the test)
        """
        while running:
            try:
                # Use server's apply_command() which handles everything internally
                # This will block until command received (or 1s timeout)
                action = hal_server.apply_command()
                # Command was applied and logged by the server's internal SDK
                # Note: We don't use the returned action tensor in this test script
                # (production code would pass it to env.step(action))
            except RuntimeError as e:
                # Timeout or server not ready - expected, just continue
                if "timeout" in str(e).lower() or "not initialized" in str(e).lower():
                    time.sleep(0.1)  # Brief sleep before retry
                    continue
                # Other runtime errors - log and continue
                logger.debug(f"Command processing error: {e}")
            except Exception as e:
                if not running:
                    break
                logger.debug(f"Unexpected error in command processing: {e}", exc_info=True)
                time.sleep(0.1)
    
    # Start background thread for command processing
    command_thread = threading.Thread(target=command_processing_loop, daemon=True)
    command_thread.start()
    logger.info("Started background thread using HAL server's apply_command()")
    
    try:
        while running:
            # Pump Pygame events if needed
            pump_pygame_events_if_needed(last_pump_time)
            time.sleep(0.1)  # Main loop just waits
            
    except KeyboardInterrupt:
        logger.info("Stopping by user request...")
    finally:
        # Clean up
        logger.info("Cleaning up...")
        control_loop.stop()
        hal_server.close()
        logger.info("Test complete!")
        logger.info("\nExpected log format from IsaacSimMCUSDK:")
        logger.info("  DEBUG - hal.server.isaac.isaacsim_mcusdk - IsaacSimMCUSDK: Applying joint command (...):")
        logger.info("    FL_hip_yaw=0.0000, FL_hip_pitch=0.1500, FL_knee=-0.0900, ...")
        logger.info("  DEBUG - hal.server.isaac.isaacsim_mcusdk - IsaacSimMCUSDK: Joint command stats - min=..., max=..., mean=..., std=...")


if __name__ == "__main__":
    main()
