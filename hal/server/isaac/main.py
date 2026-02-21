"""Entry point for IsaacSim HAL server with integrated inference client.

This entry point runs both the HAL server and inference client in the same process
using inproc ZMQ for zero-copy communication. This is the recommended deployment
for production use where server and client run together.

For standalone server mode (client runs separately), use TCP endpoints instead.

Simulates a robot environment (default: single robot), gathers observations,
runs inference, and applies commands to control the robot. Supports visual display
and video recording.
"""

import argparse
import logging
import os
import signal
import sys
import time

import numpy as np
import torch
from isaaclab.app import AppLauncher


logger = logging.getLogger(__name__)


def main():
    """Main entry point for IsaacSim HAL server with integrated inference."""
    parser = argparse.ArgumentParser(
        description="IsaacSim HAL server with integrated inference client"
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--control_rate",
        type=float,
        default=100.0,
        help="Control loop rate in Hz",
    )
    parser.add_argument(
        "--inference_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )

    # IsaacSim arguments
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., Isaac-Anymal-D-v0 or Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0)",
    )

    # Environment arguments
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record videos during execution",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=500,
        help="Length of the recorded video in steps",
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for environment randomization (default: 42)",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Run in real-time, if possible",
    )

    # HAL endpoints (inproc for same-process communication)
    parser.add_argument(
        "--observation_bind",
        type=str,
        default="inproc://hal_observation",
        help="Observation endpoint (inproc for same-process)",
    )
    parser.add_argument(
        "--command_bind",
        type=str,
        default="inproc://hal_commands",
        help="Command endpoint (inproc for same-process)",
    )

    # Add AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    # Always enable cameras - required for camera sensors (CameraCfg) to work
    # Camera sensors will fail to initialize without --enable_cameras flag
    args.enable_cameras = True
    logger.info(f"Setting enable_cameras=True (required for camera sensors)")

    # Launch IsaacLab
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Wait for app window to be created (needed for camera controller)
    # The window may not exist immediately after AppLauncher starts
    import omni.appwindow
    import time
    max_wait_time = 5.0
    wait_interval = 0.1
    elapsed = 0.0
    while elapsed < max_wait_time:
        app_window = omni.appwindow.get_default_app_window()
        if app_window is not None:
            break
        time.sleep(wait_interval)
        elapsed += wait_interval
    
    if omni.appwindow.get_default_app_window() is None:
        logger.warning("App window not available after waiting. Camera controller may fail.")

    # Import after AppLauncher to avoid conflicts
    import sys
    from isaaclab_tasks.utils import parse_env_cfg
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    
    # Import parkour_tasks to register gym environments
    # Add parkour_tasks to sys.path to ensure it's found
    parkour_tasks_path = "/workspace/parkour/parkour_tasks"
    if parkour_tasks_path not in sys.path:
        sys.path.insert(0, parkour_tasks_path)
    
    # Import packages to register gym environments
    # This must happen before parse_env_cfg is called
    import isaaclab_tasks  # noqa: F401
    import parkour_tasks  # noqa: F401

    from hal.client.config import HalClientConfig
    from hal.server import HalServerConfig
    from hal.server.isaac import IsaacSimHalServer
    from hal.server.isaac.robot_definition_krabby_quad import KRABBY_QUAD_DEFINITION
    from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
    from compute.parkour.inference_client import ParkourInferenceClient
    from compute.parkour.policy_interface import ModelWeights

    # Running flag for graceful shutdown
    running = True

    def signal_handler(sig, frame):
        """Handle interrupt signals."""
        nonlocal running
        logger.info("Received interrupt signal, stopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    hal_server = None
    parkour_client = None
    env = None

    # Parse environment configuration
    # Note: parse_env_cfg() will import parkour_tasks internally, which triggers
    # gym registration, but we bypass gym.make() and use direct instantiation instead
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,  # Use AppLauncher's device for environment
        use_fabric=not args.disable_fabric,
    )
    
    # Set seed for environment randomization
    env_cfg.seed = args.seed

    # Determine render mode based on video flag
    # For visual display, use None (default window rendering)
    # For video recording, use "rgb_array" 
    render_mode = "rgb_array" if args.video else None
    
    # Create environment using gym.make() to ensure proper configuration
    # This ensures all gym environment registration and configuration is properly applied
    import gymnasium as gym
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    
    logger.info(f"Created IsaacSim environment: {args.task} with {env.unwrapped.num_envs} parallel environments (render_mode={render_mode})")

    # Reset environment to initialize state
    # env.reset() calls observation_manager.compute() internally (line 178 in parkour_manager_based_env.py)
    # This is the first compute() call. We need a second compute() call after reset to match
    # the environment's initialization pattern, which we do by calling set_observation() after reset.
    reset_obs_dict, reset_extras = env.reset()
    logger.info("Environment reset complete")

    # Create HAL server config
    hal_server_config = HalServerConfig(
        observation_bind=args.observation_bind,
        command_bind=args.command_bind,
    )

    # Create and initialize HAL server
    robot_definition = KRABBY_QUAD_DEFINITION
    model_definition = PARKOUR_MODEL_OBSERVATION_DEFINITION
    observation_dimensions = model_definition.get_observation_dimensions(robot_definition)
    hal_server = IsaacSimHalServer(hal_server_config, robot_definition, env=env, observation_dimensions=observation_dimensions)
    hal_server.initialize()
    logger.info("HAL server initialized")

    # Get transport context for inproc connections
    transport_context = hal_server.get_transport_context()

    # Create HAL client config
    hal_client_config = HalClientConfig(
        observation_endpoint=args.observation_bind,
        command_endpoint=args.command_bind,
    )

    model_weights = ModelWeights(
        checkpoint_path=args.checkpoint,
        observation_dimensions=observation_dimensions,
        action_dim=model_definition.action_dim,
    )
    parkour_client = ParkourInferenceClient(
        hal_client_config=hal_client_config,
        model_weights=model_weights,
        observation_dimensions=observation_dimensions,
        robot_definition=robot_definition,
        control_rate=args.control_rate,
        device=args.inference_device,
        transport_context=transport_context,
    )
    # Initialize inference client first (creates model)
    parkour_client.initialize()
    logger.info("Parkour inference client initialized")

    # Start inference client in separate thread
    parkour_client.start_thread(running_flag=lambda: running)

    logger.info(f"Starting integrated loop at {args.control_rate} Hz")
    period_s = 1.0 / args.control_rate
    
    # Get environment step dt for real-time mode
    dt = env.unwrapped.step_dt
    timestep = 0

    # Publish initial observation through HAL
    # This is the second compute() call after reset. We need this because gym.make() + reset()
    # requires a second observation computation to match the environment's initialization pattern.
    # set_observation() will compute the observation (second call) and cache it
    hal_server.set_observation()

    # Main loop: step simulation and publish observations
    # Pattern: wait for action (for observation we published) -> step -> publish new observation
    # ALL communication must go through HAL - no direct inference calls
    # Note: Initial observation was already published before the loop
    timestep = 0
    while running and simulation_app.is_running():
        loop_start_ns = time.time_ns()

        # Wait for action corresponding to the observation we published (synchronous matching)
        # For timestep 0, this is the initial observation published before the loop
        # For subsequent timesteps, this is the observation published at the end of the previous iteration
        # apply_command() will loop internally until command received or timeout (throws if timeout)
        # apply_command() automatically expands single action to all environments if num_envs > 1
        action = hal_server.apply_command()
        
        # Ensure action has correct shape [num_envs, action_dim]
        # apply_command() should already expand to [num_envs, action_dim] if needed
        num_envs = env.unwrapped.num_envs
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] != num_envs:
            # If we got a single action, expand it to all environments
            if action.shape[0] == 1:
                action = action.expand(num_envs, -1)
            else:
                raise ValueError(f"Action tensor shape {action.shape} != [{num_envs}, action_dim]")
        action = action.to(device=env.unwrapped.device)

        # Track last applied action BEFORE stepping, so set_observation() can use it for previous_action
        # Extract first 12 joints (action_dim) from action tensor for first environment
        action_np = action[0].cpu().numpy() if action.ndim == 2 else action.cpu().numpy()
        if len(action_np) >= 12:
            hal_server._last_applied_action[:] = action_np[:12].astype(np.float32)
        else:
            logger.warning(f"Action length {len(action_np)} < 12, only tracking {len(action_np)} values")
            hal_server._last_applied_action[:len(action_np)] = action_np.astype(np.float32)
        
        # Step environment with action for all environments
        # Note: env.step() increments episode_length_buf BEFORE computing observations
        obs_dict, _, _, _, extras = env.step(action)
        
        # Cache the observation from env.step() for set_observation() to use
        # We use the observation directly from env.step() to avoid recomputing it, which ensures
        # we use the exact observation that the environment computed with the correct previous_action
        # from action_history_buf (updated during env.step() via action_manager.process_action()).
        hal_server._latest_obs_dict = obs_dict
        hal_server._latest_obs_tensor = obs_dict["policy"]
        
        # Publish new observation through HAL for next iteration
        # set_observation() will use the cached observation from env.step()
        # The inference client running in background thread will process it and send back action
        hal_server.set_observation()

        timestep += 1
        
        # Handle video recording limit
        if args.video:
            if timestep >= args.video_length:
                logger.info(f"Reached video length limit ({args.video_length} steps), stopping...")
                break
        
        # Timing control
        if args.real_time:
            # Real-time mode: sleep based on environment step dt
            loop_end_ns = time.time_ns()
            loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
            sleep_time = dt - loop_duration_s
            if sleep_time > 0:
                time.sleep(sleep_time)
        else:
            # Fixed rate mode: sleep based on control rate
            loop_end_ns = time.time_ns()
            loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
            sleep_time = max(0.0, period_s - loop_duration_s)

            if sleep_time > 0:
                time.sleep(sleep_time)

    # Clean up in reverse order of creation
    if parkour_client:
        parkour_client.close()
    if hal_server:
        hal_server.close()
    if env:
        env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

