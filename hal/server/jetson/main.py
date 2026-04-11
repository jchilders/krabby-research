"""Entry point for Jetson HAL server with integrated inference client.

This entry point runs both the HAL server and inference client in the same process
using inproc ZMQ for zero-copy communication. This is the recommended deployment
for production use where server and client run together on the robot.

For standalone server mode (client runs separately), use TCP endpoints instead.

Gathers observations from real hardware (camera, sensors), runs inference,
and applies commands to control the robot actuators.
"""

import argparse
import logging
import signal
import sys
import threading
import time

from data_collection.collector import start_collector_thread
from data_collection.collector_settings import build_data_collector_config
from hal.client.config import HalClientConfig
from hal.server import HalServerConfig
from hal.server.jetson import JetsonHalServer
from compute.parkour.inference_client import ParkourInferenceClient
from compute.parkour.policy_interface import ModelWeights
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from hal.server.jetson.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for Jetson production deployment."""
    parser = argparse.ArgumentParser(description="Jetson production deployment with HAL server and inference")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--control_rate", type=float, default=100.0, help="Control loop rate in Hz")
    parser.add_argument(
        "--inference_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
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
    parser.add_argument(
        "--data-collector",
        action="store_true",
        help="Enable second HalClient + rosbag2 (mcap) recording (settings: data_collection/collector_settings.py)",
    )
    parser.add_argument(
        "--data-collector-output-dir",
        type=str,
        default=None,
        help="Override bag output directory (default: DEFAULT_OUTPUT_DIR in collector_settings.py)",
    )

    args = parser.parse_args()

    model_definition = PARKOUR_MODEL_OBSERVATION_DEFINITION
    robot_definition = KRABBY_HEX_DEFINITION
    observation_dimensions = model_definition.get_observation_dimensions(robot_definition)

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
    collector_stop: threading.Event | None = None
    collector_thread: threading.Thread | None = None

    try:
        # Create HAL server config (inproc for production)
        hal_server_config = HalServerConfig(
            observation_bind=args.observation_bind,
            command_bind=args.command_bind,
        )

        # Create and initialize HAL server
        hal_server = JetsonHalServer(
            hal_server_config,
            observation_dimensions=observation_dimensions,
            action_dim=model_definition.action_dim,
            robot_definition=robot_definition,
        )
        hal_server.initialize()

        # Initialize hardware (camera, sensors, actuators)
        # TODO: Re-enable camera initialization once ZED SDK/pyzed is properly configured
        # hal_server.initialize_camera()
        hal_server.initialize_sensors()
        hal_server.initialize_actuators()

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
        parkour_client.initialize()
        logger.info("Parkour inference client initialized")

        # Start inference client in separate thread
        parkour_client.start_thread(running_flag=lambda: running)

        if args.data_collector:
            dc_cfg = build_data_collector_config(
                observation_endpoint=args.observation_bind,
                command_endpoint=args.command_bind,
                output_dir=args.data_collector_output_dir,
            )
            collector_stop = threading.Event()
            _collector, collector_thread = start_collector_thread(
                dc_cfg,
                transport_context,
                collector_stop,
            )
            _collector.initialize()
            collector_thread.start()
            logger.info("HalDataCollector thread started (output_dir=%s)", dc_cfg.output_dir)

        logger.info(f"Starting production loop at {args.control_rate} Hz")
        period_s = 1.0 / args.control_rate

        # Main loop: HAL server operations
        try:
            while running:
                loop_start_ns = time.time_ns()

                # Publish observations from real sensors
                hal_server.set_observation()

                # Try to get joint command from inference client (non-blocking)
                # This command was generated from observations we published in a PREVIOUS iteration
                # We'll apply it in THIS iteration
                command = hal_server.get_joint_command(timeout_ms=1)  # 1ms timeout for non-blocking check
                if command is not None:
                    # Apply the command to actuators
                    hal_server.apply_command(command)
                # If no new command available, reuse the last command to maintain current pose
                # This allows the robot to continue while inference processes observations
                # The robot will continue moving based on the last applied command

                # Timing control
                loop_end_ns = time.time_ns()
                loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
                sleep_time = max(0.0, period_s - loop_duration_s)

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    if loop_duration_s > period_s * 1.1:
                        logger.warning(
                            f"Loop unable to keep up! "
                            f"Frame time: {loop_duration_s*1000:.2f}ms "
                            f"exceeds target: {period_s*1000:.2f}ms"
                        )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Failed to run Jetson HAL server: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if collector_stop is not None:
            collector_stop.set()
        if collector_thread is not None and collector_thread.is_alive():
            collector_thread.join(timeout=8.0)
            if collector_thread.is_alive():
                logger.warning("HalDataCollector thread did not exit within timeout")
        # Clean up in reverse order of creation
        if parkour_client:
            parkour_client.close()
        if hal_server:
            hal_server.close()


if __name__ == "__main__":
    main()
