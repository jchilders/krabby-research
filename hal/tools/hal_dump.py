"""CLI tool to inspect HAL server state."""

import argparse
import logging
import time
from typing import Optional

import numpy as np
import zmq

from compute.parkour.model_definition import (
    PARKOUR_MODEL_OBSERVATION_DEFINITION,
    ObservationDimensions,
)
from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
from hal.client.data_structures.hardware import HardwareObservations
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def format_quaternion(quat: np.ndarray) -> str:
    """Format quaternion for display."""
    return f"[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"


def format_vector(vec: np.ndarray, name: str = "vec") -> str:
    """Format vector for display."""
    return f"[{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]"


def dump_observation_details(
    observation: np.ndarray,
    observation_dimensions: ObservationDimensions,
    action_dim: int,
):
    """Dump detailed breakdown of observation array.

    Args:
        observation: Complete observation array in training format
        observation_dimensions: Layout from model_definition.get_observation_dimensions(robot_definition)
        action_dim: Action dimension (for joint positions/velocities)
    """
    d = observation_dimensions
    if len(observation) != d.obs_dim:
        print(f"  ⚠️  Warning: Observation length {len(observation)} != expected {d.obs_dim}")
        return

    proprioceptive = observation[: d.num_prop]
    scan = observation[d.num_prop : d.num_prop + d.num_scan]
    start_vis = d.num_prop + d.num_scan
    vision = observation[start_vis : start_vis + d.num_vision]
    start_pe = start_vis + d.num_vision
    priv_explicit = observation[start_pe : start_pe + d.num_priv_explicit]
    start_pl = start_pe + d.num_priv_explicit
    priv_latent = observation[start_pl : start_pl + d.num_priv_latent]
    history = observation[-d.history_dim :]

    print(f"\n  Observation Breakdown:")
    print(f"    Total Dimension: {d.obs_dim}")
    print(f"    Proprioceptive ({d.num_prop}):")
    print(f"      Root angular velocity (body frame): {format_vector(proprioceptive[0:3])}")
    print(f"      IMU (roll, pitch): [{proprioceptive[3]:.4f}, {proprioceptive[4]:.4f}]")
    print(f"      Delta yaw: {proprioceptive[6]:.4f}")
    print(f"      Delta next yaw: {proprioceptive[7]:.4f}")
    print(f"      Commands (vx, vy): [{proprioceptive[8]:.4f}, {proprioceptive[9]:.4f}]")
    print(f"      Command (vx): {proprioceptive[10]:.4f}")
    print(f"      Env index: {proprioceptive[11]:.0f}")
    print(f"      Invert env index: {proprioceptive[12]:.0f}")
    if len(proprioceptive) >= 13 + action_dim:
        joint_pos = proprioceptive[13 : 13 + action_dim]
        print(f"      Joint positions ({action_dim}): min={joint_pos.min():.4f}, max={joint_pos.max():.4f}, mean={joint_pos.mean():.4f}")
    if len(proprioceptive) >= 13 + 2 * action_dim:
        joint_vel = proprioceptive[13 + action_dim : 13 + 2 * action_dim]
        print(f"      Joint velocities ({action_dim}): min={joint_vel.min():.4f}, max={joint_vel.max():.4f}, mean={joint_vel.mean():.4f}")

    print(f"    Scan Features ({d.num_scan}):")
    print(f"      Min: {scan.min():.4f}, Max: {scan.max():.4f}, Mean: {scan.mean():.4f}")
    print(f"      First 5: {scan[:5]}")
    print(f"      Last 5: {scan[-5:]}")

    if d.num_vision > 0:
        print(f"    Vision ({d.num_vision}):")
        print(f"      Min: {vision.min():.4f}, Max: {vision.max():.4f}, Mean: {vision.mean():.4f}")

    print(f"    Privileged Explicit ({d.num_priv_explicit}):")
    print(f"      Base linear velocity (world): {format_vector(priv_explicit[0:3])}")

    print(f"    Privileged Latent ({d.num_priv_latent}):")
    print(f"      Body mass: {priv_latent[0]:.4f}")
    print(f"      Body COM: {format_vector(priv_latent[1:4])}")
    print(f"      Friction: {priv_latent[4]:.4f}")

    print(f"    History ({d.history_dim}):")
    print(f"      Min: {history.min():.4f}, Max: {history.max():.4f}, Mean: {history.mean():.4f}")


def dump_hal_state(
    observation_endpoint: str,
    observation_dimensions: ObservationDimensions,
    action_dim: int,
    command_endpoint: Optional[str] = None,
    verbose: bool = False,
):
    """Dump current HAL server state.

    Args:
        observation_endpoint: Observation endpoint (required)
        observation_dimensions: Layout from model_definition.get_observation_dimensions(robot_definition)
        command_endpoint: Command endpoint (optional)
        action_dim: Action dimension for joint parsing
        verbose: Show detailed breakdown
    """
    context = zmq.Context()

    print("=" * 80)
    print("HAL Server State Dump")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Subscribe to observation endpoint
    obs_sub = context.socket(zmq.SUB)
    obs_sub.setsockopt(zmq.SUBSCRIBE, b"observation")
    obs_sub.setsockopt(zmq.RCVHWM, 1)
    obs_sub.connect(observation_endpoint)

    time.sleep(0.1)  # Give time for connection

    if obs_sub.poll(1000, zmq.POLLIN):
        frame = obs_sub.recv()
        topic_prefix = b"observation"
        if frame.startswith(topic_prefix) and len(frame) > len(topic_prefix):
            payload = frame[len(topic_prefix):]
            try:
                hw_obs = HardwareObservations.from_bytes(payload)

                mapper = HWObservationsToParkourMapper(observation_dimensions)
                model_obs = mapper.map(hw_obs)
                observation = model_obs.to_array()

                print(f"\n📊 Observation:")
                print(f"  Topic: observation")
                timestamp_s = hw_obs.timestamp_ns / 1e9
                print(f"  Timestamp: {hw_obs.timestamp_ns} ns ({timestamp_s:.6f} s)")
                print(f"  Shape: {observation.shape}")
                print(f"  Dtype: {observation.dtype}")
                print(f"  Stats: min={observation.min():.3f}, max={observation.max():.3f}, mean={observation.mean():.3f}")

                if verbose and len(observation) == observation_dimensions.obs_dim:
                    dump_observation_details(
                        observation, observation_dimensions, action_dim
                    )
            except Exception as e:
                print(f"\n📊 Observation: Error deserializing - {e}")
    else:
        print("\n📊 Observation: No data available")

    obs_sub.close()

    # Command endpoint (test connection)
    if command_endpoint:
        try:
            command_req = context.socket(zmq.REQ)
            command_req.setsockopt(zmq.RCVTIMEO, 1000)
            command_req.connect(command_endpoint)

            test_command = np.array([0.0] * action_dim, dtype=np.float32)
            command_req.send(test_command.tobytes())

            if command_req.poll(1000, zmq.POLLIN):
                response = command_req.recv()
                print(f"\n⚙️  Command Endpoint:")
                print(f"  Status: ✅ Connected")
                print(f"  Response: {response.decode('utf-8', errors='ignore')}")
                print(f"  Test Command Shape: {test_command.shape}")
                print(f"  Note: Commands are REQ/REP, no history available")
            else:
                print(f"\n⚙️  Command Endpoint: ⚠️  No response (timeout)")

            command_req.close()
        except Exception as e:
            print(f"\n⚙️  Command Endpoint: ❌ Error - {e}")

    print("\n" + "=" * 80)
    context.term()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dump HAL server state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage:
  hal_dump --observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002

  # Verbose mode with detailed breakdown:
  hal_dump --observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002 --verbose

  # In-process endpoints:
  hal_dump --observation_endpoint inproc://hal_observation --command_endpoint inproc://hal_command
        """,
    )
    parser.add_argument(
        "--observation_endpoint",
        type=str,
        default="tcp://localhost:6001",
        help="Observation endpoint (default: tcp://localhost:6001)",
    )
    parser.add_argument(
        "--command_endpoint",
        type=str,
        default="tcp://localhost:6002",
        help="Command endpoint (default: tcp://localhost:6002)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed state breakdown",
    )

    args = parser.parse_args()
    model_definition = PARKOUR_MODEL_OBSERVATION_DEFINITION
    observation_dimensions = model_definition.get_observation_dimensions(
        KRABBY_HEX_DEFINITION
    )
    dump_hal_state(
        observation_endpoint=args.observation_endpoint,
        observation_dimensions=observation_dimensions,
        action_dim=model_definition.action_dim,
        command_endpoint=args.command_endpoint,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

