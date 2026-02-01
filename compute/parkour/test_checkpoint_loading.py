"""Script to test loading model checkpoint.

This script verifies that checkpoints can be loaded successfully.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION, ObservationDimensions
from compute.parkour.parkour_types import ParkourModelIO, ParkourObservation
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from hal.client.observation.types import NavigationCommand
from hal.server.jetson.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_checkpoint_loading(
    checkpoint_path: str,
    observation_dimensions: ObservationDimensions,
    device: str = "cuda",
):
    """Test loading a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        observation_dimensions: Layout from model_definition.get_observation_dimensions(robot_definition)
        device: Device to load on ("cuda" or "cpu")

    Returns:
        True if checkpoint loaded successfully, False otherwise
    """
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False

    d = observation_dimensions
    logger.info(f"Testing checkpoint loading: {checkpoint_path}")
    logger.info(f"Device: {device}, Obs dim: {d.obs_dim}")

    try:
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            else:
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA version: {torch.version.cuda}")

        weights = ModelWeights(
            checkpoint_path=str(checkpoint_path),
            observation_dimensions=observation_dimensions,
            action_dim=PARKOUR_MODEL_OBSERVATION_DEFINITION.action_dim,
        )
        logger.info("Loading model...")
        model = ParkourPolicyModel(weights, device=device)
        logger.info("Model loaded successfully")

        if device == "cuda":
            logger.info("Model loaded on CUDA")

        logger.info("Testing model forward pass...")
        obs_array = np.zeros(d.obs_dim, dtype=np.float32)
        observation = ParkourObservation(
            timestamp_ns=time.time_ns(),
            observation=obs_array,
            observation_dimensions=observation_dimensions,
        )
        model_io = ParkourModelIO(
            timestamp_ns=time.time_ns(),
            nav_cmd=NavigationCommand.create_now(vx=0.0, vy=0.0, yaw_rate=0.0),
            observation=observation,
        )
        result = model.inference(model_io)
        logger.info(f"Inference success: {result.success}")

        logger.info("Checkpoint loading test PASSED")
        return True

    except FileNotFoundError as e:
        logger.error(f"Checkpoint file not found: {e}")
        return False
    except ValueError as e:
        logger.error(f"Checkpoint loading failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test checkpoint loading")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")

    args = parser.parse_args()
    observation_dimensions = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions_for_checkpoint(
        args.checkpoint, KRABBY_HEX_DEFINITION
    )
    success = test_checkpoint_loading(
        checkpoint_path=args.checkpoint,
        observation_dimensions=observation_dimensions,
        device=args.device,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

