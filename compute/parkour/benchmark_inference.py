"""Script to benchmark inference latency.

This script measures inference latency to ensure it meets real-time requirements (< 15ms target).
"""

import argparse
import logging
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from compute.parkour.model_definition import (
    PARKOUR_MODEL_OBSERVATION_DEFINITION,
    ObservationDimensions,
)
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from hal.server.jetson.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_dummy_observation_tensor(observation_dimensions: ObservationDimensions, device: str = "cuda") -> torch.Tensor:
    """Create dummy observation tensor for testing (matches production path)."""
    obs_tensor = torch.zeros(1, observation_dimensions.obs_dim, dtype=torch.float32, device=device)
    return obs_tensor


def benchmark_inference(
    checkpoint_path: str,
    observation_dimensions: ObservationDimensions,
    device: str = "cuda",
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    action_dim: int = 12,
):
    """Benchmark inference latency."""
    d = observation_dimensions
    logger.info(f"Benchmarking inference on {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Action dim: {action_dim}, Obs dim: {d.obs_dim}")
    logger.info(f"Iterations: {num_iterations} (warmup: {warmup_iterations})")

    weights = ModelWeights(
        checkpoint_path=checkpoint_path,
        observation_dimensions=observation_dimensions,
        action_dim=action_dim,
    )
    model = ParkourPolicyModel(weights, device=device)
    obs_tensor = create_dummy_observation_tensor(observation_dimensions, device=device)

    # Warmup
    logger.info("Running warmup iterations...")
    for _ in range(warmup_iterations):
        _ = model.inference_tensor(obs_tensor)

    # Synchronize if using CUDA
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    logger.info("Running benchmark iterations...")
    latencies_ms = []

    for i in range(num_iterations):
        if device == "cuda":
            torch.cuda.synchronize()

        start_time_ns = time.time_ns()
        result = model.inference_tensor(obs_tensor)
        end_time_ns = time.time_ns()

        if device == "cuda":
            torch.cuda.synchronize()

        latency_ns = end_time_ns - start_time_ns
        latency_ms = latency_ns / 1_000_000.0
        latencies_ms.append(latency_ms)

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{num_iterations} iterations")

    # Calculate statistics
    results = {
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": np.percentile(latencies_ms, 95),
        "p99_ms": np.percentile(latencies_ms, 99),
        "num_iterations": num_iterations,
        "device": device,
    }

    # Log results
    logger.info("=" * 60)
    logger.info("Benchmark Results")
    logger.info("=" * 60)
    logger.info(f"Mean latency: {results['mean_ms']:.3f} ms")
    logger.info(f"Median latency: {results['median_ms']:.3f} ms")
    logger.info(f"Min latency: {results['min_ms']:.3f} ms")
    logger.info(f"Max latency: {results['max_ms']:.3f} ms")
    logger.info(f"Std deviation: {results['std_ms']:.3f} ms")
    logger.info(f"P50 (median): {results['p50_ms']:.3f} ms")
    logger.info(f"P95: {results['p95_ms']:.3f} ms")
    logger.info(f"P99: {results['p99_ms']:.3f} ms")
    logger.info("=" * 60)

    # Check if meets target
    target_ms = 15.0
    if results["mean_ms"] < target_ms:
        logger.info(f"✓ Mean latency ({results['mean_ms']:.3f} ms) < target ({target_ms} ms)")
    else:
        logger.warning(f"✗ Mean latency ({results['mean_ms']:.3f} ms) >= target ({target_ms} ms)")

    if results["p95_ms"] < target_ms:
        logger.info(f"✓ P95 latency ({results['p95_ms']:.3f} ms) < target ({target_ms} ms)")
    else:
        logger.warning(f"✗ P95 latency ({results['p95_ms']:.3f} ms) >= target ({target_ms} ms)")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")

    args = parser.parse_args()
    observation_dimensions = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions_for_checkpoint(
        args.checkpoint, KRABBY_HEX_DEFINITION
    )
    results = benchmark_inference(
        checkpoint_path=args.checkpoint,
        observation_dimensions=observation_dimensions,
        device=args.device,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        action_dim=PARKOUR_MODEL_OBSERVATION_DEFINITION.action_dim,
    )

    # Exit with error if latency too high
    if results["mean_ms"] >= 15.0 or results["p95_ms"] >= 15.0:
        logger.error("Inference latency does not meet target (< 15ms)")
        exit(1)


if __name__ == "__main__":
    main()

