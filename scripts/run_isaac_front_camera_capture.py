"""Run Isaac Sim with a ZED-like front camera and capture RGB + depth.

Uses the parkour default scene (terrain, robot, sky) plus front_camera (depth) and
front_rgb (RGB) from hal/server/isaac/zed_like_scene_cfg. Does not modify parkour.

Usage (from repo root, with Isaac Sim environment):
  python scripts/run_isaac_front_camera_capture.py
  python scripts/run_isaac_front_camera_capture.py --headless  # print stats only
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Run Isaac Sim with ZED-like front camera and capture RGB + depth."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument(
    "--headless",
    action="store_true",
    help="Do not open display window; only print frame stats",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after AppLauncher so Isaac Sim context is set
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from hal.server.isaac.zed_like_scene_cfg import ZedLikeSceneCfg


def main() -> int:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 0.0, 1.5], target=[0.0, 0.0, 0.5])

    scene_cfg = ZedLikeSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    if not hasattr(scene, "sensors") or "front_camera" not in scene.sensors or "front_rgb" not in scene.sensors:
        print("[ERROR] Scene needs front_camera and front_rgb sensors.", file=sys.stderr)
        return 1

    depth_sensor = scene.sensors["front_camera"]
    rgb_sensor = scene.sensors["front_rgb"]
    sim_dt = sim.get_physics_dt()
    step = 0

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1

        if step % 100 == 0:
            rgb, depth, rgb_np, d_np = None, None, None, None
            if hasattr(rgb_sensor, "data") and hasattr(rgb_sensor.data, "output"):
                rgb = rgb_sensor.data.output.get("rgb")
                if rgb is not None:
                    rgb_np = rgb[0].detach().cpu().numpy() if rgb.dim() > 3 else rgb.detach().cpu().numpy()
            if hasattr(depth_sensor, "data") and hasattr(depth_sensor.data, "output"):
                depth = depth_sensor.data.output.get("distance_to_camera")
                if depth is not None:
                    d = depth[0] if depth.dim() > 2 else depth
                    d_np = d.detach().cpu().numpy()
            if rgb_np is not None:
                print(f"Step {step}: rgb shape={rgb_np.shape}")
            if d_np is not None and d_np.size > 0:
                print(f"Step {step}: depth shape={d_np.shape} range=[{d_np.min():.3f}, {d_np.max():.3f}]")

            if args_cli.headless and step >= 300:
                break

            if not args_cli.headless and step == 200 and (rgb_np is not None or d_np is not None):
                try:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    if rgb_np is not None:
                        rgb_show = rgb_np[0] if rgb_np.ndim > 3 else rgb_np
                        rgb_show = rgb_show[..., :3] if rgb_show.shape[-1] >= 3 else rgb_show
                        axes[0].imshow(rgb_show)
                    axes[0].set_title("Front camera RGB")
                    axes[0].axis("off")
                    if d_np is not None:
                        d_show = (d_np[0] if d_np.ndim > 2 else d_np).squeeze()
                        im = axes[1].imshow(d_show, cmap="viridis")
                        plt.colorbar(im, ax=axes[1])
                    axes[1].set_title("Front camera depth (m)")
                    axes[1].axis("off")
                    plt.suptitle("ZED-like front camera capture")
                    plt.tight_layout()
                    plt.show()
                except ImportError:
                    print("matplotlib not available; skip display")
                break

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
