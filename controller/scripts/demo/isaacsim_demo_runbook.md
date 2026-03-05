# Isaac Sim joystick demo runbook

**Overview.** This runbook describes how to test Isaac Sim with a Nintendo Pro Controller (or compatible gamepad): run the HAL server in joystick mode, then the `krabby-uno-sim` client on the host. Joystick mode uses a minimal window, disables texture wait and heavy PBR/RTX for faster startup; use the Teacher task for quickest launch (Student adds a depth camera).

## Pairing (Ubuntu)

- `bluetoothctl`: power on, scan-on, pair `<MAC>`, trust `<MAC>`
- Verify: `python -m controller.input --list`

## Prerequisites

The Isaac Sim HAL server must run in an environment where **Isaac Sim** and **Isaac Lab** are available. Two options:

- **Docker (recommended):** Build the image with `make build-isaacsim-image` (requires `make isaaclab-cache` and `make build-wheels`). Then run the server inside the container (see Launch). The client (`krabby-uno-sim`) runs on the host and connects to the container via TCP. When using Docker, set up GPU access first: run `./scripts/setup-docker-gpu.sh` from the repo root, or see [DOCKER_DEPENDENCIES.md](../../../docs/DOCKER_DEPENDENCIES.md) (GPU Support Setup).
- **Native:** Install Isaac Sim and Isaac Lab per [DEVELOPER.md](../../DEVELOPER.md), then run the module with the Isaac Lab Python (e.g. from the Isaac Lab repo: set `PYTHONPATH` to include the krabby-research repo root and run `./isaaclab.sh -p python -m hal.server.isaac.main ...`).

## Launch

1. Start Isaac Sim HAL server in joystick mode (minimal 640×360 window, no Isaac Lab UI). **Only quad (12-joint Go2)** is supported; the registered tasks are Go2-based. Hexapod (18-joint) is not supported—there is no hexapod task, and `--robot hex` will raise an error at startup.

   **Option A – Docker** (from **krabby-research**, after `make build-isaacsim-image`; publish 5555/5556):

   For a **visible Isaac Sim window**, the container needs the host display. The script below already passes `-e DISPLAY` and `-v /tmp/.X11-unix` to `docker run`—do not pass those to the script (they are Docker options, not app options). Run the script with no arguments, or only app options (e.g. `--seed 0`). If you use a raw `docker run` without those flags, the server runs **headless** and you will see: `DISPLAY environment variable is not set, running in headless mode`.

   ```bash
   ./scripts/run_isaac_hal_server.sh
   ```
   Add `--debug` for verbose logs (e.g. per-command joint values). Or manually (includes display for GUI):
   ```bash
   xhost +local:docker 2>/dev/null
   docker run --rm --gpus all -p 5555:5555 -p 5556:5556 \
     -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
     krabby-isaacsim:latest --joystick --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0
   ```

   **Option B – Native** (Isaac Lab env, `PYTHONPATH` includes krabby-research root):
   ```bash
   ./isaaclab.sh -p python -m hal.server.isaac.main --joystick --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0
   ```

   Server binds TCP 5555/5556 by default.

2. Start the **client** in a **second terminal** (install once: `pip install -e ./hal/client` and `pip install -e ./controller`):
   ```bash
   krabby-uno-sim --quad
   ```
   Add `--debug` to log every gamepad→joint command and mapper activity.

   The **Pro Controller is used by the client**, not the server. The server only logs when it receives commands from the client (e.g. `Joystick: first command received`, `Joystick: command applied`). If you move the controller but never start the client, you will see no joystick-related logs on the server.

   Client waits up to 10 min for a gamepad by default (`--gamepad-wait 600`). Override endpoints with `--observation_endpoint tcp://127.0.0.1:5555 --command_endpoint tcp://127.0.0.1:5556` if needed.

In joystick mode the viewport camera is set closer to the robot (eye 0, 1.2, 0.8) so the robot is clearly visible. Move the joystick to confirm joints in the sim.

## Where to see logs

- **Server (Docker):** The terminal where you ran `./scripts/run_isaac_hal_server.sh` **is** the Docker container’s logs (stdout/stderr). You don’t need a separate command to “see Docker logs.” Add `--debug` to that script to get per-command debug lines on the server.
- **Client:** The terminal where you run `krabby-uno-sim --quad` shows client logs. Add `--debug` there to see every gamepad→joint command.
- **When you move the Pro Controller:** The controller is read by the **client** (krabby-uno-sim). The client sends commands to the server. So you only see server-side debug logs (and robot movement) when **both** are running: (1) server in one terminal, (2) `krabby-uno-sim --quad` in a second terminal. Then move sticks/buttons—server terminal will show “Client connected…” and every 5 s “Joystick: N steps…”, and with `--debug` on the server you’ll see “Joint command received: …” for each command.

## Stopping

- **HAL server (Docker):** Ctrl+C in the server terminal often does not stop Isaac Sim. **Use `docker stop` from another terminal:** run `docker ps` to get the container ID (e.g. `abc123def456`), then `docker stop <container_id>`. The container will exit and be removed (`--rm`).
- **HAL server (native):** Press **Ctrl+C** in the terminal running `hal.server.isaac.main`. The process logs "Received interrupt signal, stopping..." and shuts down.
- **Client (krabby-uno-sim):** Press **Ctrl+C** in the client terminal. It logs "Received interrupt, stopping..." and exits.

## Leg selection

LT/LB/LS/RS/RT/RB and combos (LT+LB, RT+RB) follow OVERVIEW Appendix B. Use `krabby-uno-sim --quad` so the client sends 12 joints (FL, FR, RL, RR). Axis scaling: full stick (±1) → ±1.0 rad (hip up/down, knee, hip yaw) so motion is clearly visible; tune in mapper or control loop config if needed.

## Troubleshooting

- **No Isaac Sim window (Docker):** Use `./scripts/run_isaac_hal_server.sh` (it sets up the display), or a manual `docker run` with `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix` and `xhost +local:docker`. Without the display, the app runs headless.
- **`unrecognized arguments: -e DISPLAY -v /tmp/.X11-unix`:** You passed Docker flags to the script; those are for `docker run` only. Run `./scripts/run_isaac_hal_server.sh` with no args (or only app args like `--seed 0`).
- **`Robot joint count (18 for --robot hex) does not match the task's action dimension (12)`:** Hexapod is not supported in this demo; only Go2/quad (12-joint) tasks exist. Omit `--robot hex` and use `./scripts/run_isaac_hal_server.sh` with the client `krabby-uno-sim --quad`.
- **No server logs when moving the Pro Controller:** Start `krabby-uno-sim` in a second terminal. The server only logs when it **receives** commands from the client. Normal server messages: "Client connected (joint command received).", then every 5 s "Joystick: N steps in X.Xs (~XX Hz)". If you see none of these, ensure the server was started with `--joystick` and the client is running and connected.
- **Robot doesn’t move when I move the Pro Controller:** The robot moves only when the **client** sends commands to the server. You must run **`krabby-uno-sim --quad`** in a **second terminal** (on the host, with the controller connected). The server terminal should show "Client connected (joint command received)." when the client connects, then "Joystick: N steps…" every 5 s while you move the sticks. If the server shows **"Joint command received: 12 joints, range=[0.000, 0.000]"** and **"all positions zero"**, the client is sending zeros—**select a leg** (e.g. hold **LT** for front-left, or **RT** for front-right) and move the **left stick** (Y/X = hip/knee); with `--debug` on the server you’ll then see "non-zero positions (joint=rad): …". If you don’t select a leg, all 12 joints stay at 0 and the robot doesn’t move.
- **Client sends commands but server shows no "Client connected" or joystick logs:** The server only logs when it **receives** a command. If the client logs "Sent joint command" but the server never shows "Client connected", the commands are not reaching the server. (1) Ensure the server is fully started—wait until you see **"Joystick: main loop ready, waiting for first command on tcp://*:5556"** in the server terminal before assuming connectivity. (2) From the host, verify the command port is reachable: `nc -zv 127.0.0.1 5556` (or `python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', 5556)); print('5556 open'); s.close()"`). (3) If you changed HAL or server code, rebuild the Docker image (`make build-isaacsim-image`) so the container runs the latest code.
