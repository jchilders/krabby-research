# End-to-End Test: Pro Controller → Krabby HAL (Two-Process)

Two-process E2E on Jetson Orin: **Pro Controller** → **ControlLoop (INPUT_CONTROLLER_KRABBY)** → **HalClient (ZMQ TCP)** → **Jetson HAL server** → **KrabbyMCUSDK** → **firmware/krabby_mcu.py**. No camera or inference; command path only.

## Prerequisites

- **Pro Controller** connected (USB or Bluetooth). Verify: `python -m controller.input --list`
- **Jetson HAL (MCU)** connected (serial). **Firmware** package: `firmware.krabby_mcu` (see [firmware/](../../../firmware/)).
- Server **exits with error** if firmware or MCU not available.

## Communication

- **Server:** Binds ZMQ — PUB observation, PULL command (e.g. `tcp://*:6001`, `tcp://*:6002`).
- **Client:** Connects — SUB observation, PUSH command (e.g. `tcp://localhost:6001`, `tcp://localhost:6002`).

## Steps

### Terminal 1: HAL server

From **krabby-research**:

```bash
python controller/scripts/jetson/main_gamepad_only.py
```

Custom bind: `--observation_bind tcp://*:6001 --command_bind tcp://*:6002`. Optional: `--mcu-port`, `--mcu-baud` (115200).

Wait for: `Gamepad-only HAL server initialized (ZMQ TCP). ... Waiting for joint commands...`

### Terminal 2: Control-loop client

From **krabby-research**:

```bash
python controller/scripts/jetson/run_gamepad_to_krabby_client.py
```

Explicit endpoints: `--observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002`. Optional: `--device-id N`, `--rate` (Hz).

## Gamepad mapping

- **Legs:** LT / LB / LS / RS / RT / RB ([gamepad_to_krabby_hal_mapper.py](../../mappers/gamepad_to_krabby_hal_mapper.py)).
- **Axes:** Left stick Y = hip; left stick X = knee; right stick Y = hip yaw.

## Stopping

**Ctrl+C** in either terminal.

## Verification

- Server: Missing firmware/MCU → error log and non-zero exit.
- Client: Pro Controller input → joint commands over ZMQ; server applies via KrabbyMCUSDK → firmware.

---

## Running with the locomotion container

The locomotion image includes the controller and firmware source so you can run the gamepad-only HAL server inside the container and the control-loop client on the host (or in the container with gamepad device access).

### Build on Orin

From **krabby-research** on the Jetson Orin:

1. Build wheels: `make build-wheels` (requires venv; see Makefile).
2. Build the locomotion image: `make build-locomotion-image` or:
   `docker build -f images/locomotion/Dockerfile -t krabby-locomotion:latest .`


### How to run (two options)
To run the HaL server and client, there are 2 options: (1) Run Hal Server in container and client in virtual python environment or (2) Run both Hal Server and Client in container in different terminals. The section below details both the options.

### Method 1 - Run Hal Server in container and client in virtual python environment
#### Terminal 1: Gamepad-only HAL server in container

Override the default entrypoint to run the gamepad-only server:

```bash
docker run --rm --privileged --runtime=nvidia \
  -v /dev:/dev \
  -p 6001:6001 -p 6002:6002 \
  --entrypoint python3 \
  krabby-locomotion:latest \
  -m controller.scripts.jetson.main_gamepad_only \
  --observation_bind tcp://*:6001 --command_bind tcp://*:6002
```

- `--privileged`: required so the container can open the serial device (e.g. `/dev/ttyACM0`) for the MCU; otherwise you may see "Operation not permitted".
- `-v /dev:/dev`: serial access for MCU (e.g. `/dev/ttyACM0`).
- Optional: `--mcu-port`, `--mcu-baud` if needed.
- Alternatively: `./controller/scripts/jetson/helper/run_gamepad_hal_server_only_in_container.sh`

#### Terminal 2: Control-loop client (on host)

On the Orin host (with repo and venv), connect the Pro Controller to the host and run:

```bash
python controller/scripts/jetson/run_gamepad_to_krabby_client.py \
  --observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002
```

The client talks to the server in the container via localhost.

### Method 2 - Run both Hal Server and Client in the container (two terminals)
Tip: This is the easient way to run Hal Server and Client

To run the client inside the same container, run the server in **detached** mode so the container stays up, then **exec** the client from a second terminal. The container already sees the host’s `/dev/input/*` via `-v /dev:/dev` and `--privileged`, so connect the Pro Controller to the host **before** starting the container.

**Step 1 – Start the server container in detached mode**

```bash
docker run -d --name hal-gamepad --rm --privileged --runtime=nvidia \
  -v /dev:/dev \
  -p 6001:6001 -p 6002:6002 \
  --entrypoint python3 \
  krabby-locomotion:latest \
  -m controller.scripts.jetson.main_gamepad_only \
  --observation_bind tcp://*:6001 --command_bind tcp://*:6002
```
Alternatively: `./controller/scripts/jetson/helper/start_gamepad_hal_server_container_detached.sh`

**Step 2 – Run the client inside the container (second terminal)**

```bash
docker exec -it hal-gamepad python3 -m controller.scripts.jetson.run_gamepad_to_krabby_client \
  --observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002
```
Alternatively: `./controller/scripts/jetson/helper/run_gamepad_client_in_container.sh`

Optional: `--device-id N`, `--rate` (Hz). To list gamepad devices inside the container: `docker exec -it hal-gamepad python3 -m controller.input --list`.

**Stopping the container.** In the client terminal press Ctrl+C, then from the host run: `docker stop hal-gamepad`. Alternatively: `./controller/scripts/jetson/helper/stop_gamepad_hal_container.sh` (With `--rm`, the container is removed after it stops.)

**Troubleshooting:**
- If the client reports no controller, ensure the Pro Controller is connected to the host before `docker run`, and that the container was started with `-v /dev:/dev` and `--privileged`.
- **Docker permission denied:** If `docker exec` (or `docker run`) fails with "permission denied" connecting to the Docker socket, add your user to the `docker` group on the host: `sudo usermod -aG docker $USER`, then run `newgrp docker` or log out and back in. See [JETSON_DEPLOYMENT.md](../../../docs/JETSON_DEPLOYMENT.md) Troubleshooting.

---

### Using the scripts

Helper scripts in `controller/scripts/jetson/helper/` (run from **krabby-research** repo root):

**Server only (client on host)**  
- **Start server:** `./controller/scripts/jetson/helper/run_gamepad_hal_server_only_in_container.sh` — foreground; Ctrl+C stops the container.  
- **Client:** Run on host (see [Terminal 2: Control-loop client (on host)](#terminal-2-control-loop-client-on-host)).

**Client inside the container**  
- **Start server (detached):** `./controller/scripts/jetson/helper/start_gamepad_hal_server_container_detached.sh`  
- **Run client (second terminal):** `./controller/scripts/jetson/helper/run_gamepad_client_in_container.sh`  
- **Stop:** After Ctrl+C in the client terminal: `./controller/scripts/jetson/helper/stop_gamepad_hal_container.sh`
