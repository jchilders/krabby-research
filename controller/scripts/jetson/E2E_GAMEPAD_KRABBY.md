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
