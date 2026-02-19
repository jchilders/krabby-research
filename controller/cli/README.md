# krabby-uno and krabby-uno-sim CLIs

## Install

When installing from source, install the HAL client first. From the **krabby-research** directory:

```bash
pip install ./hal/client
pip install ./controller
```

Use `pip install -e ./hal/client` and `pip install -e ./controller` for editable installs. With a venv activated, after both are installed, `krabby-uno` and `krabby-uno-sim` are on PATH. A single `pip install .` from the controller directory only works once `krabby-hal-client` is already installed.

## krabby-uno (real HAL)

1. **Start the HAL server** (one terminal, from `krabby-research`):

   ```bash
   python controller/scripts/jetson/main_gamepad_only.py
   ```

   Server binds observation `tcp://*:6001` and command `tcp://*:6002` by default.

2. **Run the client** (second terminal):

   ```bash
   krabby-uno
   ```

   Defaults: observation `tcp://localhost:6001`, command `tcp://localhost:6002`. Override with `--observation_endpoint` and `--command_endpoint`. Use `--device-id` or `--InputController <id>` for a specific gamepad.

## krabby-uno-sim (IsaacSim)

1. Start the IsaacSim HAL server first (e.g. from Isaac or `controller/scripts/demo/test_gamepad_to_isaacsim_hal.py`).

2. Run:

   ```bash
   git status
   ```

   Defaults: observation `tcp://127.0.0.1:5555`, command `tcp://127.0.0.1:5556`.

## Gamepad

List devices: `python -m controller.input --list`
