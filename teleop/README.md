# Teleop

## What this is

Teleop provides remote operator video viewing over WebRTC using two components:

- **`teleop.edge`** (`krabby-teleop-edge`) runs on robot-side systems and dials outbound signaling to a remote portal.
- **`teleop.portal`** (`krabby-teleop-portal`) runs on an operator-reachable host and serves HTTP + WebSocket signaling relay.

After offer/answer + ICE setup, media is browser-to-robot.

## Why it is split

- **Deployment separation**: robot and operator hosts need different dependencies and runtime surfaces.
- **Network model**: robot keeps outbound-only signaling to the portal.
- **Packaging clarity**: edge and portal ship as separate wheels.

## How it works (high level)

1. Browser opens portal UI (`/`).
2. Browser fetches ICE bootstrap from `/api/teleop-config`.
3. Browser connects to portal signaling (`/ws/browser`).
4. Robot edge agent connects outbound to portal signaling (`/ws/robot`).
5. Portal relays signaling JSON; browser and robot negotiate direct media.

## Packages and build

| Wheel | Path | Install on |
|--------|------|----------------|
| **`krabby-teleop-edge`** | **`teleop/edge/`** | Robots (Jetson HAL `--teleop`) |
| **`krabby-teleop-portal`** | **`teleop/portal/`** | Operator server / test images |

Build both wheels with `make build-wheels`.

Outputs:

- `teleop/edge/dist/*.whl`
- `teleop/portal/dist/*.whl`

## Run basics

- **Robot side**: set `TELEOP_EDGE_MODE="agent"` and `SERVER_SIGNALING_WS_URL` in `teleop.edge.robot_settings`, then run Jetson HAL with `--teleop`.
- **Portal side**: run `krabby-teleop-portal --host 0.0.0.0 --port 9000`.
- **Smoke edge signaling**: `python scripts/teleop_smoke.py signaling --url ws://127.0.0.1:9000/ws/robot`.

## Testing

- Teleop tests live under `tests/unit/teleop/`.
- Normal `make test` includes them in the x86 test image run.
- Dev helper script: `scripts/teleop_smoke.py` (`http` or `signaling`).
