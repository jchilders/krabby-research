"""WebRTC-backed input controller that mirrors ``InputController`` callback/state API.

The controller does not own a network transport. Instead, callers feed decoded payloads
from a WebRTC data channel via ``update_from_payload``.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from controller.input.state import ControllerState


class WebRTCInputController:
    """Thread-safe state holder + callback broadcaster for remote gamepad payloads."""

    def __init__(self) -> None:
        self._state = ControllerState()
        self._state_lock = threading.Lock()
        self._callback_lock = threading.Lock()
        self._callbacks: list[Callable[[ControllerState], None]] = []
        self._running = False

    def start(self, *, update_rate_hz: float = 50.0, device_id: int | None = None) -> None:
        """Match ``InputController.start`` signature for drop-in wiring.

        ``device_id``/``update_rate_hz`` are accepted for interface compatibility.
        """
        if update_rate_hz <= 0:
            raise ValueError(f"update_rate_hz must be greater than 0, got {update_rate_hz}")
        _ = device_id
        self._running = True

    def stop(self) -> None:
        self._running = False

    def register_callback(self, callback: Callable[[ControllerState], None]) -> None:
        with self._callback_lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[ControllerState], None]) -> None:
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def get_state(self) -> ControllerState:
        with self._state_lock:
            return ControllerState(
                LT=self._state.LT,
                LB=self._state.LB,
                LS=self._state.LS,
                RS=self._state.RS,
                RT=self._state.RT,
                RB=self._state.RB,
                LX=self._state.LX,
                LY=self._state.LY,
                RX=self._state.RX,
                RY=self._state.RY,
            )

    def update_from_payload(self, payload: dict[str, Any]) -> ControllerState:
        """Validate + apply a remote controller payload, then notify callbacks."""
        if not self._running:
            raise RuntimeError("WebRTCInputController must be started before update_from_payload")
        state = self._state_from_payload(payload)
        with self._state_lock:
            self._state = state
        self._notify_callbacks(state)
        return state

    @staticmethod
    def _state_from_payload(payload: dict[str, Any]) -> ControllerState:
        def _b(name: str) -> bool:
            return bool(payload.get(name, False))

        def _f(name: str) -> float:
            raw = payload.get(name, 0.0)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"controller payload field {name!r} must be numeric")
            if val < -1.0:
                return -1.0
            if val > 1.0:
                return 1.0
            return val

        return ControllerState(
            LT=_b("LT"),
            LB=_b("LB"),
            LS=_b("LS"),
            RS=_b("RS"),
            RT=_b("RT"),
            RB=_b("RB"),
            LX=_f("LX"),
            LY=_f("LY"),
            RX=_f("RX"),
            RY=_f("RY"),
        )

    def _notify_callbacks(self, state: ControllerState) -> None:
        with self._callback_lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            cb(state)
