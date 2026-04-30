"""Run teleop robot agent (outbound WebSocket) using a dedicated HalClient for viewer frames."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Optional

import numpy as np
import zmq
from zmq.error import ContextTerminated

from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.server.sensor_interface import SensorInterface
from teleop.edge.config import TeleopEdgeSettings
from teleop.edge.hal_rgb_track import HalRgbSnapshotVideoTrack
from teleop.edge.portal_client import portal_client_loop
from teleop.edge.viewer_catalog import parse_viewer_catalog_ids_from_payload

logger = logging.getLogger(__name__)


class _ViewerCatalogIds:
    """Thread-safe catalog id list; browser updates via signaling ``catalog_ids``."""

    def __init__(self, bootstrap: list[str], teleop_settings: TeleopEdgeSettings) -> None:
        self._bootstrap = list(bootstrap)
        self._teleop_settings = teleop_settings
        self._lock = threading.Lock()
        self._active = list(bootstrap)

    def apply_from_payload(self, payload: dict[str, Any]) -> None:
        parsed = parse_viewer_catalog_ids_from_payload(
            payload, max_lines=self._teleop_settings.max_video_m_lines
        )
        if parsed is None:
            return
        new_ids = list(self._bootstrap) if not parsed else list(parsed)
        with self._lock:
            self._active = new_ids

    def snapshot_poll_ids(self) -> list[str]:
        with self._lock:
            ids = list(self._active)
        return ids if ids else list(self._bootstrap)

    def catalog_id_for_track(self, track_index: int) -> str:
        ids = self.snapshot_poll_ids()
        if not ids:
            return ""
        idx = min(max(track_index, 0), len(ids) - 1)
        return ids[idx]


def start_jetson_teleop_signaling_thread(
    hal_client_config: HalClientConfig,
    transport_context: zmq.Context,
    sensor_interface: SensorInterface,
    *,
    stop_event: threading.Event,
    bootstrap_sensor_catalog_ids: list[str],
    teleop_edge_settings: TeleopEdgeSettings,
) -> threading.Thread:
    """Start outbound teleop: HalClient poll thread + asyncio signaling; stop when ``stop_event`` is set.

    ``bootstrap_sensor_catalog_ids`` seeds HAL polling until the browser sends ``catalog_ids``
    on ``hello`` / ``offer`` (see portal viewer); empty list is invalid.
    """

    def _thread_main() -> None:
        settings = teleop_edge_settings
        if not settings.agent_enabled or not settings.server_signaling_ws_url:
            logger.error(
                "Teleop requires TELEOP_EDGE_MODE=\"agent\" and SERVER_SIGNALING_WS_URL in "
                "teleop/edge/robot_settings.py; signaling thread not started",
            )
            return

        if not bootstrap_sensor_catalog_ids:
            logger.error("Teleop requires non-empty bootstrap_sensor_catalog_ids; signaling thread not started")
            return

        catalog_state = _ViewerCatalogIds(bootstrap_sensor_catalog_ids, settings)
        available_sensors = sensor_interface.list_sensors()
        sensors_by_id = {s.id: s for s in available_sensors}
        available_catalog_ids = [s.id for s in available_sensors]

        def _validate_sensor_pipeline_binding(track_count: int) -> None:
            selected = catalog_state.snapshot_poll_ids()
            if len(selected) < track_count:
                raise ValueError(
                    "offer rejected: viewer requested more video tracks than selected catalog_ids"
                )
            for idx in range(track_count):
                cid = selected[idx]
                sensor = sensors_by_id.get(cid)
                if sensor is None:
                    raise ValueError(
                        f"offer rejected: catalog_id {cid!r} not found in SensorInterface.list_sensors()"
                    )
                try:
                    handle = sensor_interface.get_gstreamer_handle(sensor)
                    pipeline = sensor_interface.build_pipeline(
                        handle,
                        encoding="h264",
                        output_element="fakesink",
                    )
                except Exception as e:
                    raise ValueError(
                        f"offer rejected: failed sensor pipeline preflight for catalog_id {cid!r}: {e}"
                    ) from e
                if "h264" not in pipeline.lower():
                    raise ValueError(
                        f"offer rejected: sensor pipeline for catalog_id {cid!r} is not H.264"
                    )

        latest_rgb: dict[str, np.ndarray] = {}
        latest_capture_ns: dict[str, int] = {}
        rgb_lock = threading.Lock()
        poll_stop = threading.Event()
        hal_teleop = HalClient(hal_client_config, context=transport_context)

        def _poll_worker() -> None:
            hal_teleop.initialize()
            try:
                while not poll_stop.is_set():
                    try:
                        obs = hal_teleop.poll(timeout_ms=15)
                    except ContextTerminated:
                        # Normal during shutdown when the shared ZMQ context is closed first.
                        logger.debug("teleop HAL client poll stopped (ZMQ context terminated)")
                        break
                    except Exception:
                        logger.warning(
                            "teleop HAL client poll failed; retrying",
                            exc_info=True,
                        )
                        continue
                    if obs is None or obs.rgbd_by_catalog_id is None:
                        continue
                    poll_ids = catalog_state.snapshot_poll_ids()
                    with rgb_lock:
                        for cid in poll_ids:
                            chunk = obs.rgbd_by_catalog_id.get(cid)
                            if chunk is not None and chunk.rgb is not None and chunk.rgb.size > 0:
                                latest_rgb[cid] = np.ascontiguousarray(chunk.rgb, dtype=np.uint8)
                                latest_capture_ns[cid] = int(obs.timestamp_ns)
            finally:
                hal_teleop.close()

        poller = threading.Thread(target=_poll_worker, name="jetson-teleop-hal-poll", daemon=True)
        poller.start()

        def _rgb_copy(catalog_id: str) -> Optional[np.ndarray]:
            with rgb_lock:
                arr = latest_rgb.get(catalog_id)
                if arr is None:
                    return None
                return np.copy(arr)

        def _make_track_factory():
            def factory(track_index: int) -> HalRgbSnapshotVideoTrack:
                cid = catalog_state.catalog_id_for_track(track_index)
                return HalRgbSnapshotVideoTrack(frame_getter=lambda c=cid: _rgb_copy(c))

            return factory

        factory = _make_track_factory()

        def _on_signaling_json(payload: dict[str, Any]) -> None:
            catalog_state.apply_from_payload(payload)

        def _hello_ack_payload() -> dict[str, Any]:
            return {"available_catalog_ids": available_catalog_ids}

        def _pong_payload() -> dict[str, Any]:
            with rgb_lock:
                cap = dict(latest_capture_ns)
            return {"capture_timestamps_ns": cap}

        async def _run() -> None:
            task = asyncio.create_task(
                portal_client_loop(
                    settings.server_signaling_ws_url,
                    teleop_edge_settings=settings,
                    video_track_factory=factory,
                    on_signaling_json=_on_signaling_json,
                    pre_offer_validator=lambda _payload, n: _validate_sensor_pipeline_binding(n),
                    hello_ack_payload_builder=_hello_ack_payload,
                    pong_payload_builder=_pong_payload,
                )
            )
            await asyncio.to_thread(stop_event.wait)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        try:
            asyncio.run(_run())
        except Exception:
            logger.exception("Teleop signaling thread exited with error")
        finally:
            poll_stop.set()
            poller.join(timeout=5.0)
            if poller.is_alive():
                logger.warning("Teleop HAL poll thread did not exit within timeout")

    t = threading.Thread(target=_thread_main, name="jetson-teleop-signaling", daemon=True)
    t.start()
    return t
