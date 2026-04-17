"""Shared GStreamer runtime for HAL (Gst.init, parse_launch, appsrc, bus).

Pipeline **strings** remain in ``JetsonSensorInterface`` / ``IsaacSensorInterface`` (``build_pipeline``).
This module **owns** in-process Gst: lazy init, ``Gst.parse_launch``, ``appsrc name=src`` access,
buffer push with timestamps, state changes, and bus **ERROR** / **EOS** handling.

**Runtime contract:** PyGObject as pinned in HAL Docker images (``PyGObject>=3.42,<3.51``). Pipelines
must use ``appsrc name=src``; ``get_by_name("src")`` must yield ``GstApp.AppSrc`` (typical for
``parse_launch`` of that string—no GObject ``cast`` helpers in these bindings).

Threading: ``run_pipeline_with_appsrc_sync`` runs on the **caller's thread**: it polls the bus in
a loop (no ``GLib.MainLoop``). That is enough for **short** headless checks (e.g. ``fakesink``).
For **continuous** encode/stream (live sinks, heavy bus traffic), run the pipeline on
a **dedicated thread** so policy / HAL loops are not blocked; attach a ``GLib.MainLoop`` on that
thread if an element or sink requires it, or keep bus polling co-located with ``push_buffer`` on
the same thread—**do not** share one ``Gst.Bus`` / pipeline across threads without GStreamer's
threading rules for your element mix.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GstApp
    _GST_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:
    Gst = None  # type: ignore[assignment]
    GstApp = None  # type: ignore[assignment]
    _GST_IMPORT_ERROR = e

logger = logging.getLogger(__name__)

_gst_lock = threading.Lock()
_gst_initialized = False


class GstRuntimeUnavailable(RuntimeError):
    """Raised when PyGObject / GStreamer cannot be loaded (e.g. missing gi or plugins)."""


def ensure_gst_initialized() -> None:
    """Idempotent ``Gst.init`` (and GstApp) for the process. Thread-safe."""
    global _gst_initialized
    if _GST_IMPORT_ERROR is not None:
        raise GstRuntimeUnavailable(str(_GST_IMPORT_ERROR)) from _GST_IMPORT_ERROR
    with _gst_lock:
        if _gst_initialized:
            return

        Gst.init(None)
        _gst_initialized = True


def _encoding_element_chain(encoding: str) -> str:
    if encoding == "h264":
        return "x264enc tune=zerolatency ! h264parse"
    if encoding == "h265":
        return "x265enc ! h265parse"
    return "x264enc tune=zerolatency ! h264parse"


def build_software_appsrc_encode_pipeline_string(
    width: int,
    height: int,
    fps: int,
    format_caps: str = "RGB",
    encoding: str = "h264",
    output_element: str = "fakesink",
) -> str:
    """HAL-style **software** pipeline: ``appsrc name=src`` → videoconvert → encode → sink.

    Used by Isaac sim and matches the Jetson software tail (``videoconvert`` + x264/x265).
    """
    caps = (
        f"appsrc name=src is-live=true format=time ! "
        f"video/x-raw,format={format_caps},width={width},height={height},framerate={fps}/1"
    )
    if encoding == "raw":
        return f"{caps} ! videoconvert ! {output_element}"
    enc = _encoding_element_chain(encoding)
    return f"{caps} ! videoconvert ! {enc} ! {output_element}"


@dataclass(frozen=True)
class AppSrcPipelineResult:
    """Outcome of ``run_pipeline_with_appsrc_sync``."""

    success: bool
    n_pushed: int = 0
    error_message: Optional[str] = None


def _numpy_rgb_to_buffer(frame: np.ndarray, pts: int, duration: int, gst_mod: Any) -> Any:
    Gst = gst_mod
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB uint8 array (H,W,3); got shape {frame.shape}")
    data = np.ascontiguousarray(frame).tobytes()
    try:
        buf = Gst.Buffer.new_wrapped(data)
    except (TypeError, AttributeError):
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
    buf.pts = pts
    buf.duration = duration
    return buf


def _numpy_gray16_le_to_buffer(frame: np.ndarray, pts: int, duration: int, gst_mod: Any) -> Any:
    Gst = gst_mod
    if frame.dtype != np.uint16:
        raise ValueError(f"Expected GRAY16_LE uint16 array (H,W); got dtype {frame.dtype}")
    if frame.ndim != 2:
        raise ValueError(f"Expected GRAY16_LE uint16 array (H,W); got shape {frame.shape}")
    data = np.ascontiguousarray(frame).tobytes()
    try:
        buf = Gst.Buffer.new_wrapped(data)
    except (TypeError, AttributeError):
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
    buf.pts = pts
    buf.duration = duration
    return buf


def float32_depth_to_gray16_le(
    depth_m: np.ndarray,
    d_min: float,
    d_max: float,
) -> np.ndarray:
    """Map metric depth (meters, float32) to **GRAY16_LE** uint16.

    **Encode** (this function): ``d_min`` / ``d_max`` are **per stream** (different cameras use
    different bands). In-range depth in ``[d_min, d_max]`` maps **linearly** to **0..65534**;
    **65535** = invalid (NaN, Inf, or outside the band). Encoding:

    ``u = round((d - d_min) / (d_max - d_min) * 65534)`` clipped to **0..65534**.

    **Decode** (inverse, for consumers of the uint16 buffer): use the **same** ``d_min`` /
    ``d_max`` that were used at encode time. For each stored value ``u``:

    - If ``u == 65535``: no valid depth (sentinel).
    - If ``u <= 65534``: ``d = d_min + (u / 65534.0) * (d_max - d_min)`` approximates meters
      (same linear map as encode; rounding on encode means small quantization error).
    """
    if d_max <= d_min:
        raise ValueError("d_max must be greater than d_min")
    d = np.asarray(depth_m, dtype=np.float32)
    out = np.full(d.shape, 65535, dtype=np.uint16)
    valid = np.isfinite(d) & (d >= d_min) & (d <= d_max)
    span = float(d_max - d_min)
    scaled = (d[valid].astype(np.float64) - d_min) / span * 65534.0
    u = np.rint(scaled).astype(np.int64)
    u = np.clip(u, 0, 65534).astype(np.uint16)
    out[valid] = u
    return out


def run_pipeline_with_appsrc_sync(
    pipeline_description: str,
    frames: Sequence[Any],
    *,
    fps: int = 30,
    timeout_s: float = 20.0,
) -> AppSrcPipelineResult:
    """Parse a HAL pipeline string, push frames to ``appsrc name=src``, wait for EOS or ERROR.

    * **Synchronous** — blocks the caller until EOS/ERROR or ``timeout_s`` elapses on the bus.
    * **Frames:** each element is ``numpy.ndarray`` — either ``uint8`` ``(H, W, 3)`` **RGB**, or
      ``uint16`` ``(H, W)`` **GRAY16_LE** (two bytes per pixel, little-endian layout).
    * **Timestamps:** ``pts`` / ``duration`` use ``Gst.SECOND // fps`` per frame (``format=time`` appsrc).

    After the last buffer, sends end-of-stream on the appsrc and drains the bus.
    """
    if not frames:
        return AppSrcPipelineResult(False, 0, "no frames")

    ensure_gst_initialized()

    pipeline: Optional[Gst.Element] = None
    deadline = time.monotonic() + timeout_s

    def remaining_ns() -> int:
        r = deadline - time.monotonic()
        if r <= 0:
            return 0
        return int(r * Gst.SECOND)

    try:
        pipeline = Gst.parse_launch(pipeline_description)
        if pipeline is None:
            return AppSrcPipelineResult(False, 0, "Gst.parse_launch returned None")

        appsrc_el = pipeline.get_by_name("src")
        if appsrc_el is None:
            return AppSrcPipelineResult(
                False,
                0,
                "pipeline has no element named 'src' (expected appsrc name=src)",
            )
        if not isinstance(appsrc_el, GstApp.AppSrc):
            return AppSrcPipelineResult(
                False,
                0,
                f"element 'src' must be GstApp.AppSrc (appsrc name=src), got {type(appsrc_el).__name__}",
            )
        appsrc = appsrc_el

        bus = pipeline.get_bus()

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            return AppSrcPipelineResult(False, 0, "set_state(PLAYING) returned FAILURE")

        state_ret, state, _pending = pipeline.get_state(5 * Gst.SECOND)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            return AppSrcPipelineResult(False, 0, "get_state after PLAYING returned FAILURE")
        if state != Gst.State.PLAYING:
            logger.warning(
                "Pipeline not yet PLAYING (state=%s pending=%s); continuing for fakesink smoke",
                state,
                _pending,
            )

        duration = Gst.SECOND // max(1, int(fps))
        pts = 0
        n = 0
        Gst_mod = Gst
        for frame in frames:
            if hasattr(frame, "ndim") and frame.ndim == 2:
                buf = _numpy_gray16_le_to_buffer(frame, pts, duration, Gst_mod)
            else:
                buf = _numpy_rgb_to_buffer(frame, pts, duration, Gst_mod)
            flow = appsrc.push_buffer(buf)
            if flow not in (Gst.FlowReturn.OK, Gst.FlowReturn.FLUSHING):
                return AppSrcPipelineResult(
                    False,
                    n,
                    f"push_buffer returned {flow!r}",
                )
            pts += duration
            n += 1

        if hasattr(appsrc, "end_stream"):
            appsrc.end_stream()
        else:
            appsrc.emit("end-of-stream")

        while True:
            wait = remaining_ns()
            if wait <= 0:
                return AppSrcPipelineResult(False, n, "timeout waiting for EOS on bus")
            slice_ns = min(wait, int(0.5 * Gst.SECOND))
            msg = bus.timed_pop_filtered(
                slice_ns,
                Gst.MessageType.EOS | Gst.MessageType.ERROR,
            )
            if msg is None:
                continue
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                text = f"{err.message} ({dbg})"
                logger.error("GStreamer ERROR: %s", text)
                return AppSrcPipelineResult(False, n, text)
            if msg.type == Gst.MessageType.EOS:
                logger.debug("GStreamer EOS after %d buffers", n)
                return AppSrcPipelineResult(True, n, None)

    except GstRuntimeUnavailable:
        raise
    except Exception as e:
        logger.exception("run_pipeline_with_appsrc_sync failed")
        return AppSrcPipelineResult(False, 0, str(e))
    finally:
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
