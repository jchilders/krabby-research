"""Map logical ``sensor_id`` values to HAL **H.264** (or other) GStreamer pipeline strings.

Used by **HAL-side** streaming / recording code (not the teleop HTTP server). Both
:class:`~hal.server.jetson.sensor_backend_jetson.JetsonSensorInterface` and
:class:`~hal.server.isaac.sensor_backend_isaac.IsaacSensorInterface` implement
the same :class:`~hal.server.sensor_interface.SensorInterface` API.
"""

from __future__ import annotations

from typing import Any, Iterator

from hal.server.sensor_interface import SensorInfo, SensorInterface


def sensor_by_id(interface: SensorInterface, sensor_id: str) -> SensorInfo:
    """Return the listed :class:`SensorInfo` whose ``id`` matches ``sensor_id``."""
    for s in interface.list_sensors():
        if s.id == sensor_id:
            return s
    raise KeyError(f"sensor_id not in list_sensors(): {sensor_id!r}")


def build_encoded_pipeline_for_sensor_id(
    interface: SensorInterface,
    sensor_id: str,
    *,
    encoding: str = "h264",
    output_element: str = "fakesink",
    **build_kwargs: Any,
) -> str:
    """``get_gstreamer_handle`` → ``build_pipeline`` for one logical sensor.

    Args:
        interface: Jetson or Isaac sensor backend.
        sensor_id: Logical id (e.g. ``front_rgbd``, ``radar_front``).
        encoding: ``h264``, ``h265``, or ``raw`` (passed to ``build_pipeline``).
        output_element: GStreamer sink element name (e.g. ``fakesink``, ``appsink``).
        **build_kwargs: Extra backend args (e.g. ``use_nvenc``, ``bitrate`` on Jetson).

    Returns:
        Pipeline description string suitable for ``Gst.parse_launch``.

    Raises:
        KeyError: Unknown ``sensor_id`` for this interface's ``list_sensors()``.
    """
    sensor = sensor_by_id(interface, sensor_id)
    handle = interface.get_gstreamer_handle(sensor)
    return interface.build_pipeline(
        handle,
        encoding=encoding,
        output_element=output_element,
        **build_kwargs,
    )


def iter_encoded_pipelines_for_sensor_ids(
    interface: SensorInterface,
    sensor_ids: list[str],
    *,
    encoding: str = "h264",
    output_element: str = "fakesink",
    **build_kwargs: Any,
) -> Iterator[tuple[str, str]]:
    """Yield ``(sensor_id, pipeline_string)`` for each id (stable order = input order)."""
    for sid in sensor_ids:
        yield sid, build_encoded_pipeline_for_sensor_id(
            interface,
            sid,
            encoding=encoding,
            output_element=output_element,
            **build_kwargs,
        )
