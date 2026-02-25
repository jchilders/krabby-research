"""HAL server implementation for Jetson."""
# Lazy import so that importing robot_definition_* (e.g. from controller) does not
# pull in hal_server or compute.parkour; only code that needs JetsonHalServer pays the cost.

__all__ = ["JetsonHalServer"]


def __getattr__(name: str):
    if name == "JetsonHalServer":
        from hal.server.jetson.hal_server import JetsonHalServer
        return JetsonHalServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
