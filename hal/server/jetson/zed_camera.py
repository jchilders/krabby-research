"""ZED camera integration for Jetson robot deployment.

This module provides a wrapper for the ZED camera SDK to capture RGB and depth (meters).
ZedCamera implements the RgbDepthCamera protocol (see rgb_depth_camera.py).
Policy scan vectors are built in JetsonHalServer from the same depth frames.

Production code: Requires ZED SDK (pyzed) and hardware to be available.
Fails fast if dependencies are missing.
"""

import logging
import time
from typing import Optional

import numpy as np

from hal.server.jetson.rgb_depth_camera import RgbDepthCamera

logger = logging.getLogger(__name__)


class ZedCamera(RgbDepthCamera):
    """ZED camera wrapper implementing RgbDepthCamera.

    Handles ZED SDK initialization and frame capture.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        depth_mode: str = "PERFORMANCE",
        serial_number: Optional[int] = None,
    ):
        """Initialize ZED camera.

        Args:
            resolution: Camera resolution (width, height). Default (640, 480)
            fps: Frames per second. Default 30
            depth_mode: Depth mode ("PERFORMANCE", "QUALITY", "ULTRA"). Default "PERFORMANCE"
            serial_number: If set, open this USB ZED (Stereolabs serial). Default opens first device.

        Raises:
            RuntimeError: If camera initialization fails
        """
        self.resolution = resolution
        self.fps = fps
        self.depth_mode = depth_mode
        self.serial_number = serial_number

        self.camera = None
        self.initialized = False
        self.last_frame_time_ns = 0
        self.frame_period_ns = 1_000_000_000 // fps  # Nanoseconds per frame

        # Pre-allocated buffers to avoid allocation in hot path
        self.depth_image = None
        # Last captured frame (one grab fills both; used by get_rgb_image/get_depth_map)
        self._last_rgb: Optional[np.ndarray] = None
        self._last_depth_np: Optional[np.ndarray] = None

        # Initialize camera
        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """Initialize ZED SDK and open camera.

        Raises:
            RuntimeError: If camera initialization fails or ZED SDK is not available
        """
        # Import ZED SDK - required for production
        try:
            import pyzed.sl as sl
            self._zed_module = sl
        except ImportError as e:
            raise RuntimeError(
                "ZED SDK (pyzed) not available. "
                "Install pyzed and ensure ZED SDK is installed on the system."
            ) from e

        try:
            # Create camera object
            self.camera = self._zed_module.Camera()

            # Create init parameters
            init_params = self._zed_module.InitParameters()
            init_params.camera_resolution = self._zed_module.RESOLUTION.VGA  # 640x480
            if self.resolution == (1280, 720):
                init_params.camera_resolution = self._zed_module.RESOLUTION.HD720
            elif self.resolution == (1920, 1080):
                init_params.camera_resolution = self._zed_module.RESOLUTION.HD1080

            init_params.camera_fps = self.fps
            depth_mode_map = {
                "PERFORMANCE": self._zed_module.DEPTH_MODE.PERFORMANCE,
                "QUALITY": self._zed_module.DEPTH_MODE.QUALITY,
                "ULTRA": self._zed_module.DEPTH_MODE.ULTRA,
            }
            init_params.depth_mode = depth_mode_map.get(
                self.depth_mode, self._zed_module.DEPTH_MODE.PERFORMANCE
            )
            init_params.coordinate_units = self._zed_module.UNIT.METER
            init_params.coordinate_system = self._zed_module.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

            if self.serial_number is not None:
                sn = int(self.serial_number)
                set_serial = getattr(init_params, "set_from_serial_number", None)
                if callable(set_serial):
                    rc = set_serial(sn)
                    if rc is not None and rc != self._zed_module.ERROR_CODE.SUCCESS:
                        raise RuntimeError(f"ZED set_from_serial_number({sn}) failed: {rc}")
                elif hasattr(init_params, "serial_number"):
                    init_params.serial_number = sn
                elif hasattr(init_params, "camera_serial_number"):
                    init_params.camera_serial_number = sn
                else:
                    raise RuntimeError(
                        "pyzed InitParameters has no set_from_serial_number / serial_number; "
                        "cannot select ZED by serial."
                    )

            # Open camera
            status = self.camera.open(init_params)
            if status != self._zed_module.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Failed to open ZED camera: {status}")

            # Get camera information
            camera_info = self.camera.get_camera_information()
            logger.info(f"ZED camera initialized: {camera_info.camera_model}")
            logger.info(f"Resolution: {self.resolution}, FPS: {self.fps}")

            # Create depth and RGB image mats
            self.depth_image = self._zed_module.Mat()
            self._rgb_image = self._zed_module.Mat()

            self.initialized = True
            logger.info("ZED camera initialized successfully")

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"ZED camera initialization failed: {e}") from e

    def close(self) -> None:
        """Close camera and release resources."""
        if self.camera is not None:
            try:
                self.camera.close()
                logger.info("ZED camera closed")
            except Exception as e:
                logger.error(f"Error closing ZED camera: {e}")
            finally:
                self.camera = None
                self.initialized = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def _grab_frame(self) -> bool:
        """Grab one frame and fill _last_rgb and _last_depth_np. Returns True on success.

        Expected failure (no frame available, USB glitch, etc.) is reported by the SDK
        via grab() return value; we return False and callers get None. We do not catch
        exceptions: any raise from the SDK or numpy is a bug or serious error and
        should propagate.
        """
        if not self.initialized:
            return False
        if self.camera.grab() != self._zed_module.ERROR_CODE.SUCCESS:
            return False
        # Retrieve depth
        self.camera.retrieve_measure(
            self.depth_image, self._zed_module.MEASURE.DEPTH, self._zed_module.MEM.CPU
        )
        depth_array = self.depth_image.get_data()
        self._last_depth_np = np.asarray(depth_array, dtype=np.float32).copy()
        # Retrieve left RGB
        self.camera.retrieve_image(
            self._rgb_image, self._zed_module.VIEW.LEFT, self._zed_module.MEM.CPU
        )
        rgb_data = self._rgb_image.get_data()
        # ZED returns BGRA (4 channels); convert to RGB uint8 (H, W, 3)
        rgb_np = np.asarray(rgb_data)
        if rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[:, :, :3]  # drop alpha
        elif rgb_np.shape[-1] != 3:
            rgb_np = rgb_np[:, :, :3]
        self._last_rgb = np.ascontiguousarray(rgb_np.astype(np.uint8))
        self.last_frame_time_ns = time.time_ns()
        return True

    def get_camera_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture one frame and return (RGB, depth) for HAL observations.

        One grab retrieves both left RGB and depth. Use this when populating
        camera_rgb and camera_depth to avoid double capture.

        Returns:
            (rgb, depth): rgb (H, W, 3) uint8, depth (H, W) float32 in meters; either may be None on failure.
        """
        if not self._grab_frame():
            return None, None
        return self._last_rgb, self._last_depth_np

    def get_rgb_image(self) -> Optional[np.ndarray]:
        """Capture and return left RGB image (H, W, 3) uint8. Performs one grab."""
        if not self._grab_frame():
            return None
        return self._last_rgb

    def get_depth_map(self) -> Optional[np.ndarray]:
        """Return the last captured depth map (H, W) float32 in meters.

        No grab is performed; use get_camera_frames() or get_rgb_image() first to capture a frame.
        """
        return self._last_depth_np

    def is_ready(self) -> bool:
        """Check if camera is ready to capture frames.

        Returns:
            True if camera is initialized and ready
        """
        return self.initialized

    def get_frame_rate(self) -> float:
        """Get actual frame rate.

        Returns:
            Frame rate in Hz
        """
        if self.last_frame_time_ns == 0:
            return 0.0

        # This is a simplified version - could track multiple frame times for better accuracy
        return self.fps  # Return configured FPS for now


def create_zed_camera(
    resolution: tuple[int, int] = (640, 480),
    fps: int = 30,
    depth_mode: str = "PERFORMANCE",
    serial_number: Optional[int] = None,
) -> Optional[ZedCamera]:
    """Factory function to create ZED camera with error handling.

    Args:
        resolution: Camera resolution (width, height)
        fps: Frames per second
        depth_mode: Depth mode
        serial_number: Optional USB serial to select a specific ZED when multiple are connected.

    Returns:
        ZedCamera instance if successful, None if initialization fails
    """
    try:
        camera = ZedCamera(
            resolution=resolution,
            fps=fps,
            depth_mode=depth_mode,
            serial_number=serial_number,
        )
        return camera
    except RuntimeError as e:
        logger.error(f"Failed to create ZED camera: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating ZED camera: {e}", exc_info=True)
        return None
