"""Camera scaling strategy: pixels-per-meter and image<->body deprojection.

Three swappable implementations, selected at module load time by the
``CAMERA_SCALING_METHOD`` constant:

- :class:`FOVScaling` — derives ``ppm_x`` from ``HORIZONTAL_FOV_DEG`` +
  ``IMAGE_WIDTH`` and ``ppm_y`` from ``VERTICAL_FOV_DEG`` + ``IMAGE_HEIGHT``.
  Anisotropic. Today's behavior — works without calibration.
- :class:`EFLScaling` — derives a single isotropic ``f_px`` from the lens's
  effective focal length and the sensor pixel pitch
  (``f_px = EFL_mm · 1000 / pixel_um``). Same value on both axes.
- :class:`IntrinsicScaling` — uses a calibrated ``K`` matrix and distortion
  vector. Replace the placeholder values in :mod:`hook.core.constants` with
  ``cv2.calibrateCamera`` output to get the correct lens model. Overrides
  :meth:`image_offset_to_body` to deproject through the distortion model
  with ``cv2.undistortPoints``, so off-axis pixels are correct too.

All three expose the same interface so the rest of the perception module
delegates through :func:`get_camera_scaling` and :func:`make_all_camera_scalings`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

from hook.core.constants import (
    CAMERA_EFL_MM,
    CAMERA_INTRINSIC_DIST,
    CAMERA_INTRINSIC_K,
    CAMERA_PIXEL_SIZE_UM,
    CAMERA_SCALING_METHOD,
    HORIZONTAL_FOV_DEG,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    VERTICAL_FOV_DEG,
)


def _depth(altitude_m: Optional[float], target_height_m: float) -> float:
    if altitude_m is None:
        return 0.0
    d = altitude_m - target_height_m
    return d if d > 0.0 else 0.0


class CameraScaling(ABC):
    """Strategy for converting image pixels <-> body-frame meters.

    The down-camera convention is ``image -y -> body +x`` (forward) and
    ``image +x -> body -y`` (right). All depths are taken to a target plane
    at ``target_height_m`` above the ground, so ``depth = altitude - target_height_m``.
    """

    name: str = "camera_scaling"

    @abstractmethod
    def ppm_x(
        self, altitude_m: Optional[float], target_height_m: float = 0.0
    ) -> float:
        """Pixels per meter along image-x (== body-y). Zero if depth invalid."""

    @abstractmethod
    def ppm_y(
        self, altitude_m: Optional[float], target_height_m: float = 0.0
    ) -> float:
        """Pixels per meter along image-y (== body-x). Zero if depth invalid."""

    def image_offset_to_body(
        self,
        image_dx: float,
        image_dy: float,
        altitude_m: Optional[float],
        target_height_m: float = 0.0,
    ) -> Tuple[float, float]:
        ppm_x = self.ppm_x(altitude_m, target_height_m)
        ppm_y = self.ppm_y(altitude_m, target_height_m)
        if ppm_x <= 0.0 or ppm_y <= 0.0:
            return (0.0, 0.0)
        return (-image_dy / ppm_y, -image_dx / ppm_x)

    def body_offset_to_image(
        self,
        body_x_m: float,
        body_y_m: float,
        altitude_m: Optional[float],
        target_height_m: float = 0.0,
    ) -> Tuple[float, float]:
        ppm_x = self.ppm_x(altitude_m, target_height_m)
        ppm_y = self.ppm_y(altitude_m, target_height_m)
        return (-body_y_m * ppm_x, -body_x_m * ppm_y)


class FOVScaling(CameraScaling):
    name = "fov"

    def __init__(
        self,
        hfov_deg: float = HORIZONTAL_FOV_DEG,
        vfov_deg: float = VERTICAL_FOV_DEG,
        image_width: int = IMAGE_WIDTH,
        image_height: int = IMAGE_HEIGHT,
    ) -> None:
        self._half_hfov_tan = math.tan(math.radians(hfov_deg / 2.0))
        self._half_vfov_tan = math.tan(math.radians(vfov_deg / 2.0))
        self._image_width = image_width
        self._image_height = image_height

    def ppm_x(self, altitude_m, target_height_m=0.0):
        depth = _depth(altitude_m, target_height_m)
        if depth <= 0.0 or self._half_hfov_tan <= 0.0:
            return 0.0
        return self._image_width / (2.0 * depth * self._half_hfov_tan)

    def ppm_y(self, altitude_m, target_height_m=0.0):
        depth = _depth(altitude_m, target_height_m)
        if depth <= 0.0 or self._half_vfov_tan <= 0.0:
            return 0.0
        return self._image_height / (2.0 * depth * self._half_vfov_tan)


class EFLScaling(CameraScaling):
    name = "efl"

    def __init__(
        self,
        efl_mm: float = CAMERA_EFL_MM,
        pixel_size_um: float = CAMERA_PIXEL_SIZE_UM,
    ) -> None:
        if pixel_size_um <= 0.0:
            raise ValueError("pixel_size_um must be > 0")
        self._f_px = efl_mm * 1000.0 / pixel_size_um

    def ppm_x(self, altitude_m, target_height_m=0.0):
        depth = _depth(altitude_m, target_height_m)
        if depth <= 0.0:
            return 0.0
        return self._f_px / depth

    def ppm_y(self, altitude_m, target_height_m=0.0):
        return self.ppm_x(altitude_m, target_height_m)


class IntrinsicScaling(CameraScaling):
    """Calibrated pinhole + radial/tangential distortion.

    Uses ``CAMERA_INTRINSIC_K`` and ``CAMERA_INTRINSIC_DIST`` from
    :mod:`hook.core.constants`. Both ``image_offset_to_body`` and
    ``body_offset_to_image`` use the calibrated principal point and
    distortion model internally, while their inputs/outputs remain
    expressed as offsets from the *geometric* image center
    ``(IMAGE_CENTER_X, IMAGE_CENTER_Y)`` -- matching the convention used
    by :class:`FOVScaling`, :class:`EFLScaling`, and every mission state.
    The two methods are mutual inverses (to numerical precision), so the
    closed-loop ``target_pixel = IMAGE_CENTER + body_offset_to_image(...)``
    / ``body_error = image_offset_to_body(measured_pixel - IMAGE_CENTER, ...)``
    pipeline used across :mod:`hook.states` converges on the intended
    body setpoint without coordinate-frame drift.
    """

    name = "intrinsic"

    def __init__(
        self,
        K: np.ndarray = CAMERA_INTRINSIC_K,
        dist: np.ndarray = CAMERA_INTRINSIC_DIST,
    ) -> None:
        K = np.asarray(K, dtype=np.float64)
        dist = np.asarray(dist, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {K.shape}")
        self._K = K
        self._dist = dist
        self._f_x = float(K[0, 0])
        self._f_y = float(K[1, 1])
        self._c_x = float(K[0, 2])
        self._c_y = float(K[1, 2])
        self._rvec = np.zeros(3, dtype=np.float64)
        self._tvec = np.zeros(3, dtype=np.float64)

    def ppm_x(self, altitude_m, target_height_m=0.0):
        depth = _depth(altitude_m, target_height_m)
        if depth <= 0.0:
            return 0.0
        return self._f_x / depth

    def ppm_y(self, altitude_m, target_height_m=0.0):
        depth = _depth(altitude_m, target_height_m)
        if depth <= 0.0:
            return 0.0
        return self._f_y / depth

    def image_offset_to_body(
        self, image_dx, image_dy, altitude_m, target_height_m=0.0
    ):
        depth = _depth(altitude_m, target_height_m)
        if depth <= 0.0:
            return (0.0, 0.0)
        abs_pixel = np.array(
            [[[IMAGE_CENTER_X + image_dx, IMAGE_CENTER_Y + image_dy]]],
            dtype=np.float64,
        )
        normalized = cv2.undistortPoints(abs_pixel, self._K, self._dist).reshape(2)
        x_norm, y_norm = float(normalized[0]), float(normalized[1])
        return (-y_norm * depth, -x_norm * depth)

    def body_offset_to_image(
        self, body_x_m, body_y_m, altitude_m, target_height_m=0.0
    ):
        depth = _depth(altitude_m, target_height_m)
        if depth <= 0.0:
            return (0.0, 0.0)
        obj_pt = np.array(
            [[[-body_y_m, -body_x_m, depth]]], dtype=np.float64
        )
        img_pts, _ = cv2.projectPoints(
            obj_pt, self._rvec, self._tvec, self._K, self._dist
        )
        abs_pix = img_pts.reshape(2)
        return (float(abs_pix[0]) - IMAGE_CENTER_X, float(abs_pix[1]) - IMAGE_CENTER_Y)


_FACTORY = {
    "fov": FOVScaling,
    "efl": EFLScaling,
    "intrinsic": IntrinsicScaling,
}


def make_camera_scaling(method: str) -> CameraScaling:
    try:
        cls = _FACTORY[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CAMERA_SCALING_METHOD={method!r}; "
            f"valid options: {sorted(_FACTORY)}"
        ) from exc
    return cls()


def make_all_camera_scalings() -> List[CameraScaling]:
    return [FOVScaling(), EFLScaling(), IntrinsicScaling()]


_active: Optional[CameraScaling] = None


def get_camera_scaling() -> CameraScaling:
    global _active
    if _active is None:
        _active = make_camera_scaling(CAMERA_SCALING_METHOD)
    return _active


def reset_camera_scaling() -> None:
    """Drop the cached singleton. Used by tests."""
    global _active
    _active = None


def image_center_for(scaling: CameraScaling) -> Tuple[float, float]:
    """Reference center used by ``scaling`` when consuming/producing image deltas.

    All three implementations (FOV / EFL / Intrinsic) interpret their image
    deltas as offsets from the *geometric* image center
    ``(IMAGE_CENTER_X, IMAGE_CENTER_Y)``. :class:`IntrinsicScaling` shifts to
    the calibrated principal point ``(c_x, c_y)`` internally via
    ``cv2.undistortPoints`` / ``cv2.projectPoints``; callers do not (and
    should not) compensate for that themselves.
    """
    del scaling
    return (float(IMAGE_CENTER_X), float(IMAGE_CENTER_Y))
