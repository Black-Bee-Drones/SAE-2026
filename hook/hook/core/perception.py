"""Shared perception helpers for the hook mission.

Image-to-body axis convention (down camera, body FLU):
    image -y (up)    -> body +x (forward)
    image +x (right) -> body -y (right)

Pixels-per-meter is delegated to a :class:`hook.core.camera_scaling.CameraScaling`
singleton selected by the ``CAMERA_SCALING_METHOD`` constant. Three
implementations are available (FOV, EFL, intrinsic-calibrated); see
:mod:`hook.core.camera_scaling` for the contract and trade-offs.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import (
    Segmentation,
    SegmentationInput,
    SegmentationResult,
    Segmentor,
)
from nectar.vision import ImageHandler

from hook.core.camera_scaling import get_camera_scaling
from hook.core.constants import (
    CAMERA_TO_HOOK_BODY_X_M,
    CAMERA_TO_HOOK_BODY_Y_M,
    HOSE_CLASS,
    HOSE_MIN_CONTOUR_AREA,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    SEG_IMGSZ,
    SEG_IOU,
    SEG_PREDICT_CONF,
    SPHERE_ANCHOR_DISTANCE_M,
    SPHERE_CLASS,
    SPHERE_HEIGHT_M,
)

# Sphere is orange/red (mean BGR ≈ 30/100/220, so B/R ≈ 0.15). Blue takeoff
# base is blue (mean BGR ≈ 150/80/40, B/R ≈ 3-4). Any sphere candidate whose
# masked region has B/R above this is the base, not the sphere.
_SPHERE_MAX_BLUE_RED_RATIO = 1.5
_SPHERE_MIN_MASK_PIXELS = 20


def _is_blue_dominant(frame: np.ndarray, mask: Optional[np.ndarray]) -> bool:
    if frame is None or mask is None:
        return False
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) < _SPHERE_MIN_MASK_PIXELS:
        return False
    mean_bgr = cv2.mean(frame, mask=m)
    b, _, r = mean_bgr[0], mean_bgr[1], mean_bgr[2]
    return b > r * _SPHERE_MAX_BLUE_RED_RATIO


def run_seg(
    camera: ImageHandler,
    segmentor: Segmentor,
    class_filter: PerClassConfidenceFilter,
) -> Tuple[Optional[np.ndarray], Optional[SegmentationResult]]:
    """Capture a frame, run segmentation at SEG_IMGSZ, apply per-class filter.

    Predict is run at SEG_PREDICT_CONF (= min per-class threshold) so YOLO
    can prune candidates server-side before NMS / mask decoding. The
    PerClassConfidenceFilter then enforces stricter per-class cutoffs.
    A final colour gate drops any ``sphere`` whose masked region is
    blue-dominant — the takeoff base is the only known false-positive
    geometry the segmentor flips to ``sphere`` on Jetson FP16 engines.
    """
    frame = camera.take_photo()
    if frame is None:
        return None, None

    pred = segmentor._model.predict(
        SegmentationInput(
            image=frame,
            conf_threshold=SEG_PREDICT_CONF,
            iou_threshold=SEG_IOU,
            imgsz=SEG_IMGSZ,
        )
    )
    if pred.detections is None:
        return frame, SegmentationResult()

    filtered = class_filter.filter(pred.detections)
    result = SegmentationResult.from_supervision(
        detections=filtered,
        class_names=segmentor.class_names,
        inference_time=pred.inference_time,
        model_name=getattr(segmentor, "model_source", None),
        image=frame,
    )
    result.segmentations = [
        s for s in result.segmentations
        if s.class_name != SPHERE_CLASS or not _is_blue_dominant(frame, s.mask)
    ]
    return frame, result


def best_sphere(result: Optional[SegmentationResult]) -> Optional[Segmentation]:
    """Highest-confidence sphere instance, or None."""
    if result is None or len(result) == 0:
        return None
    candidates = [s for s in result.segmentations if s.class_name == SPHERE_CLASS]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.confidence)


def hose_segments(result: Optional[SegmentationResult]) -> List[Segmentation]:
    """All hose instances, sorted by confidence desc."""
    if result is None or len(result) == 0:
        return []
    return sorted(
        (s for s in result.segmentations if s.class_name == HOSE_CLASS),
        key=lambda s: s.confidence,
        reverse=True,
    )


def hose_pose(
    seg: Segmentation, min_area: int = HOSE_MIN_CONTOUR_AREA
) -> Optional[Tuple[float, float, float, float, Tuple[float, float]]]:
    """Fit minAreaRect to a hose mask.

    Returns (cx, cy, angle_deg, length_px, axis_unit) where:
        cx, cy: mask centroid (image px).
        angle_deg: signed deviation from horizontal in (-90, 90].
                   0 means hose horizontal in image (drone perpendicular).
        length_px: long-axis length in pixels.
        axis_unit: unit vector along the rope in image frame.
    Returns None if the mask is unusable.
    """
    if seg.mask is None:
        return None
    mask_u8 = (seg.mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    (cx, cy), (w, h), raw_a = cv2.minAreaRect(largest)
    long_side = max(w, h)
    long_axis_deg = raw_a if w >= h else raw_a + 90.0
    if long_axis_deg > 90.0:
        long_axis_deg -= 180.0
    elif long_axis_deg <= -90.0:
        long_axis_deg += 180.0

    rad = math.radians(long_axis_deg)
    return (
        float(cx),
        float(cy),
        float(long_axis_deg),
        float(long_side),
        (math.cos(rad), math.sin(rad)),
    )


def px_per_meter_x(
    altitude_m: Optional[float], target_height_m: float = 0.0
) -> float:
    """Pixels per meter along image-x (== body-y). Delegates to the active
    :class:`CameraScaling`."""
    return get_camera_scaling().ppm_x(altitude_m, target_height_m)


def px_per_meter_y(
    altitude_m: Optional[float], target_height_m: float = 0.0
) -> float:
    """Pixels per meter along image-y (== body-x). Delegates to the active
    :class:`CameraScaling`."""
    return get_camera_scaling().ppm_y(altitude_m, target_height_m)


def image_offset_to_body(
    image_offset: Tuple[float, float],
    altitude_m: Optional[float],
    target_height_m: float = 0.0,
) -> Tuple[float, float]:
    """Deproject ``(image_dx, image_dy)`` from image center to body-frame
    ``(body_x_m, body_y_m)``. Delegates to the active :class:`CameraScaling`,
    so :class:`IntrinsicScaling` removes lens distortion at the corners
    while :class:`FOVScaling` / :class:`EFLScaling` apply the closed-form
    pinhole deprojection.
    """
    dx, dy = image_offset
    return get_camera_scaling().image_offset_to_body(
        dx, dy, altitude_m, target_height_m
    )


def hook_image_offset(
    altitude_m: Optional[float], target_height_m: float = 0.0
) -> Tuple[float, float]:
    """Pixel offset ``(dx, dy)`` from image center to where the hook
    projects onto the depth plane at height ``target_height_m``.
    """
    return get_camera_scaling().body_offset_to_image(
        CAMERA_TO_HOOK_BODY_X_M,
        CAMERA_TO_HOOK_BODY_Y_M,
        altitude_m,
        target_height_m,
    )


def image_bearing_to_body_unit(
    image_bearing_unit: Tuple[float, float],
    altitude_m: Optional[float],
    target_height_m: float = 0.0,
) -> Tuple[float, float]:
    """Convert an image-frame unit-norm bearing ``(ux, uy)`` to the
    body-frame unit direction it represents on the ground.

    The result is yaw-invariant against altitude (the image bearing is
    re-projected to body, then re-normalized) — only the ``ppm_y/ppm_x``
    ratio matters, and that ratio is constant across altitudes for a
    fixed FOV pair. Returns the input unchanged when the deprojected
    vector is degenerate.
    """
    ux, uy = image_bearing_unit
    bx, by = image_offset_to_body((ux, uy), altitude_m, target_height_m)
    norm = math.hypot(bx, by)
    if norm < 1e-9:
        return image_bearing_unit
    return (bx / norm, by / norm)


def approach_setpoint(
    body_bearing_unit: Tuple[float, float],
    target_distance_m: float,
    altitude_m: Optional[float],
    target_height_m: float = 0.0,
) -> Tuple[float, float]:
    """Image-space radial setpoint, anisotropic-correct.

    ``body_bearing_unit`` is the body-frame unit vector from the drone-start
    toward the sphere, captured once at the start of approach (yaw-invariant).
    The sphere should appear at the returned image position when the drone
    has parked ``target_distance_m`` meters short of the sphere along the
    world line drone-start → sphere. The hook (not the camera) is the
    point held at distance ``target_distance_m``.

    Each tick re-evaluates the target with current altitude (and thus
    current ``ppm_x`` / ``ppm_y``).
    """
    bx, by = body_bearing_unit
    ppm_x = px_per_meter_x(altitude_m, target_height_m)
    ppm_y = px_per_meter_y(altitude_m, target_height_m)
    hx, hy = hook_image_offset(altitude_m, target_height_m)
    return (
        IMAGE_CENTER_X - target_distance_m * by * ppm_x + hx,
        IMAGE_CENTER_Y - target_distance_m * bx * ppm_y + hy,
    )


def anchor_sign_for_side(side_image_unit: Tuple[float, float]) -> int:
    """Sign of the sphere's image-x offset from the hook after the drone
    yaws to the closest perpendicular and parks at SPHERE_ANCHOR_DISTANCE_M
    from the sphere along the chosen hose direction.

    After yaw convergence the side direction in image becomes
    (sign(ux), 0). The sphere sits at -side direction from the hook, so
    its image-x offset is `-sign(ux) * D * ppm`. Tie at `ux == 0`
    defaults to +1.
    """
    return -1 if side_image_unit[0] > 0 else 1


def alignment_targets(
    side_image_unit: Tuple[float, float],
    altitude_m: Optional[float],
    standoff_m: float = 0.0,
    target_height_m: float = SPHERE_HEIGHT_M,
) -> Tuple[float, float, int]:
    """Image-frame setpoints for the dual-anchor controller.

    Returns ``(target_sphere_cx, target_hose_cy, anchor_sign)``.

    - ``target_sphere_cx``: image-x of the sphere centroid when the
      hook is parked ``SPHERE_ANCHOR_DISTANCE_M`` from the sphere along
      the chosen hose direction. Image-x ↔ body-y, so this uses
      ``ppm_x``.
    - ``target_hose_cy``: image-y of the hose centroid. With
      ``standoff_m == 0`` the hook is directly over the rope
      (descend phase). With ``standoff_m > 0`` the rope is held that
      many meters in front of the hook in body frame (image upper
      half). Image-y ↔ body-x, so this uses ``ppm_y``.
    - ``anchor_sign``: +1 if the sphere should appear right of the
      hook in image, -1 if left.
    """
    ppm_x = px_per_meter_x(altitude_m, target_height_m)
    ppm_y = px_per_meter_y(altitude_m, target_height_m)
    hook_dx, hook_dy = hook_image_offset(altitude_m, target_height_m)
    sign = anchor_sign_for_side(side_image_unit)
    target_sphere_cx = IMAGE_CENTER_X + hook_dx + sign * SPHERE_ANCHOR_DISTANCE_M * ppm_x
    target_hose_cy = IMAGE_CENTER_Y + hook_dy - standoff_m * ppm_y
    return target_sphere_cx, target_hose_cy, sign


def pick_hose_by_dir(
    sphere: Optional[Segmentation],
    hoses: List[Segmentation],
    side_image_unit: Tuple[float, float],
) -> Optional[Segmentation]:
    """Return the hose whose centroid offset from the sphere best aligns
    with the chosen image-frame unit vector (max dot product)."""
    if sphere is None or not hoses:
        return None
    sx, sy = sphere.center
    best, best_score = None, -math.inf
    for h in hoses:
        hx, hy = h.center
        dx, dy = hx - sx, hy - sy
        norm = math.hypot(dx, dy)
        if norm < 1e-3:
            continue
        score = (dx * side_image_unit[0] + dy * side_image_unit[1]) / norm
        if score > best_score:
            best_score = score
            best = h
    return best


def predict_orient_yaw(
    side_unit: Tuple[float, float],
    sphere_xy: Tuple[float, float],
    ppm_x: float,
    ppm_y: float,
    image_center: Tuple[float, float] = (IMAGE_CENTER_X, IMAGE_CENTER_Y),
) -> float:
    """Closest **body-frame** yaw rotation that makes the chosen rope
    perpendicular to body +x AND keeps the sphere in front of the drone.

    Vision-only, anisotropic-correct. Both ``ppm_x`` and ``ppm_y`` (px/m)
    are required so the image-frame inputs can be deprojected to body
    frame; only their ratio matters (ppm scales out), but passing the
    actual values keeps the API uniform with the rest of the perception
    module. The result is the body-yaw delta in radians, in
    ``(-π, π]`` — a positive value means the drone should yaw CCW.

    Algorithm (once both inputs are deprojected to body):

    1. Rope direction in body: ``rb = (-uy/ppm_y, -ux/ppm_x)``.
    2. Sphere body-frame coords: ``bs = (-(sy-cy)/ppm_y, -(sx-cx)/ppm_x)``.
    3. Two body yaws make the rope perpendicular to body +x:
       ``Δa = atan2(-rb_x, rb_y)`` and ``Δb = Δa + π``.
    4. Pick the one that puts the sphere in front of the drone (body +x
       positive after rotation).

    Reduces to the old image-rotation formula when ``ppm_x == ppm_y``.
    """
    ux, uy = side_unit
    sx, sy = sphere_xy
    cx, cy = image_center
    if ppm_x <= 0.0 or ppm_y <= 0.0:
        return 0.0

    bs_x = -(sy - cy) / ppm_y
    bs_y = -(sx - cx) / ppm_x
    rb_x = -uy / ppm_y
    rb_y = -ux / ppm_x

    delta_a = math.atan2(-rb_x, rb_y)
    new_bs_x = math.cos(delta_a) * bs_x + math.sin(delta_a) * bs_y
    delta = delta_a if new_bs_x > 0.0 else delta_a + math.pi
    return math.atan2(math.sin(delta), math.cos(delta))


def sphere_body_angle(
    sphere_xy: Tuple[float, float],
    ppm_x: float,
    ppm_y: float,
    image_center: Tuple[float, float] = (IMAGE_CENTER_X, IMAGE_CENTER_Y),
) -> float:
    """Body-frame polar angle of the sphere as seen from the drone:
    ``atan2(body_y, body_x)`` where ``body_x`` is forward and ``body_y``
    is left, derived from the sphere's image position by anisotropic
    deprojection.

    Used as the PID feedback during the yaw spin in :class:`OrientToHook`
    because the change in this angle equals minus the body-yaw delta (a
    pure rotation of the body frame around the camera nadir): if the
    drone yaws CCW by Δ, the sphere's body-frame polar angle decreases
    by Δ. That makes the loop's transient response invariant to the
    anisotropic ppm — which the image-frame polar angle is **not**.
    """
    sx, sy = sphere_xy
    cx, cy = image_center
    if ppm_x <= 0.0 or ppm_y <= 0.0:
        return 0.0
    bs_x = -(sy - cy) / ppm_y
    bs_y = -(sx - cx) / ppm_x
    return math.atan2(bs_y, bs_x)


def rotate_image_vector_under_body_yaw(
    image_vec: Tuple[float, float],
    delta_body: float,
    ppm_x: float,
    ppm_y: float,
) -> Tuple[float, float]:
    """Image-frame transformation of a unit direction caused by a body
    yaw rotation under anisotropic pixel scaling.

    Derivation: body yaw by ``delta_body`` (CCW) rotates body coordinates
    of fixed world points by ``-delta_body`` in body frame. Re-projecting
    through the anisotropic pinhole ``image_x = -body_y·ppm_x``,
    ``image_y = -body_x·ppm_y`` yields

    .. code-block:: text

        new_ux = cos(Δ)·ux - sin(Δ)·(ppm_x/ppm_y)·uy
        new_uy = sin(Δ)·(ppm_y/ppm_x)·ux + cos(Δ)·uy

    which is **not** a pure rotation when ``ppm_x != ppm_y``. The output
    is renormalized so consumers (e.g. ``pick_hose_by_dir``) can compare
    it against image-frame unit vectors directly.
    """
    if ppm_x <= 0.0 or ppm_y <= 0.0:
        return image_vec
    ux, uy = image_vec
    cos_d, sin_d = math.cos(delta_body), math.sin(delta_body)
    new_ux = cos_d * ux - sin_d * (ppm_x / ppm_y) * uy
    new_uy = sin_d * (ppm_y / ppm_x) * ux + cos_d * uy
    norm = math.hypot(new_ux, new_uy)
    if norm < 1e-9:
        return image_vec
    return (new_ux / norm, new_uy / norm)
