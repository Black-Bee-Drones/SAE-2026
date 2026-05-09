"""Shared perception helpers for the hook mission.

Image-to-body axis convention (down camera, body FLU):
    image -y (up)    -> body +x (forward)
    image +x (right) -> body -y (right)
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

from hook.core.constants import (
    CAMERA_TO_HOOK_BODY_X_M,
    CAMERA_TO_HOOK_BODY_Y_M,
    HOSE_CLASS,
    HORIZONTAL_FOV_DEG,
    HOSE_MIN_CONTOUR_AREA,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    IMAGE_WIDTH,
    SEG_IMGSZ,
    SEG_IOU,
    SEG_PREDICT_CONF,
    SPHERE_ANCHOR_DISTANCE_M,
    SPHERE_CLASS,
    SPHERE_HEIGHT_M,
)


def run_seg(
    camera: ImageHandler,
    segmentor: Segmentor,
    class_filter: PerClassConfidenceFilter,
) -> Tuple[Optional[np.ndarray], Optional[SegmentationResult]]:
    """Capture a frame, run segmentation at SEG_IMGSZ, apply per-class filter.

    Predict is run at SEG_PREDICT_CONF (= min per-class threshold) so YOLO
    can prune candidates server-side before NMS / mask decoding. The
    PerClassConfidenceFilter then enforces stricter per-class cutoffs.
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


def px_per_meter(altitude_m: Optional[float], target_height_m: float = 0.0) -> float:
    """Pixel-per-meter at the depth `altitude_m - target_height_m`.

    The down camera sees a target at height `target_height_m` above ground
    as if the camera were at altitude `altitude_m - target_height_m` looking
    at the ground. Default 0 keeps the legacy ground-projection behaviour.
    """
    if altitude_m is None:
        return 0.0
    depth = altitude_m - target_height_m
    if depth <= 0.0:
        return 0.0
    half_fov_rad = math.radians(HORIZONTAL_FOV_DEG / 2.0)
    return IMAGE_WIDTH / (2.0 * depth * math.tan(half_fov_rad))


def hook_image_offset(
    altitude_m: Optional[float], target_height_m: float = 0.0
) -> Tuple[float, float]:
    """Pixel offset (dx, dy) from image center to where the hook projects
    onto the depth plane at height `target_height_m` above ground.
    """
    rate = px_per_meter(altitude_m, target_height_m)
    return (-CAMERA_TO_HOOK_BODY_Y_M * rate, -CAMERA_TO_HOOK_BODY_X_M * rate)


def approach_setpoint(
    bearing_unit: Tuple[float, float],
    target_px: float,
    altitude_m: Optional[float],
    target_height_m: float = 0.0,
) -> Tuple[float, float]:
    """Image-space RADIAL setpoint along the captured bearing.

    Sphere should appear at `image_center + bearing*target_px + hook_offset`,
    so the drone parks D meters short on the line drone-start -> sphere
    (yaw-invariant). Hook offset uses the same depth plane as the bearing
    to keep the HOOK (not the camera) at distance D from the sphere.
    """
    ux0, uy0 = bearing_unit
    hx, hy = hook_image_offset(altitude_m, target_height_m)
    return (
        IMAGE_CENTER_X + ux0 * target_px + hx,
        IMAGE_CENTER_Y + uy0 * target_px + hy,
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

    - ``target_sphere_cx``: where the sphere centroid should sit when the
      hook is parked SPHERE_ANCHOR_DISTANCE_M from the sphere along the
      chosen hose direction.
    - ``target_hose_cy``: where the hose centroid should sit. With
      ``standoff_m=0`` the hook is directly over the rope (DESCEND).
      With ``standoff_m>0`` the rope is held that many meters in front
      of the hook in body frame (rope appears in the upper half of the
      image), keeping the lidar (mounted aft of the hook) clear of the
      rope during ALIGN.
    - ``anchor_sign``: +1 if the sphere should appear right of the hook
      in image, -1 if left.
    """
    ppm = px_per_meter(altitude_m, target_height_m)
    hook_dx, hook_dy = hook_image_offset(altitude_m, target_height_m)
    sign = anchor_sign_for_side(side_image_unit)
    target_sphere_cx = IMAGE_CENTER_X + hook_dx + sign * SPHERE_ANCHOR_DISTANCE_M * ppm
    target_hose_cy = IMAGE_CENTER_Y + hook_dy - standoff_m * ppm
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
    image_center: Tuple[float, float] = (IMAGE_CENTER_X, IMAGE_CENTER_Y),
) -> float:
    """Closest body-yaw rotation that makes the chosen rope horizontal in
    image AND keeps the sphere in the upper half (rope in front of drone).

    Image-frame rotation by Δ (positive = body CCW yaw) maps a point
    ``(x, y)`` to ``(cosΔ·x − sinΔ·y, sinΔ·x + cosΔ·y)``. Two yaw
    rotations make the rope horizontal:

    - ``Δ_a = -atan2(uy, ux)`` (rope direction → +image-x)
    - ``Δ_b = wrap_pi(Δ_a + π)`` (rope direction → -image-x)

    The rope passes through the sphere, so it ends up in the image upper
    half iff ``new_sy = sin(Δ)·dx + cos(Δ)·dy < 0``. That sign
    discriminates which Δ to use. Result is in ``(-π, π]``.

    Scale-invariant: depends only on the sphere's image position relative
    to image center, and the rope's image direction. No ``ppm``, no
    altitude. Vision-only.
    """
    ux, uy = side_unit
    sx, sy = sphere_xy
    cx, cy = image_center
    dx, dy = sx - cx, sy - cy
    delta_a = math.atan2(-uy, ux)
    disc = math.sin(delta_a) * dx + math.cos(delta_a) * dy
    delta = delta_a if disc < 0 else delta_a + math.pi
    return math.atan2(math.sin(delta), math.cos(delta))
