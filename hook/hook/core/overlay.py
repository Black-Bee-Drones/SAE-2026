"""Shared visualization helpers for the hook mission states.

Two layers per saved frame:

1. ``annotate_seg(frame, result)`` runs the same supervision-based pipeline
   that the ``nectar`` ``sequence_inference.py`` example uses — a custom
   high-contrast palette, mask fill, polygon outline, and label chip with
   black text. Replaces the SDK's :meth:`Segmentor.draw_segmentations`,
   which uses the supervision default palette and skips the polygon outline
   (the masks otherwise blend into the rope's own red).

2. ``draw_*`` helpers add the per-state controller geometry (image center,
   hook projection, target sphere, target hose row, error arrows, tolerance
   bands) and a pair of small HUDs with the live errors and commanded
   velocities. APPROACH keeps its own bespoke overlay because the radial
   bearing math is unique to that state.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

# ----------------------------------------------------------------------
# Visual style constants
# ----------------------------------------------------------------------

# High-contrast palette: visible against orange sphere + red rose mask. Order
# is fixed across runs so each class always renders in the same color.
PALETTE = sv.ColorPalette(
    [
        sv.Color(0, 255, 255),  # cyan
        sv.Color(255, 0, 255),  # magenta
        sv.Color(0, 255, 0),  # lime
        sv.Color(255, 255, 0),  # yellow
    ]
)
MASK_OPACITY = 0.30
OUTLINE_THICKNESS = 3
LABEL_TEXT_SCALE = 0.6
LABEL_TEXT_THICKNESS = 2

# Drawing colors (BGR for OpenCV).
COLOR_CENTER = (255, 255, 0)  # cyan: image center
COLOR_HOOK = (255, 0, 255)  # magenta: hook image projection
COLOR_HOSE_AXIS = (0, 255, 255)  # yellow: detected hose axis line
COLOR_TARGET_HOSE = (0, 255, 0)  # green: target hose row + tolerance band
COLOR_TARGET_SPHERE = (0, 255, 255)  # yellow: target sphere ring
COLOR_TARGET_SPHERE_LOST = (0, 0, 255)  # red: target sphere when out of FOV
COLOR_SPHERE = (0, 0, 255)  # red: detected sphere centroid
COLOR_ERR_VX = (255, 255, 0)  # cyan: vx (hose-row) error arrow
COLOR_ERR_VY = (0, 255, 0)  # green: vy (sphere) error arrow
COLOR_ALT_BAR = (0, 255, 0)
COLOR_ALT_BAR_RELEASE = (0, 0, 255)
COLOR_HUD_TEXT = (0, 255, 0)
COLOR_HUD_BG = (0, 0, 0)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------------------------------------------------------
# Segmentation annotation
# ----------------------------------------------------------------------


def annotate_seg(frame: np.ndarray, result) -> np.ndarray:
    """Mask fill + polygon outline + label chip, using PALETTE. Returns a
    new image; ``frame`` is left untouched. Empty results return a copy.
    """
    if frame is None:
        return frame
    if not result or not getattr(result, "segmentations", None):
        return frame.copy()

    detections = result.to_supervision()
    out = frame.copy()
    out = sv.MaskAnnotator(color=PALETTE, opacity=MASK_OPACITY).annotate(
        scene=out, detections=detections
    )
    out = sv.PolygonAnnotator(color=PALETTE, thickness=OUTLINE_THICKNESS).annotate(
        scene=out, detections=detections
    )
    labels = [f"{s.class_name} {s.confidence:.2f}" for s in result.segmentations]
    if any(labels):
        out = sv.LabelAnnotator(
            color=PALETTE,
            text_color=sv.Color.BLACK,
            text_scale=LABEL_TEXT_SCALE,
            text_thickness=LABEL_TEXT_THICKNESS,
        ).annotate(scene=out, detections=detections, labels=labels)
    return out


# ----------------------------------------------------------------------
# Primitive drawing helpers
# ----------------------------------------------------------------------


def draw_dashed_line(
    img: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash: int = 14,
    gap: int = 10,
) -> None:
    """Draw a dashed segment from p1 to p2 in image space."""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1.0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    step = dash + gap
    n = int(length // step) + 1
    for i in range(n):
        s = i * step
        e = min(s + dash, length)
        a = (int(x1 + dx * s), int(y1 + dy * s))
        b = (int(x1 + dx * e), int(y1 + dy * e))
        cv2.line(img, a, b, color, thickness)


def draw_hud(
    img: np.ndarray,
    lines: Iterable[str],
    anchor: str = "tl",
    *,
    text_scale: float = 0.6,
    text_thickness: int = 2,
    line_height: int = 26,
    pad: int = 10,
) -> None:
    """Black filled rectangle + green text. anchor in {tl, tr, bl, br}."""
    lines = [str(line) for line in lines if line]
    if not lines:
        return
    sizes = [
        cv2.getTextSize(line, _FONT, text_scale, text_thickness)[0] for line in lines
    ]
    w_max = max(s[0] for s in sizes) + 2 * pad
    h_total = line_height * len(lines) + 2 * pad

    H, W = img.shape[:2]
    margin = 8
    if anchor == "tl":
        x0, y0 = margin, margin
    elif anchor == "tr":
        x0, y0 = W - w_max - margin, margin
    elif anchor == "bl":
        x0, y0 = margin, H - h_total - margin
    elif anchor == "br":
        x0, y0 = W - w_max - margin, H - h_total - margin
    else:
        x0, y0 = margin, margin

    cv2.rectangle(img, (x0, y0), (x0 + w_max, y0 + h_total), COLOR_HUD_BG, -1)
    for i, line in enumerate(lines):
        y = y0 + pad + (i + 1) * line_height - 8
        cv2.putText(
            img,
            line,
            (x0 + pad, y),
            _FONT,
            text_scale,
            COLOR_HUD_TEXT,
            text_thickness,
            cv2.LINE_AA,
        )


def _draw_image_center(img: np.ndarray, center: Tuple[int, int]) -> None:
    cv2.drawMarker(img, center, COLOR_CENTER, cv2.MARKER_TILTED_CROSS, 22, 2)


def _draw_hook_projection(img: np.ndarray, hook_xy: Tuple[int, int]) -> None:
    cv2.drawMarker(img, hook_xy, COLOR_HOOK, cv2.MARKER_CROSS, 28, 2)


def _draw_hose_axis(
    img: np.ndarray,
    hose_pose: Tuple[float, float, float, float, Tuple[float, float]],
) -> None:
    cx, cy, _, length_px, (ux, uy) = hose_pose
    half = max(length_px * 0.5, 60.0)
    p1 = (int(cx - ux * half), int(cy - uy * half))
    p2 = (int(cx + ux * half), int(cy + uy * half))
    cv2.line(img, p1, p2, COLOR_HOSE_AXIS, 2, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), 5, COLOR_HOSE_AXIS, -1)


def _draw_target_hose_row(
    img: np.ndarray,
    target_hose_cy: float,
    tol_px: float,
    image_width: int,
) -> None:
    y = int(target_hose_cy)
    draw_dashed_line(img, (0, y), (image_width, y), COLOR_TARGET_HOSE, thickness=2)
    if tol_px > 1:
        upper = max(0, int(y - tol_px))
        lower = int(y + tol_px)
        # Draw thin tolerance lines (no filled overlay to keep the frame readable).
        cv2.line(
            img, (0, upper), (image_width, upper), COLOR_TARGET_HOSE, 1, cv2.LINE_AA
        )
        cv2.line(
            img, (0, lower), (image_width, lower), COLOR_TARGET_HOSE, 1, cv2.LINE_AA
        )


def _draw_target_sphere(
    img: np.ndarray,
    target_xy: Tuple[float, float],
    tol_px: float,
    in_fov: bool,
) -> None:
    color = COLOR_TARGET_SPHERE if in_fov else COLOR_TARGET_SPHERE_LOST
    cx, cy = int(target_xy[0]), int(target_xy[1])
    radius = max(int(tol_px), 8)
    cv2.circle(img, (cx, cy), radius, color, 2, cv2.LINE_AA)
    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 18, 2)


def _draw_sphere(img: np.ndarray, sphere_xy: Tuple[float, float]) -> None:
    cv2.circle(img, (int(sphere_xy[0]), int(sphere_xy[1])), 8, COLOR_SPHERE, -1)


def _draw_error_arrows(
    img: np.ndarray,
    sphere_xy: Optional[Tuple[float, float]],
    target_sphere_xy: Tuple[float, float],
    hose_cy: float,
    target_hose_cy: float,
    hook_xy: Tuple[int, int],
) -> None:
    """Two arrows: one for the sphere-anchor (vy) error, one for the
    hose-row (vx) error. Both anchored at where the relevant target sits.
    """
    # Sphere anchor: green arrow current -> target.
    if sphere_xy is not None:
        cv2.arrowedLine(
            img,
            (int(sphere_xy[0]), int(sphere_xy[1])),
            (int(target_sphere_xy[0]), int(target_sphere_xy[1])),
            COLOR_ERR_VY,
            2,
            tipLength=0.15,
            line_type=cv2.LINE_AA,
        )

    # Hose-row error: vertical arrow at the hook x from current rope cy
    # toward the target row.
    cv2.arrowedLine(
        img,
        (hook_xy[0], int(hose_cy)),
        (hook_xy[0], int(target_hose_cy)),
        COLOR_ERR_VX,
        2,
        tipLength=0.18,
        line_type=cv2.LINE_AA,
    )


def _draw_altitude_bar(
    img: np.ndarray,
    altitude: float,
    alt_min: float,
    alt_max: float,
    release_alt: float,
) -> None:
    """Right-edge altitude bar with a marker for current altitude and a
    horizontal red dashed line at ``release_alt``.
    """
    H, W = img.shape[:2]
    x0 = W - 70
    y0 = 80
    x1 = W - 40
    y1 = H - 80

    cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)

    span = max(alt_max - alt_min, 0.1)

    def alt_to_y(a: float) -> int:
        # alt_max -> y0 (top), alt_min -> y1 (bottom).
        a_clamped = max(alt_min, min(alt_max, a))
        frac = (alt_max - a_clamped) / span
        return int(y0 + frac * (y1 - y0))

    # Release altitude marker.
    yr = alt_to_y(release_alt)
    draw_dashed_line(
        img,
        (x0 - 8, yr),
        (x1 + 8, yr),
        COLOR_ALT_BAR_RELEASE,
        thickness=2,
        dash=8,
        gap=4,
    )
    cv2.putText(
        img,
        f"R={release_alt:.1f}",
        (x0 - 80, yr + 5),
        _FONT,
        0.5,
        COLOR_ALT_BAR_RELEASE,
        1,
        cv2.LINE_AA,
    )

    # Current altitude marker.
    if altitude is not None:
        ya = alt_to_y(altitude)
        cv2.line(img, (x0 - 8, ya), (x1 + 8, ya), COLOR_ALT_BAR, 3, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{altitude:.2f}m",
            (x0 - 90, ya - 4),
            _FONT,
            0.5,
            COLOR_ALT_BAR,
            1,
            cv2.LINE_AA,
        )


# ----------------------------------------------------------------------
# Per-state composite drawers
# ----------------------------------------------------------------------


def draw_lower_and_align(
    img: np.ndarray,
    *,
    image_center: Tuple[int, int],
    hook_xy: Tuple[int, int],
    hose_pose: Optional[Tuple[float, float, float, float, Tuple[float, float]]],
    sphere_center: Optional[Tuple[float, float]],
    target_sphere_xy: Tuple[float, float],
    target_hose_cy: float,
    tol_center_m: float,
    tol_anchor_m: float,
    ppm_x: float,
    ppm_y: float,
    anchor_sign: int,
    phase: str,
    sphere_can_anchor: bool,
    altitude: Optional[float],
    standoff_m: float,
    alt_min: float,
    alt_max: float,
    release_alt: float,
    err_center_m: float,
    err_anchor_m: float,
    err_angle_deg: float,
    err_center_px: float,
    err_anchor_px: float,
    vx: float,
    vy: float,
    vz: float,
    vyaw: float,
) -> None:
    """LOWER_AND_ALIGN composite overlay (modifies ``img`` in place).

    Renders the controller geometry (image center, hook, hose axis,
    target sphere ring, target hose row, error arrows) plus a left HUD
    with phase / altitude / errors / commanded velocities and a right
    HUD with anchor sign / standoff / tolerances. Includes the altitude
    bar (with release marker).

    Anisotropic px↔m: the hose-row tolerance band uses ``ppm_y`` (image-y
    is the body-x error axis); the sphere-anchor tolerance ring uses
    ``ppm_x`` (image-x is the body-y error axis).
    """
    _, W = img.shape[:2]
    tol_center_px = max(tol_center_m * ppm_y, 1.0)
    tol_anchor_px = max(tol_anchor_m * ppm_x, 1.0)

    _draw_target_hose_row(img, target_hose_cy, tol_center_px, W)
    _draw_target_sphere(img, target_sphere_xy, tol_anchor_px, in_fov=sphere_can_anchor)
    if hose_pose is not None:
        _draw_hose_axis(img, hose_pose)
    if sphere_center is not None:
        _draw_sphere(img, sphere_center)
    if hose_pose is not None:
        hose_cy = hose_pose[1]
        _draw_error_arrows(
            img,
            sphere_center if sphere_can_anchor else None,
            target_sphere_xy,
            hose_cy,
            target_hose_cy,
            hook_xy,
        )
    _draw_image_center(img, image_center)
    _draw_hook_projection(img, hook_xy)
    _draw_altitude_bar(img, altitude or 0.0, alt_min, alt_max, release_alt)

    alt_txt = f"{altitude:.2f}m" if altitude is not None else "n/a"
    anchor_state = "full" if sphere_can_anchor else "hose-only"
    vz_line = f"  vz={vz:+.2f}" if phase == "descend" else ""
    draw_hud(
        img,
        [
            f"LOWER_AND_ALIGN[{phase:>7s}]  alt={alt_txt}  "
            f"ppm_x={ppm_x:4.0f} ppm_y={ppm_y:4.0f}",
            f"anchor: {anchor_state}",
            f"err: hose_cy={err_center_m:+.3f}m ({err_center_px:+5.0f}px)",
            f"     anchor  ={err_anchor_m:+.3f}m ({err_anchor_px:+5.0f}px)",
            f"     angle   ={err_angle_deg:+5.1f}deg",
            f"cmd: vx={vx:+.2f}  vy={vy:+.2f}{vz_line}  vyaw={vyaw:+.2f}",
        ],
        anchor="tl",
    )
    draw_hud(
        img,
        [
            f"anchor_sign={anchor_sign:+d}  standoff={standoff_m:.2f}m",
            f"tol: center={tol_center_m:.2f}m  anchor={tol_anchor_m:.2f}m",
            f"release={release_alt:.2f}m",
        ],
        anchor="tr",
    )


def draw_search_ascend(
    img: np.ndarray,
    *,
    altitude: Optional[float],
    alt_max: float,
    confirmations: int,
    target_confirmations: int,
    sphere_center: Optional[Tuple[float, float]],
    phase: Optional[str] = None,
) -> None:
    """SEARCH_ASCEND composite overlay: minimal — just confirm sphere
    detection, the ascent cap, and the active phase (``ascend`` while
    climbing, ``yaw_search`` once the cap is reached)."""
    if sphere_center is not None:
        _draw_sphere(img, sphere_center)

    alt_txt = f"{altitude:.2f}m" if altitude is not None else "n/a"
    header = (
        f"SEARCH_ASCEND[{phase}]  alt={alt_txt}  cap={alt_max:.2f}m"
        if phase is not None
        else f"SEARCH_ASCEND  alt={alt_txt}  cap={alt_max:.2f}m"
    )
    draw_hud(
        img,
        [
            header,
            f"sphere conf: {confirmations}/{target_confirmations}",
        ],
        anchor="tl",
    )


def draw_orient(
    img: np.ndarray,
    *,
    image_center: Tuple[int, int],
    phase: str,
    sphere_xy: Optional[Tuple[float, float]],
    hose_pose: Optional[Tuple[float, float, float, float, Tuple[float, float]]],
    delta_target: Optional[float],
    err_rad: Optional[float],
    tol_rad: float,
    vyaw: float,
    sample_idx: Optional[int],
    sample_total: int,
) -> None:
    """ORIENT composite overlay (modifies ``img`` in place).

    Draws the image center, the live sphere centroid, the chosen rope
    axis (when available), and a HUD with the sample/spin status, the
    per-frame predicted body-yaw ``delta_target`` (sample phase) or the
    live body-yaw error ``err_rad`` (spin phase), and commanded ``vyaw``.

    The spin loop runs in **body frame** (the cumulative body yaw is
    computed from the sphere's body-frame polar angle), so this overlay
    no longer renders an image-frame target ray — that quantity is
    anisotropy-dependent and would be misleading. The chosen-rope axis
    line is the most direct visual cue: convergence = horizontal in
    image AND sphere in upper half.
    """
    if hose_pose is not None:
        _draw_hose_axis(img, hose_pose)

    if sphere_xy is not None:
        _draw_sphere(img, sphere_xy)

    _draw_image_center(img, image_center)

    if delta_target is None:
        delta_str = "delta_tgt =  n/a"
    else:
        delta_str = f"delta_tgt ={math.degrees(delta_target):+6.1f}deg (body)"
    if err_rad is None:
        err_str = "err       =  n/a"
    else:
        err_str = (
            f"err       ={math.degrees(err_rad):+6.1f}deg  "
            f"tol={math.degrees(tol_rad):.1f}deg"
        )
    sample_str = (
        f"sample {sample_idx}/{sample_total}"
        if sample_idx is not None
        else f"spin (target {sample_total} samples consumed)"
    )

    if phase == "sample":
        lines = [
            f"ORIENT[{phase}]  {sample_str}",
            delta_str,
            f"cmd: vyaw={vyaw:+.2f}",
        ]
    else:
        lines = [
            f"ORIENT[{phase}]  {sample_str}",
            delta_str,
            err_str,
            f"cmd: vyaw={vyaw:+.2f}",
        ]

    draw_hud(img, lines, anchor="tl")


def draw_precision_land(
    img: np.ndarray,
    *,
    image_center: Tuple[int, int],
    base_center: Optional[Tuple[float, float]],
    base_bbox: Optional[Iterable[int]],
    altitude: Optional[float],
    target_altitude: float,
    err_x_m: float,
    err_y_m: float,
    err_total_m: float,
    tol_m: float,
    phase: str,
    confirmations: int,
    target_confirmations: int,
    vx: float,
    vy: float,
    vz: float,
) -> None:
    """PRECISION_LAND composite overlay (modifies ``img`` in place).

    Renders the detected blue-base bbox + centroid, an arrow from the
    base centroid to the image center, the image-center cross, and a
    HUD with phase / altitude / errors in meters / commanded velocities.
    """
    if base_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in base_bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_TARGET_SPHERE, 3)
    if base_center is not None:
        bx, by = int(base_center[0]), int(base_center[1])
        cv2.circle(img, (bx, by), 6, COLOR_SPHERE, -1)
        cv2.arrowedLine(
            img,
            (bx, by),
            image_center,
            COLOR_ERR_VY,
            2,
            tipLength=0.15,
            line_type=cv2.LINE_AA,
        )
    _draw_image_center(img, image_center)

    alt_txt = f"{altitude:.2f}m" if altitude is not None else "n/a"
    vz_line = f"  vz={vz:+.2f}" if phase == "descend" else ""
    draw_hud(
        img,
        [
            f"PRECISION_LAND[{phase:>7s}]  alt={alt_txt}  target={target_altitude:.2f}m",
            f"err: x={err_x_m:+.3f}m  y={err_y_m:+.3f}m  total={err_total_m:.3f}m  tol={tol_m:.2f}m",
            f"conf: {confirmations}/{target_confirmations}",
            f"cmd: vx={vx:+.2f}  vy={vy:+.2f}{vz_line}",
        ],
        anchor="tl",
    )


def draw_select_side(
    img: np.ndarray,
    *,
    sphere_center: Optional[Tuple[float, float]],
    u_axis: Optional[Tuple[float, float]],
    side_unit: Optional[Tuple[float, float]],
    length_plus: float,
    length_minus: float,
    plus_is_long: bool,
    sample_idx: int,
    sample_total: int,
    image_center: Tuple[int, int],
) -> None:
    """SELECT_SIDE composite overlay: shows the rope axis, the chosen side
    direction (sphere -> chosen segment), and the running per-side length
    accumulators.
    """
    if sphere_center is not None:
        _draw_sphere(img, sphere_center)
        sx, sy = int(sphere_center[0]), int(sphere_center[1])
        if u_axis is not None:
            ux, uy = u_axis
            scale = 600
            p1 = (int(sx - ux * scale), int(sy - uy * scale))
            p2 = (int(sx + ux * scale), int(sy + uy * scale))
            draw_dashed_line(img, p1, p2, COLOR_HOSE_AXIS, thickness=2, dash=18, gap=10)
        if side_unit is not None:
            ux, uy = side_unit
            tip = (int(sx + ux * 250), int(sy + uy * 250))
            color = COLOR_TARGET_HOSE if plus_is_long else COLOR_ERR_VX
            cv2.arrowedLine(
                img, (sx, sy), tip, color, 4, tipLength=0.15, line_type=cv2.LINE_AA
            )

    _draw_image_center(img, image_center)

    plus_str = f"+{length_plus:5.0f}px"
    minus_str = f"-{length_minus:5.0f}px"
    chosen = "+" if plus_is_long else "-"
    draw_hud(
        img,
        [
            f"SELECT_SIDE  sample {sample_idx}/{sample_total}",
            f"length_plus  = {plus_str}",
            f"length_minus = {minus_str}",
            f"chosen side  = {chosen}",
        ],
        anchor="tl",
    )
