import math
import os

import numpy as np
from ament_index_python.packages import get_package_share_directory

# --- Altitude (meters) ---
INITIAL_TAKEOFF_ALTITUDE = 4.5
MAX_ASCEND_ALTITUDE = 6.2
WORK_ALTITUDE = 3.4
RELEASE_ALTITUDE = 2.05
RTL_ALTITUDE = 3.0

# --- Camera (Arducam 2MP IMX662, USB) ---
# Specs: 1920x1080, FOV 102(D) x 86(H) x 47(V), EFL 3.9mm, F1.0
IMAGE_SOURCE = "webcam"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_CENTER_X = IMAGE_WIDTH // 2
IMAGE_CENTER_Y = IMAGE_HEIGHT // 2
HORIZONTAL_FOV_DEG = 86.0
VERTICAL_FOV_DEG = 47.0

# Frame layout (FLU body): +x forward, +y left, +z up.
#
# CAMERA_BODY_OFFSET_*: camera position in body frame (meters from drone
# center).
# Used by states that reason about distance from the drone center to a
# world point (e.g. APPROACH safety floor).
#
# CAMERA_TO_HOOK_BODY_*: hook position relative to the camera, in body
# frame (= hook_body - camera_body). Used by perception.hook_image_offset
# to project the hook into the image so the controller can park the HOOK
# (not the camera) over the rope.

CAMERA_BODY_OFFSET_X_M = 0.03  # 5 cm forward of drone center
CAMERA_BODY_OFFSET_Y_M = -0.03  # 3 cm to the right (-y)
CAMERA_TO_HOOK_BODY_X_M = 0.00  # hook on the same +x line as camera
CAMERA_TO_HOOK_BODY_Y_M = 0.09  # hook 10 cm to the left of camera

# Used by px_per_meter_x / px_per_meter_y to remove the parallax error
# (target plane height instead of the ground plane).
SPHERE_HEIGHT_M = 1.7  # sphere mounted on hose at top of supports

# Pixel <-> meter conversion strategy. Selects one of three implementations
# in :mod:`hook.core.camera_scaling`:
#   "fov"       - per-axis from HFOV/VFOV. No calibration needed but disagrees
#                 with the calibrated lens at 1920x1080 by ~24% (datasheet
#                 HFOV=86 deg, calibrated HFOV=70.7 deg).
#   "efl"       - isotropic pinhole from EFL_mm and pixel pitch.
#   "intrinsic" - calibrated K + distortion below. Recommended.
CAMERA_SCALING_METHOD = "fov"
CAMERA_EFL_MM = 3.9
CAMERA_PIXEL_SIZE_UM = 2.9

# Intrinsic matrix and distortion from a ChArUco calibration at 1920x1080
# (nectar.vision.camera.calibration.CharucoCalibration, 5x7 board, 37.04 mm
# square / 27.78 mm marker, DICT_4X4_1000). 46 valid views, mean reprojection
# error 0.46 px. Calibrated HFOV = 2*atan(960/fx) = 70.7 deg, VFOV = 43.4 deg.
CAMERA_INTRINSIC_K = np.array(
    [
        [1353.3683044877657, 0.0, 1140.923487147102],
        [0.0, 1355.4448404309912, 529.9287253668848],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
CAMERA_INTRINSIC_DIST = np.array(
    [
        -0.3568019973766494,
        0.16045446741334152,
        0.001923261467798276,
        0.00026084634626530193,
        -0.05407327493687658,
    ],
    dtype=np.float64,
)

# Segmentation model
SEG_MODEL_PATH = os.path.join(
    get_package_share_directory("hook"),
    "models",
    "sae-2026-hang-all-yolo26n-seg-v2-960.pt",
)
SEG_IMGSZ = 960
SEG_IOU = 0.6

SPHERE_CLASS = "sphere"
HOSE_CLASS = "rose"

SPHERE_CONF = 0.70
HOSE_CONF = 0.4
SEG_PREDICT_CONF = min(SPHERE_CONF, HOSE_CONF)

# Search and ascend
ASCEND_VELOCITY = 0.19  # m/s upward while searching
ASCENT_STOP_CONFIRMATIONS = 6
ASCENT_TIMEOUT = 80.0  # seconds
# After hitting MAX_ASCEND_ALTITUDE without a sphere debounce, the state
# switches to yaw-search at this rate (FLU, +vyaw = CCW) until the sphere
# is detected or ASCENT_TIMEOUT fires. Covers the case where the takeoff
# yaw leaves the sphere outside the camera frustum.
ASCEND_YAW_RATE_RAD_S = 0.11

# Approach sphere
APPROACH_TARGET_DISTANCE_M = 0.75  # parked hook-to-sphere horizontal distance
APPROACH_MIN_SAFE_DISTANCE_M = 0.60  # safety floor (never push closer)
APPROACH_KP_M = 0.32  # m/s per m_error (same Kp on both axes)
APPROACH_TOL_M = 0.3  # convergence band
APPROACH_CONFIRMATIONS = 5  # consecutive frames inside tol to declare parked
APPROACH_MAX_VELOCITY_XY = 0.2
APPROACH_INIT_BEARING_FRAMES = 3
APPROACH_MIN_INIT_DIST_M = 0.12  # skip bearing lock if first sample closer than this
APPROACH_DESCEND_VELOCITY = 0.16  # m/s downward during step 2
APPROACH_TIMEOUT = 180  # seconds
APPROACH_MAX_LOST_FRAMES = 60

# Hose side selection
SIDE_SAMPLE_FRAMES = 9
SIDE_LENGTH_RATIO = 1.4
SIDE_TIMEOUT = 30.0

# ORIENT — predictive yaw to put the chosen rope perpendicular AND in front.
# Sample N frames after SELECT_SIDE; per frame compute the closest BODY
# yaw rotation Δ (anisotropic-deprojected via px_per_meter_x/_y) that
# makes the rope perpendicular to body +x AND keeps the sphere in front.
# Vector-mean over (cosΔ, sinΔ) handles wrap. If |Δ| < ORIENT_SKIP_THRESHOLD_RAD:
# SUCCEED with no rotation. Otherwise spin: PID drives the cumulative
# body yaw (computed from the sphere's body-frame polar angle) to Δ_target.
ORIENT_SAMPLE_FRAMES = 7
ORIENT_SKIP_THRESHOLD_RAD = math.radians(7.0)
ORIENT_YAW_KP = 0.25  # rad/s per rad of polar-angle error
ORIENT_MAX_YAW_VELOCITY = 0.18
ORIENT_ANGLE_TOLERANCE_RAD = math.radians(14.0)
ORIENT_CONFIRMATIONS = 4
ORIENT_TIMEOUT = 180.0  # seconds
ORIENT_MAX_LOST_FRAMES = 65  # sphere-loss tolerance during the spin

PID_MIN_OUTPUT_VELOCITY_XY = 0.05  # m/s
PID_MIN_OUTPUT_VYAW = 0.05  # rad/s

HOSE_MIN_CONTOUR_AREA = 200  # px², minimum contour area to fit a hose pose

# Hose alignment
HOSE_ANGLE_TOLERANCE_DEG = 7.0
HOSE_ANGLE_KP = 0.00731  # rad/s per degree
HOSE_ANGLE_MAX_VELOCITY = 0.11  # rad/s
HOSE_CENTER_TOLERANCE_M = 0.235
HOSE_CENTER_KP = 0.34  # m/s per m
HOSE_CENTER_MAX_VELOCITY = 0.15  # m/s
HOSE_ALIGN_CONFIRMATIONS = 3
HOSE_ALIGN_TIMEOUT = 180  # seconds
# Chosen-hose loss tolerance for the merged LOWER_AND_ALIGN state. Sphere
# loss is NOT counted: when sphere is missing the controller falls back to
# hose-only (vy=0). Only consecutive frames where the chosen rope itself
# can't be detected count toward this limit.
LOWER_MAX_LOST_FRAMES = 86
# Once the chosen-rope-lost streak exceeds this (but is still below
# LOWER_MAX_LOST_FRAMES), the state stops descending / hovering and
# climbs at +LOWER_RECOVERY_VZ to widen the FOV. As soon as detection
# returns, normal control resumes. Handles the failure mode where the
# drone drifts at low altitude and both sphere and chosen rope leave
# the frame.
LOWER_RECOVERY_LOST_FRAMES = 12
LOWER_RECOVERY_VZ = 0.10  # m/s upward during the climb-recovery

# Sphere anchor (along-hose) used by LOWER_AND_ALIGN.
SPHERE_ANCHOR_DISTANCE_M = 0.8  # meters from sphere center along chosen hose direction
SPHERE_ANCHOR_TOLERANCE_M = 0.25
SPHERE_ANCHOR_KP = 0.17  # m/s per m

# Rope held this far in front of the hook (body +x) during ALIGN. Keeps
# the rope in the upper half of the image so the lidar (mounted aft of
# the hook) never passes over the rope while yaw/lateral converge. DESCEND
# linearly ramps this to 0 over DESCEND_STANDOFF_RAMP_SEC to avoid the
# step-input dash forward that crossed the rope in earlier runs.
ALIGN_STANDOFF_M = 0.25

# Yaw-first sub-phase: while |rope_angle| > this, only command vyaw and hold
# vx=vy=0. Prevents the position controllers from acting on hose_cy and
# sphere_cx while the rope is still far from horizontal in the image (their
# signals are geometrically meaningless until yaw is close to perpendicular).
ALIGN_YAW_FIRST_TOLERANCE_DEG = 17.0

# Vertical command in the descend phase is a symmetric P controller around
# RELEASE_ALTITUDE with a +-RELEASE_ALTITUDE_TOLERANCE_M deadband:
#   err = altitude - RELEASE_ALTITUDE
#   |err| <= tol           -> vz = 0  (in release band, can succeed)
#   err >  tol             -> vz = -clip(VZ_KP*err, VZ_MIN, VZ_MAX)   (descend)
#   err < -tol             -> vz = +clip(VZ_KP|err|, VZ_MIN, VZ_MAX)  (climb back)
# Bidirectional so an unstable lidar reading or wind kick below the floor
# is actively recovered instead of allowing the drone to ride down while
# still chasing lateral/yaw - which is the regime that risks hitting the
# rope/sphere/ground.
DESCEND_VZ_KP = 0.1
DESCEND_VZ_MIN = 0.02
DESCEND_VZ_MAX = 0.1
DESCEND_RELEASE_CONFIRMATIONS = 5
DESCEND_TIMEOUT = 180  # seconds
# Release succeeds only when the drone is inside this symmetric band
# around RELEASE_ALTITUDE (and lateral/angle/anchor are within tolerance
# for DESCEND_RELEASE_CONFIRMATIONS ticks).
RELEASE_ALTITUDE_TOLERANCE_M = 0.15

# Linear ramp from ALIGN_STANDOFF_M to 0 over this many control ticks at
# the start of DESCEND. Eliminates the ~200 px target step (~0.30 m at
# WORK_ALTITUDE) that previously saturated vx and made the drone shoot
# past the rope. Tick-counted (not wall-clock)
DESCEND_STANDOFF_RAMP_TICKS = 10

# Sphere is the lateral anchor only while its target image-x is comfortably
# inside the frame. As altitude drops, ppm grows and target_sphere_cx walks
# off the image; below the cutoff the drone has already aligned, so we hold
# vy=0 (FCU position-hold) and finish descent on hose-only references.
DESCEND_SPHERE_TARGET_MARGIN_PX = 100

# Precision land (visual alignment on the blue base)
PRECISION_LAND_MODEL_PATH = os.path.join(
    get_package_share_directory("hook"),
    "models",
    "best_7_class.pt",
)
PRECISION_LAND_CLASS_ID = 7
PRECISION_LAND_CONF = 0.5
PRECISION_LAND_IOU = 0.5
PRECISION_LAND_IMGSZ = 960

PRECISION_LAND_RTL_ALTITUDE = 4.0
PRECISION_LAND_TARGET_ALTITUDE = 0.6
PRECISION_LAND_ALT_TOLERANCE_M = 0.15
PRECISION_LAND_BASE_HEIGHT_M = 0.0

PRECISION_LAND_KP = 0.32
PRECISION_LAND_MAX_VELOCITY_XY = 0.18
PRECISION_LAND_TOL_M = 0.24
PRECISION_LAND_INIT_CONFIRMATIONS = 5
PRECISION_LAND_LAND_CONFIRMATIONS = 5

PRECISION_LAND_VZ_KP = 0.20
PRECISION_LAND_VZ_MAX = 0.17

PRECISION_LAND_MAX_LOST_FRAMES = 60
PRECISION_LAND_TIMEOUT = 120.0

# Servo
SERVO_CHANNEL = 3
HOLD_PWM = 1000.0
RELEASE_PWM = 2000.0

# Saving detections
SAVE_DETECTIONS = True
DETECTION_SAVE_PATH = os.path.expanduser("~/sae2026")

# Live mission monitoring
MISSION_FRAME_TOPIC = "/hook/mission/image/compressed"
MISSION_FRAME_JPEG_QUALITY = 80

# Simulation mode
SIM_MODE = os.environ.get("HOOK_SIM", "0") == "1"

SIM_IMAGE_SOURCE = "/down_camera/compressed"
SIM_IMAGE_COMPRESSED = True

if SIM_MODE:
    IMAGE_SOURCE = SIM_IMAGE_SOURCE
    # Pulls controller-gain / velocity-cap overrides and the rule-book
    # geometry into this module's namespace. See constants_sim.py for the
    # full list and the scaling rule. Anything not redefined there keeps
    # the real-drone value above.
    from .constants_sim import * 
