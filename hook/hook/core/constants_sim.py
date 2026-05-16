"""Simulation overrides for :mod:`hook.core.constants`.

Imported into the ``constants`` module namespace by ``constants.py`` when
``HOOK_SIM=1``. Real-drone tuning stays the source of truth; this file
only redefines the values that are meaningfully different in Gazebo.

Scaling rule (controllers): every ``Kp`` and every velocity cap is
multiplied by :data:`SIM_SPEED_MULT` relative to the real-drone value.
Same ratio ``Kp / max_vel`` -> same saturation point in meters ->
identical closed-loop shape at ~4x wall-clock speed.

Unchanged on purpose: confirmation counts, tolerances, distances,
output deadbands, timeouts, camera/image/model params. Scaling those
either trips false positives or breaks convergence at higher speed.

Two literal overrides match the rule-book arena geometry exactly:
``SPHERE_HEIGHT_M`` and ``RELEASE_ALTITUDE``.
"""

SIM_SPEED_MULT = 4.0

# --- Altitudes (rule-book arena) ---
SPHERE_HEIGHT_M = 1.7
RELEASE_ALTITUDE = 1.95
WORK_ALTITUDE = 4.0

# --- Search and ascend ---
ASCEND_VELOCITY = 1.0  # 0.30 * 4
ASCEND_VELOCITY_SLOW = 0.28  # 0.07 * 4
ASCEND_YAW_RATE_RAD_S = 1.0  # 0.35 * 4

# --- Approach sphere ---
APPROACH_KP_M = 0.76  # 0.32 * 4
APPROACH_MAX_VELOCITY_XY = 0.45  # 0.20 * 4
APPROACH_DESCEND_VELOCITY = 0.48  # 0.16 * 4

# --- Orient to hook ---
ORIENT_YAW_KP = 0.3  # 0.25 * 4
ORIENT_MAX_YAW_VELOCITY = 0.5  # 0.18 * 4

# --- Lower and align (xy / yaw) ---
HOSE_ANGLE_KP = 0.01489  # 0.00731 * 4
HOSE_ANGLE_MAX_VELOCITY = 0.24  # 0.11 * 4
HOSE_CENTER_KP = 0.48  # 0.34 * 4
HOSE_CENTER_MAX_VELOCITY = 0.32  # 0.15 * 4
SPHERE_ANCHOR_KP = 0.38  # 0.17 * 4

# --- Lower and align (vertical) ---
DESCEND_VZ_KP = 0.20  # 0.10 * 4
DESCEND_VZ_MIN = 0.02  # 0.02 * 4
DESCEND_VZ_MAX = 0.30  # 0.10 * 4
LOWER_RECOVERY_VZ = 0.40  # 0.10 * 4


PRECISION_LAND_KP = 0.49
PRECISION_LAND_MAX_VELOCITY_XY = 0.22
PRECISION_LAND_TOL_M = 0.15
