import time

import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone

from bouncing.constants import (
    TAKEOFF_ALTITUDE,
    TAKEOFF_SLEEP,
    TAKEOFF_TIMEOUT,
    ALTITUDE_TOLERANCE,
)


class Takeoff(State):
    """Arms the drone and takes off to search altitude."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        if "mavdrone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("MavDrone not available in Takeoff state.")
            return ABORT
        mavdrone: MavDrone = blackboard["mavdrone"]

        try:
            yasmin.YASMIN_LOG_INFO(f"Taking off to altitude: {TAKEOFF_ALTITUDE}m...")
            mavdrone.arm_takeoff(TAKEOFF_ALTITUDE)
            mavdrone.delay(TAKEOFF_SLEEP)

            start_time = time.time()
            while time.time() - start_time < TAKEOFF_TIMEOUT:
                mavdrone.delay(0.1)

                current_alt = mavdrone.get_rng_alt.range
                yasmin.YASMIN_LOG_INFO(f"Current altitude: {current_alt:.2f}m")

                altitude_error = TAKEOFF_ALTITUDE - current_alt

                if abs(altitude_error) < ALTITUDE_TOLERANCE:
                    yasmin.YASMIN_LOG_INFO("Takeoff altitude reached. Ready to start search pattern.")

                    mavdrone.offboard_velocity(0.0, 0.0, 0.0, 0.0)
                    mavdrone.delay(2)

                    return SUCCEED

                correction_velocity = max(-0.5, min(0.5, 0.3 * altitude_error))
                mavdrone.offboard_velocity(0.0, 0.0, correction_velocity, 0.0)

            yasmin.YASMIN_LOG_ERROR("Takeoff timeout reached.")
            return ABORT

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Takeoff failed: {e}")
            return ABORT
