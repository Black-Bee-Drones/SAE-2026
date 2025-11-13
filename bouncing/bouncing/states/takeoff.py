import time

import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone


class Takeoff(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.TAKEOFF_ALTITUDE = self.node.get_parameter_or('takeoff_altitude', 3.0)
        self.TAKEOFF_SLEEP = self.node.get_parameter_or('takeoff_sleep', 5)
        self.TAKEOFF_TIMEOUT = self.node.get_parameter_or('takeoff_timeout', 60)
        self.TAKEOFF_ALTITUDE_TOLERANCE = self.node.get_parameter_or('takeoff_altitude_tolerance', 0.1)

    @property
    def node(self):
        return YasminNode.get_instance()

    def execute(self, blackboard: Blackboard):
        if "mavdrone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("MavDrone not available in Takeoff state.")
            return ABORT
        mavdrone: MavDrone = blackboard["mavdrone"]

        try:
            yasmin.YASMIN_LOG_INFO(f"Taking off to altitude: {self.TAKEOFF_ALTITUDE}m...")
            mavdrone.arm_takeoff(self.TAKEOFF_ALTITUDE)
            mavdrone.delay(self.TAKEOFF_SLEEP)

            start_time = time.time()
            while time.time() - start_time < self.TAKEOFF_TIMEOUT:
                mavdrone.delay(0.1)

                current_alt = mavdrone.get_rng_alt.range
                yasmin.YASMIN_LOG_INFO(f"Current altitude: {current_alt:.2f}m")

                altitude_error = self.TAKEOFF_ALTITUDE - current_alt

                if abs(altitude_error) < self.TAKEOFF_ALTITUDE_TOLERANCE:
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
