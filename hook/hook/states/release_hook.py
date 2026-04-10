import time

import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.control import MavrosDrone

from hook.core.constants import SERVO_CHANNEL, HOLD_PWM, RELEASE_PWM


class ReleaseHook(State):
    """Activate servo to release the hook onto the rope."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]

        try:
            yasmin.YASMIN_LOG_INFO("Releasing hook...")
            drone.move_velocity(0.0, 0.0, 0.0, 0.0)
            time.sleep(1)

            drone.do_servo(SERVO_CHANNEL, HOLD_PWM)
            time.sleep(1)

            drone.do_servo(SERVO_CHANNEL, RELEASE_PWM)
            time.sleep(2)

            yasmin.YASMIN_LOG_INFO("Hook released.")
            return SUCCEED

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Hook release failed: {e}")
            return ABORT
