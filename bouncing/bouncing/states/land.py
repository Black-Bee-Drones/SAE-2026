import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone


class Land(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        if ("mavdrone" not in blackboard) or not blackboard["mavdrone"]:
            yasmin.YASMIN_LOG_ERROR("Mavdrone not available in Land state.")
            return ABORT
        mavdrone: MavDrone = blackboard["mavdrone"]

        try:
            yasmin.YASMIN_LOG_INFO("Start land.")

            mavdrone.land()
            mavdrone.delay(10)

            yasmin.YASMIN_LOG_INFO("Landed successfully.")

            return SUCCEED

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Landing failed: {e}")
            return ABORT
