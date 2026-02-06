import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

import time

from mirela_sdk.control.mavros.drone import MavrosDrone

from faulty_or_not.parameters import TAKEOFF_ALTITUDE


class Takeoff(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.drone : MavrosDrone = None
        self.node = YasminNode.get_instance()

    def execute(self, blackboard : Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Could not retrieve MAVDRONE instance from blackboard.")
            return ABORT

        self.drone: MavrosDrone = blackboard["drone"]
        yasmin.YASMIN_LOG_INFO("Taking off...")

        try:
            self.drone.arm()
            self.drone.takeoff(TAKEOFF_ALTITUDE)
            time.sleep(3)
            yasmin.YASMIN_LOG_INFO("Takeoff successful.")
            return SUCCEED
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Takeoff failed: {e}")
            return ABORT
        