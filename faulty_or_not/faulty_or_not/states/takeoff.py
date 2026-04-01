import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from zaxis.drone import Drone

from faulty_or_not.parameters import TAKEOFF_ALTITUDE


class Takeoff(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.node = YasminNode.get_instance()

    def execute(self, blackboard : Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Could not retrieve Drone instance from blackboard.")
            return ABORT

        self.drone : Drone = blackboard["drone"]
        yasmin.YASMIN_LOG_INFO("Taking off...")

        try:
            self.drone.arm().wait()
            self.drone.takeoff(TAKEOFF_ALTITUDE).wait(timeout=20)
            print("Primary origin:", self.drone.primary_origin)
            yasmin.YASMIN_LOG_INFO("Takeoff successful.")
            return SUCCEED
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Takeoff failed: {e}")
            return ABORT
        