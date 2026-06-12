from yasmin import State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from yasmin import Blackboard
import yasmin

from zaxis.drone import Drone

class ReturnToLaunch(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard : Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Could not retrieve Drone instance from blackboard.")
            return ABORT
        
        self.drone: Drone = blackboard["drone"]

        yasmin.YASMIN_LOG_INFO("Returning to launch...")
        try:
            self.drone.rtl().wait(timeout=30)
            yasmin.YASMIN_LOG_INFO("Return to launch initiated.")
            return SUCCEED
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Return to launch failed: {e}")
            return ABORT
        