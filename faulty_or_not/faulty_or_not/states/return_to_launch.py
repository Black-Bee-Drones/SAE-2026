from yasmin import State
from mirela_sdk.control.mavros.drone import MavrosDrone
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from yasmin import Blackboard
import yasmin

from ..parameters import TAKEOFF_ALTITUDE

class ReturnToLaunch(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.drone : MavrosDrone = None

    def execute(self, blackboard : Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Could not retrieve MAVDRONE instance from blackboard.")
            return ABORT
        
        self.drone: MavrosDrone = blackboard["drone"]

        yasmin.YASMIN_LOG_INFO("Returning to launch...")
        try:
            self.drone.rtl(rtl_alt=TAKEOFF_ALTITUDE, rtl_strategy="PID")
            yasmin.YASMIN_LOG_INFO("Return to launch initiated.")
            return SUCCEED
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Return to launch failed: {e}")
            return ABORT
        