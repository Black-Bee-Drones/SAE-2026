import yasmin
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, ABORT

from nectar.control import MavrosDrone

from bouncing.constants import (
    RECOVERY_STEP,
    RECOVERY_LIMITE_ALTITUDE,
)

class Recovery(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, ABORT])


    def execute(self, blackboard: Blackboard):
        if ('drone' not in blackboard) or not blackboard['drone']:
            yasmin.YASMIN_LOG_ERROR('drone not available.')
            return ABORT
        drone: MavrosDrone = blackboard['drone']

        yasmin.YASMIN_LOG_INFO('Start.')

        if drone.rel_alt + RECOVERY_STEP > RECOVERY_LIMITE_ALTITUDE:
            yasmin.YASMIN_LOG_ERROR('Limit altitude reached.')
            return FAIL

        drone.move_to(z=RECOVERY_STEP)

        yasmin.YASMIN_LOG_INFO('Completed successfully.')
        return SUCCEED
