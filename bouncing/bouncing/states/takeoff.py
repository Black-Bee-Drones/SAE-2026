import yasmin
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.control import MavrosDrone

from bouncing.constants import (
    TAKEOFF_ALTITUDE,
)


class Takeoff(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])


    def execute(self, blackboard: Blackboard):
        if 'drone' not in blackboard:
            yasmin.YASMIN_LOG_ERROR('drone not available.')
            return ABORT
        drone: MavrosDrone = blackboard['drone']

        yasmin.YASMIN_LOG_INFO('Start.')

        yasmin.YASMIN_LOG_INFO(f'Taking off to altitude: {TAKEOFF_ALTITUDE}m...')
        try:
            drone.takeoff(TAKEOFF_ALTITUDE, adjust_altitude=False)
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f'Taking off failed: {e}.')
            return ABORT

        yasmin.YASMIN_LOG_INFO('Completed successfully.')
        return SUCCEED
