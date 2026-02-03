import yasmin
from yasmin import State, Blackboard
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone

from bouncing.constants import (
    TAKEOFF_ALTITUDE,
    TAKEOFF_SLEEP,
)


class Takeoff(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.node = YasminNode.get_instance()


    def execute(self, blackboard: Blackboard):
        if 'mavdrone' not in blackboard:
            yasmin.YASMIN_LOG_ERROR('Takeoff(State): MavDrone not available.')
            return ABORT
        mavdrone: MavDrone = blackboard['mavdrone']

        yasmin.YASMIN_LOG_INFO('Takeoff(State): Start.')


        yasmin.YASMIN_LOG_INFO(f'Taking off to altitude: {TAKEOFF_ALTITUDE}m...')
        try:
            mavdrone.arm_takeoff(TAKEOFF_ALTITUDE)
            mavdrone.delay(TAKEOFF_SLEEP)
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f'Takeoff(State): Taking off failed: {e}.')
            return ABORT


        yasmin.YASMIN_LOG_INFO('Takeoff(State): Completed successfully.')
        return SUCCEED
