import yasmin
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone


class Land(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])


    def execute(self, blackboard: Blackboard):
        if ('mavdrone' not in blackboard) or not blackboard['mavdrone']:
            yasmin.YASMIN_LOG_ERROR('Land(State): MavDrone not available.')
            return ABORT
        mavdrone: MavDrone = blackboard['mavdrone']

        yasmin.YASMIN_LOG_ERROR('Land(State): Start.')

        try:
            mavdrone.land()

        except Exception as e:
            yasmin.YASMIN_LOG_INFO(f'Land(State): Landing failed: {e}.')
            return ABORT

        yasmin.YASMIN_LOG_INFO('Land(State): Completed successfully.')
        return SUCCEED