import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone


class Land(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    @property
    def __state_name__(self):
        return f'{self.__class__.__name__}({', '.join([cls.__name__ for cls in self.__class__.__bases__])})'

    def execute(self, blackboard):
        if ('mavdrone' not in blackboard) or not blackboard['mavdrone']:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: MavDrone not available.')
            return ABORT
        mavdrone: MavDrone = blackboard['mavdrone']

        yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: Start.')

        try:
            mavdrone.land()

        except Exception as e:
            yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Landing failed: {e}.')
            return ABORT

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Completed successfully.')
        return SUCCEED