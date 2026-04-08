import yasmin
from yasmin import State, Blackboard
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.control import MavrosDrone

from bouncing.constants import (
    TAKEOFF_ALTITUDE,
    TAKEOFF_SLEEP,
)


class Takeoff(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.node = YasminNode.get_instance()


    def log(self, msg, style='info'):
        class_name = f'{self.__class__.__name__}({", ".join([cls.__name__ for cls in self.__class__.__bases__])})'

        if style == 'info':
            yasmin.YASMIN_LOG_INFO(f'{class_name}: {msg}')
        elif style == 'error':
            yasmin.YASMIN_LOG_ERROR(f'{class_name}: {msg}')


    def execute(self, blackboard: Blackboard):
        if 'drone' not in blackboard:
            self.log(
                'drone not available.',
                style='error'
            )
            return ABORT
        drone: MavrosDrone = blackboard['drone']

        self.log('Start.')

        self.log(f'Taking off to altitude: {TAKEOFF_ALTITUDE}m...')
        try:
            drone.arm_takeoff(TAKEOFF_ALTITUDE)
            drone.delay(TAKEOFF_SLEEP)
        except Exception as e:
            self.log(
                f'Taking off failed: {e}.',
                style='error'
            )
            return ABORT

        self.log('Completed successfully.')
        return SUCCEED
