import yasmin
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.control import MavrosDrone


class Land(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])


    def log(self, msg, style='info'):
        class_name = f'{self.__class__.__name__}({", ".join([cls.__name__ for cls in self.__class__.__bases__])})'

        if style == 'info':
            yasmin.YASMIN_LOG_INFO(f'{class_name}: {msg}')
        elif style == 'error':
            yasmin.YASMIN_LOG_ERROR(f'{class_name}: {msg}')


    def execute(self, blackboard: Blackboard):
        if ('drone' not in blackboard) or not blackboard['drone']:
            self.log(
                'drone not available.',
                style='error',
            )
            return ABORT
        drone: MavrosDrone = blackboard['drone']

        self.log('Start.')

        try:
            drone.land()

        except Exception as e:
            self.log(
                f'Landing failed: {e}.',
                style='error'
            )
            return ABORT

        self.log('Completed successfully.')
        return SUCCEED
