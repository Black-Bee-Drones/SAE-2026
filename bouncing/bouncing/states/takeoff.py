import time

import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone


class Takeoff(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.TAKEOFF_ALTITUDE = self.node.get_parameter_or('takeoff_altitude', 3.0)
        self.TAKEOFF_SLEEP = self.node.get_parameter_or('takeoff_sleep', 5)
        self.TAKEOFF_TIMEOUT = self.node.get_parameter_or('takeoff_timeout', 60)
        self.TAKEOFF_ALTITUDE_TOLERANCE = self.node.get_parameter_or('takeoff_altitude_tolerance', 0.1)

    @property
    def node(self):
        return YasminNode.get_instance()

    @property
    def __state_name__(self):
        return f'{self.__class__.__name__}({', '.join([cls.__name__ for cls in self.__class__.__bases__])})'

    def execute(self, blackboard: Blackboard):
        if 'mavdrone' not in blackboard:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: MavDrone not available.')
            return ABORT
        mavdrone: MavDrone = blackboard['mavdrone']

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Start.')

        yasmin.YASMIN_LOG_INFO(f'Taking off to altitude: {self.TAKEOFF_ALTITUDE}m...')
        try:
            mavdrone.arm_takeoff(self.TAKEOFF_ALTITUDE)
            mavdrone.delay(self.TAKEOFF_SLEEP)

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: Taking off failed: {e}.')
            return ABORT

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Completed successfully.')
        return SUCCEED
