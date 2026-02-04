import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros.drone import MavrosDrone

from mirela_sdk.control.types import MoveReference


class Navigation(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.drone : MavrosDrone = None
        self.node = YasminNode.get_instance()

    def execute(self, blackboard : Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Could not retrieve MAVDRONE instance from blackboard.")
            return ABORT
        drone: MavDrone = blackboard["drone"]

        locations = blackboard["locations"]
        control_index = blackboard["control_index"]

        x,y,z = locations[control_index]

        drone.move_to(x=x,y=y,z=z,reference=MoveReference.WORLD)
        return SUCCEED