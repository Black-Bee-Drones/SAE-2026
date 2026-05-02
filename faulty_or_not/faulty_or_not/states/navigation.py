import threading

import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from zaxis.drone import Drone

import time

class Navigation(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.drone: Drone = None
        self.node = YasminNode.get_instance()
        self._stop_deep_search = threading.Event()


    def stop_movement(self):
        try:
            self.drone.runtime.unregister("flight_goto_local")
        except Exception as e:
            yasmin.YASMIN_LOG_WARN(f"Failed to stop movement: {e}")

    def execute(self, blackboard: Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Could not retrieve Drone instance from blackboard.")
            return ABORT
        self.drone = blackboard["drone"]

        if "locations" not in blackboard or "control_index" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Missing mission state in blackboard (locations/control_index).")
            return ABORT

        locations = blackboard["locations"]
        control_index = blackboard["control_index"]

        if not locations:
            yasmin.YASMIN_LOG_ERROR("No mission locations defined in blackboard.")
            return ABORT

        if control_index >= len(locations):
            yasmin.YASMIN_LOG_WARN(
                f"Control index {control_index} out of bounds for {len(locations)} locations; finishing navigation."
            )
            return ABORT

        x, y, z = locations[control_index]

        yasmin.YASMIN_LOG_INFO(
            f"Navigation: waypoint {control_index + 1}/{len(locations)} -> x={x}, y={y}, z={z}"
        )

        try:
            # handler = self.drone.goto_local(x=x, y=y, z=z, face_wp=True)
            global_origin=blackboard["global_origin"]
            handler = self.drone.goto_global(x=x, y=y, z=z, origin=global_origin, face_wp=True)
            time.sleep(5)
            blackboard["goto_handler"] = handler

            yasmin.YASMIN_LOG_INFO(f"Reached waypoint {control_index}.")

        except Exception as e:
            yasmin.YASMIN_LOG_WARN(f"Failed to navigate to waypoint {control_index}: {e}")
            self.stop_movement()
            return ABORT

        return SUCCEED
    
        # TODO: Implement retry waypoint logic