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
        self._diamond_stop = threading.Event()

    def stop_diamond(self):
        self._diamond_stop.set()

    def execute(self, blackboard: Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Could not retrieve Drone instance from blackboard.")
            return ABORT
        self.drone = blackboard["drone"]

        locations = blackboard["locations"]
        control_index = blackboard["control_index"]

        offsets = [
            (1, 0, 0),
            (0, -1, 0),
            (-1, 0, 0),
            (0, 1, 0)
        ]

        self._diamond_stop.clear()

        def do_diamond(x, y, z, i = 0):
            handler = self.drone.goto_local(x=x, y=y, z=z)
            while not handler.done() and not self._diamond_stop.is_set():
                time.sleep(1 / 5)
            if not self._diamond_stop.is_set() and i < len(offsets):
                do_diamond(x + offsets[i][0], y + offsets[i][1], z + offsets[i][2], (i + 1))

        x, y, z = locations[control_index]
        blackboard["navigation_state"] = self
        t = threading.Thread(target=do_diamond, args=(x, y, z)).start()