import math
import threading

import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.control.mavros.drone import MavrosDrone

from nectar.control.types import MoveReference

# Circle parameters
CIRCLE_RADIUS = 1.5   # radius in meters
CIRCLE_SPEED = 0.5    # tangential speed in m/s


class Navigation(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.drone: MavrosDrone = None
        self.node = YasminNode.get_instance()
        self._circle_thread: threading.Thread = None
        self._circle_stop_event = threading.Event()

    def execute(self, blackboard: Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Could not retrieve MAVDRONE instance from blackboard.")
            return ABORT
        self.drone = blackboard["drone"]

        locations = blackboard["locations"]
        control_index = blackboard["control_index"]

        cx, cy, cz = locations[control_index]

        # Navigate to the location center
        self.drone.move_to(x=cx, y=cy, z=cz, reference=MoveReference.TAKEOFF)

        # Move to the starting point on the circumference so the circle
        # is actually centered at (cx, cy, cz).
        start_x = cx + CIRCLE_RADIUS
        start_y = cy
        self.drone.move_to(x=start_x, y=start_y, z=cz, reference=MoveReference.TAKEOFF)

        # Start non-blocking circle flight in a separate thread
        self._start_circle()

        # Save reference so GaugeReading can stop the circle
        blackboard["navigation_state"] = self

        return SUCCEED

    def _start_circle(self) -> None:
        self._circle_stop_event.clear()
        self._circle_thread = threading.Thread(
            target=self._fly_circle, daemon=True
        )
        self._circle_thread.start()
        self.node.get_logger().info(
            f"Circle thread started (R={CIRCLE_RADIUS}m, speed={CIRCLE_SPEED}m/s)"
        )

    def stop_circle(self) -> None:
        if self._circle_thread is not None and self._circle_thread.is_alive():
            self._circle_stop_event.set()
            self._circle_thread.join(timeout=3.0)
            self.drone.move_velocity(vx=0.0, vy=0.0, vz=0.0, duration=0.5)
            self.node.get_logger().info("Circle thread stopped.")

    def _fly_circle(self) -> None:
        omega = CIRCLE_SPEED / CIRCLE_RADIUS          # angular velocity (rad/s)
        total_time = (2 * math.pi) / omega             # time for one full revolution
        dt = 1.0 / 30.0                                # ~30 Hz publish rate
        elapsed = 0.0

        self.node.get_logger().info(
            f"Flying circle: R={CIRCLE_RADIUS}m, speed={CIRCLE_SPEED}m/s, "
            f"omega={math.degrees(omega):.1f}°/s, T={total_time:.1f}s"
        )

        while elapsed < total_time and not self._circle_stop_event.is_set():
            theta = omega * elapsed

            vx = -CIRCLE_SPEED * math.sin(theta)
            vy = CIRCLE_SPEED * math.cos(theta)

            self.drone.move_velocity(
                vx=vx,
                vy=vy,
                vz=0.0,
                reference=MoveReference.TAKEOFF,
            )

            self.drone.delay(dt)
            elapsed += dt

        # Stop the drone after completing (or being interrupted)
        self.drone.move_velocity(vx=0.0, vy=0.0, vz=0.0, duration=0.5)
        self.node.get_logger().info("Circle trajectory completed (360°)")