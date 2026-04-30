import rclpy

import yasmin
from yasmin_ros import set_ros_loggers
from yasmin import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, TIMEOUT, ABORT

from bouncing.states import (
    Initialize,
    Takeoff,
    Search,
    PreciseLanding,
    Land,
)


class Bouncing(StateMachine):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.add_state(
            "INITIALIZE",
            Initialize(),
            transitions={SUCCEED:"TAKEOFF", ABORT:ABORT},
        )
        self.add_state(
            "TAKEOFF",
            Takeoff(),
            transitions={SUCCEED:"SEARCH", ABORT:"LAND"},
        )
        self.add_state(
            "SEARCH",
            Search(),
            transitions={SUCCEED:"PRECISE_LANDING", FAIL: "LAND", ABORT:"LAND"},
        )
        self.add_state(
            "PRECISE_LANDING",
            PreciseLanding(),
            transitions={SUCCEED:"LAND", FAIL: "LAND", TIMEOUT: "LAND", ABORT:"LAND"},
        )
        self.add_state(
            "LAND",
            Land(),
            transitions={SUCCEED:SUCCEED, ABORT:ABORT},
        )

        self.set_start_state("INITIALIZE")


def main():
    rclpy.init()

    set_ros_loggers()

    bouncing_sm = Bouncing()

    try:
        final_outcome = bouncing_sm()
        yasmin.YASMIN_LOG_INFO(final_outcome)
    except KeyboardInterrupt:
        if bouncing_sm.is_running():
            bouncing_sm.cancel_state()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
