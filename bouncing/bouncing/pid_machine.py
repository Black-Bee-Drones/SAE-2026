import rclpy

import yasmin
from yasmin_ros import set_ros_loggers
from yasmin import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, TIMEOUT, ABORT

from bouncing.states import (
    Initialize,
    Takeoff,
    PreciseLanding,
    Recovery,
    Hover,
    Land,
)


class Bouncing(StateMachine):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.add_state(
            "INITIALIZE",
            Initialize(start_target_base={'shape': '0', 'number': '5'}),
            transitions={SUCCEED:"TAKEOFF", ABORT:ABORT},
        )
        self.add_state(
            "TAKEOFF",
            Takeoff(),
            transitions={SUCCEED:"PRECISE_LANDING", ABORT:"LAND"},
        )
        self.add_state(
            "PRECISE_LANDING",
            PreciseLanding(),
            transitions={SUCCEED:"HOVER", FAIL: "RECOVERY", TIMEOUT: "LAND", ABORT:"LAND"},
        )
        self.add_state(
            "RECOVERY",
            Recovery(),
            transitions={SUCCEED:"PRECISE_LANDING", FAIL: "LAND", ABORT:"LAND"},
        )
        self.add_state(
            "HOVER",
            Hover(),
            transitions={SUCCEED:"LAND", FAIL: "RECOVERY", ABORT:"LAND"},
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
