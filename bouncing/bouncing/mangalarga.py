# Main State Machine for Bouncing

import rclpy

import yasmin
from yasmin_ros import set_ros_loggers
from yasmin import StateMachine
from yasmin_viewer import YasminViewerPub
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from bouncing.states import (
    Initialize,
    Takeoff,
    SearchId,
    SearchLandBase,
    Descend,
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
            transitions={SUCCEED:"SEARCH_ID", ABORT:"LAND"},
        )
        self.add_state(
            "SEARCH_ID",
            SearchId(),
            transitions={SUCCEED:"SEARCH_LAND_BASE", ABORT:"LAND"},
        )
        self.add_state(
            "SEARCH_LAND_BASE",
            SearchLandBase(),
            transitions={SUCCEED:"DESCEND", ABORT:"LAND"},
        )
        self.add_state(
            "DESCEND",
            Descend(),
            transitions={SUCCEED:"LAND", ABORT:"LAND"},
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

    YasminViewerPub("YASMIN_DEMO", bouncing_sm)

    try:
        yasmin.YASMIN_LOG_ERROR(bouncing_sm.validate())
        final_outcome = bouncing_sm()
        yasmin.YASMIN_LOG_INFO(final_outcome)
    except KeyboardInterrupt:
        if bouncing_sm.is_running():
            bouncing_sm.cancel_state()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
