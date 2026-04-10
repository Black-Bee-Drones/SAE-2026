#!/usr/bin/env python3
import rclpy

from yasmin import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from yasmin_viewer import YasminViewerPub

from hook.core.states import Initialize, Takeoff, ReturnToLaunch, Land
from hook.states.sm import HangWireSM


class HangTheWireSM(StateMachine):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.add_state(
            "INITIALIZE",
            Initialize(),
            transitions={SUCCEED: "TAKEOFF", ABORT: "LAND"},
        )

        self.add_state(
            "TAKEOFF",
            Takeoff(),
            transitions={SUCCEED: "HANG_WIRE", ABORT: "RETURN_TO_LAUNCH"},
        )

        self.add_state(
            "HANG_WIRE",
            HangWireSM(),
            transitions={SUCCEED: "RETURN_TO_LAUNCH", ABORT: "RETURN_TO_LAUNCH"},
        )

        self.add_state(
            "RETURN_TO_LAUNCH",
            ReturnToLaunch(),
            transitions={SUCCEED: "LAND", ABORT: "LAND"},
        )

        self.add_state(
            "LAND",
            Land(),
            transitions={SUCCEED: SUCCEED, ABORT: ABORT},
        )

        self.set_start_state("INITIALIZE")


def main():
    rclpy.init()

    try:
        sm = HangTheWireSM()
        viewer = YasminViewerPub("hang_the_wire", sm)

        print("\nStarting Hang the Right Wire mission...\n")
        outcome = sm()
        print(f"\nMission finished: {outcome}")

    except KeyboardInterrupt:
        print("\nMission interrupted by user.")
    except Exception as e:
        print(f"\nMission failed: {e}")
    finally:
        rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
