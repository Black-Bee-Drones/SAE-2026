#!/usr/bin/env python3
"""Hook mission entry point.

Run the full pipeline or any contiguous prefix of stages, picked via CLI:

    ros2 run hook mangalarga                              # full mission, RTL+land
    ros2 run hook mangalarga --stages search_ascend       # up through 1st stage
    ros2 run hook mangalarga --stages approach            # up through 2nd stage
    ros2 run hook mangalarga --stages search_ascend,approach --end land
    ros2 run hook mangalarga --end none                   # full mission, no auto end
    ros2 run hook mangalarga --list                       # print stage names
"""
import argparse
import sys
from typing import List, Tuple, Type

import rclpy
from yasmin import State, StateMachine
from yasmin_ros.basic_outcomes import ABORT, SUCCEED
from yasmin_viewer import YasminViewerPub

from hook.core.states import Initialize, Land, ReturnToLaunch, Takeoff
from hook.states import STAGES


def _resolve_stages(spec: str) -> List[Tuple[str, Type[State]]]:
    """Resolve a `--stages` spec into a contiguous prefix of STAGES.

    Accepted forms:
      - "all"               -> all stages
      - "<name>"            -> STAGES up to and including <name>
      - "<n1>,<n2>,...,<nk>" -> must equal STAGES[:k]
    """
    if spec == "all":
        return list(STAGES)

    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise SystemExit("--stages: empty")

    valid = [n for n, _ in STAGES]
    for n in names:
        if n not in valid:
            raise SystemExit(
                f"--stages: unknown stage '{n}'. Valid: {', '.join(valid)}"
            )

    if len(names) == 1:
        idx = valid.index(names[0])
        return list(STAGES[: idx + 1])

    expected = valid[: len(names)]
    if names != expected:
        raise SystemExit(
            f"--stages: must be a contiguous prefix. "
            f"Got {names}, expected start {expected}"
        )
    return list(STAGES[: len(names)])


def build_sm(stages: List[Tuple[str, Type[State]]], end: str) -> StateMachine:
    """Build a flat state machine: INITIALIZE -> TAKEOFF -> [stages] -> end."""
    if end == "rtl":
        success_terminal = "RETURN_TO_LAUNCH"
        abort_terminal = "RETURN_TO_LAUNCH"
    elif end == "land":
        success_terminal = "LAND"
        abort_terminal = "LAND"
    elif end == "none":
        success_terminal = SUCCEED
        abort_terminal = ABORT
    else:
        raise ValueError(f"Unknown end: {end}")

    first_stage = stages[0][0].upper() if stages else success_terminal

    sm = StateMachine(outcomes=[SUCCEED, ABORT])
    sm.add_state(
        "INITIALIZE", Initialize(),
        transitions={SUCCEED: "TAKEOFF", ABORT: abort_terminal},
    )
    sm.add_state(
        "TAKEOFF", Takeoff(),
        transitions={SUCCEED: first_stage, ABORT: abort_terminal},
    )
    for i, (name, cls) in enumerate(stages):
        next_state = (
            stages[i + 1][0].upper() if i + 1 < len(stages) else success_terminal
        )
        sm.add_state(
            name.upper(), cls(),
            transitions={SUCCEED: next_state, ABORT: abort_terminal},
        )

    if end == "rtl":
        sm.add_state(
            "RETURN_TO_LAUNCH", ReturnToLaunch(),
            transitions={SUCCEED: "LAND", ABORT: "LAND"},
        )
        sm.add_state(
            "LAND", Land(),
            transitions={SUCCEED: SUCCEED, ABORT: ABORT},
        )
    elif end == "land":
        sm.add_state(
            "LAND", Land(),
            transitions={SUCCEED: SUCCEED, ABORT: ABORT},
        )

    sm.set_start_state("INITIALIZE")
    return sm


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mangalarga")
    parser.add_argument(
        "--stages", default="all",
        help="contiguous prefix of stages: 'all', '<name>', or '<n1>,<n2>,...'",
    )
    parser.add_argument(
        "--end", default="rtl", choices=["rtl", "land", "none"],
        help="what to do after the last stage (default: rtl)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="print stage names in mission order and exit",
    )
    return parser.parse_args(argv)


def main():
    argv = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    args = _parse_args(argv)

    if args.list:
        for name, _ in STAGES:
            print(name)
        return

    stages = _resolve_stages(args.stages)

    rclpy.init()
    try:
        sm = build_sm(stages, args.end)
        YasminViewerPub(sm, fsm_name="hang_the_wire")

        names = ", ".join(n for n, _ in stages) or "<none>"
        print(f"\nStarting mission: stages=[{names}] end={args.end}\n")
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
