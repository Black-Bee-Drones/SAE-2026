#!/usr/bin/env python3
"""Hook mission entry point.

Run the full pipeline or any contiguous prefix of stages, picked via CLI:

    ros2 run hook mangalarga                              # full mission, RTL+land
    ros2 run hook mangalarga --stages search_ascend       # up through 1st stage
    ros2 run hook mangalarga --stages approach            # up through 2nd stage
    ros2 run hook mangalarga --stages search_ascend,approach --end land
    ros2 run hook mangalarga --end none                   # full mission, no auto end
    ros2 run hook mangalarga --list                       # print stage names

Ctrl+C handling (competition safety requirement)
------------------------------------------------
Yasmin's ``StateMachine.__call__`` calls ``sigaction(SIGINT, ...)`` at
runtime, so any Python ``signal.signal(SIGINT, ...)`` is silently
overridden once the SM starts. Yasmin's own SIGINT handler does not
trigger emergency land — it just exits.

Workaround: split into two processes. The parent (this script) does
nothing but spawn the SM in a child with ``start_new_session=True`` and
forward Ctrl+C as SIGTERM. The child runs the SM with a SIGTERM handler
that lands the drone (Yasmin does not intercept SIGTERM). Second Ctrl+C
escalates to SIGKILL.
"""
import argparse
import os
import signal
import subprocess
import sys
from typing import List, Tuple, Type

import rclpy
from rclpy.signals import SignalHandlerOptions
from yasmin import Blackboard, State, StateMachine
from yasmin_ros.basic_outcomes import ABORT, SUCCEED
from yasmin_viewer import YasminViewerPub

from hook.core.states import Initialize, Land, ReturnToLaunch, Takeoff
from hook.states import STAGES, PrecisionLand

_CHILD_ENV = "HOOK_MANGALARGA_CHILD"


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
    elif end == "precision_land":
        success_terminal = "PRECISION_LAND"
        abort_terminal = "PRECISION_LAND"
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
    elif end == "precision_land":
        sm.add_state(
            "PRECISION_LAND", PrecisionLand(),
            transitions={SUCCEED: "LAND", ABORT: "LAND"},
        )
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
        "--end", default="rtl", choices=["rtl", "land", "precision_land", "none"],
        help="what to do after the last stage (default: rtl)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="print stage names in mission order and exit",
    )
    return parser.parse_args(argv)


def _emergency_land(blackboard: Blackboard) -> None:
    """Unconditional land on the drone stored in ``blackboard``.

    Safety hook for Ctrl+C — competition rules require the drone to land
    when the operator interrupts the mission, regardless of the active
    state. No-op if the drone hasn't been created yet (e.g., Ctrl+C
    during INITIALIZE).
    """
    if not blackboard.contains("drone"):
        return
    try:
        blackboard["drone"].move_velocity(0.0, 0.0, 0.0, 0.0, duration=1.0)
        blackboard["drone"].land()
    except Exception as e:
        print(f"Emergency land failed: {e}")


def _parent_main() -> int:
    """Spawn the SM child, forward Ctrl+C as SIGTERM."""
    argv = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    args = _parse_args(argv)
    if args.list:
        for name, _ in STAGES:
            print(name)
        return 0

    env = os.environ.copy()
    env[_CHILD_ENV] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "hook.mangalarga"] + sys.argv[1:],
        env=env,
        start_new_session=True,
    )

    sigint_count = [0]

    def _on_sigint(signum, frame):
        sigint_count[0] += 1
        if sigint_count[0] == 1:
            print("\nCtrl+C: emergency land...", flush=True)
            try:
                proc.terminate()
            except Exception:
                pass
        else:
            print("\nDouble Ctrl+C: force kill.", flush=True)
            try:
                proc.kill()
            except Exception:
                pass

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)
    return proc.wait()


def _child_main() -> None:
    """Run the SM with a SIGTERM handler that emergency-lands the drone."""
    argv = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    args = _parse_args(argv)
    stages = _resolve_stages(args.stages)

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    blackboard = Blackboard()
    landing = {"in_progress": False}

    def _on_sigterm(signum, frame):
        if landing["in_progress"]:
            os._exit(1)
        landing["in_progress"] = True
        print("\nEmergency land triggered.", flush=True)
        _emergency_land(blackboard)
        try:
            rclpy.shutdown()
        except Exception:
            pass
        os._exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        sm = build_sm(stages, args.end)
        sm.set_sigint_handler(False)
        # YasminViewerPub(sm, fsm_name="hang_the_wire")

        names = ", ".join(n for n, _ in stages) or "<none>"
        print(f"\nStarting mission: stages=[{names}] end={args.end}\n")
        outcome = sm(blackboard)
        print(f"\nMission finished: {outcome}")
    except Exception as e:
        print(f"\nMission failed: {e}")
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("Shutdown complete.")


def main():
    if os.environ.get(_CHILD_ENV) == "1":
        _child_main()
    else:
        sys.exit(_parent_main())


if __name__ == "__main__":
    main()
