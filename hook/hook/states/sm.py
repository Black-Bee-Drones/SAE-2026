from yasmin import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from hook.states.detect_sphere import DetectSphere
from hook.states.approach_sphere import ApproachSphere
from hook.states.align_to_hose import AlignToHose
from hook.states.descend import DescendAndAlign
from hook.states.release_hook import ReleaseHook


class HangWireSM(StateMachine):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.add_state(
            "DETECT_SPHERE",
            DetectSphere(),
            transitions={SUCCEED: "APPROACH_SPHERE", ABORT: ABORT},
        )

        self.add_state(
            "APPROACH_SPHERE",
            ApproachSphere(),
            transitions={SUCCEED: "ALIGN_TO_HOSE", ABORT: "DETECT_SPHERE"},
        )

        self.add_state(
            "ALIGN_TO_HOSE",
            AlignToHose(),
            transitions={SUCCEED: "DESCEND_AND_ALIGN", ABORT: "APPROACH_SPHERE"},
        )

        self.add_state(
            "DESCEND_AND_ALIGN",
            DescendAndAlign(),
            transitions={SUCCEED: "RELEASE_HOOK", ABORT: ABORT},
        )

        self.add_state(
            "RELEASE_HOOK",
            ReleaseHook(),
            transitions={SUCCEED: SUCCEED, ABORT: SUCCEED},
        )

        self.set_start_state("DETECT_SPHERE")
