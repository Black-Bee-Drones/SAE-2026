from yasmin import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from hook.states.detect_sphere import DetectSphere
from hook.states.align_and_approach import AlignAndApproach
from hook.states.center_on_rope import CenterOnRope
from hook.states.descend import DescendAndCenter
from hook.states.release_hook import ReleaseHook


class HangWireSM(StateMachine):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.add_state(
            "DETECT_SPHERE",
            DetectSphere(),
            transitions={SUCCEED: "ALIGN_AND_APPROACH", ABORT: ABORT},
        )

        self.add_state(
            "ALIGN_AND_APPROACH",
            AlignAndApproach(),
            transitions={SUCCEED: "CENTER_ON_ROPE", ABORT: "DETECT_SPHERE"},
        )

        self.add_state(
            "CENTER_ON_ROPE",
            CenterOnRope(),
            transitions={SUCCEED: "DESCEND_AND_CENTER", ABORT: "ALIGN_AND_APPROACH"},
        )

        self.add_state(
            "DESCEND_AND_CENTER",
            DescendAndCenter(),
            transitions={SUCCEED: "RELEASE_HOOK", ABORT: ABORT},
        )

        self.add_state(
            "RELEASE_HOOK",
            ReleaseHook(),
            transitions={SUCCEED: SUCCEED, ABORT: SUCCEED},
        )

        self.set_start_state("DETECT_SPHERE")
