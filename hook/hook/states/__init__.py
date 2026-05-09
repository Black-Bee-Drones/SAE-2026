from hook.states.approach_sphere import ApproachSphere
from hook.states.lower_and_align import LowerAndAlign
from hook.states.orient_to_hook import OrientToHook
from hook.states.release_hook import ReleaseHook
from hook.states.search_and_ascend import SearchAndAscend
from hook.states.select_hose_side import SelectHoseSide

STAGES = [
    ("search_ascend", SearchAndAscend),
    ("approach", ApproachSphere),
    ("select_side", SelectHoseSide),
    ("orient_to_hook", OrientToHook),
    ("lower_and_align", LowerAndAlign),
    ("release", ReleaseHook),
]

__all__ = [
    "STAGES",
    "SearchAndAscend",
    "ApproachSphere",
    "SelectHoseSide",
    "OrientToHook",
    "LowerAndAlign",
    "ReleaseHook",
]
