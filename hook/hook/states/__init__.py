from hook.states.align_to_hose import AlignToHose
from hook.states.approach_sphere import ApproachSphere
from hook.states.descend import DescendAndAlign
from hook.states.release_hook import ReleaseHook
from hook.states.search_and_ascend import SearchAndAscend
from hook.states.select_hose_side import SelectHoseSide

STAGES = [
    ("search_ascend", SearchAndAscend),
    ("approach", ApproachSphere),
    ("select_side", SelectHoseSide),
    ("align", AlignToHose),
    ("descend", DescendAndAlign),
    ("release", ReleaseHook),
]

__all__ = [
    "STAGES",
    "SearchAndAscend",
    "ApproachSphere",
    "SelectHoseSide",
    "AlignToHose",
    "DescendAndAlign",
    "ReleaseHook",
]
