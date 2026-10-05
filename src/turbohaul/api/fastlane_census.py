"""GET /api/fastlane/census — Fast Lane discovered-address census.

The route lives here; the underlying data comes from
TurbohaulManager.fastlane_census_snapshot(), when the manager provides it.
Guarded via getattr the same way as config_put.py's invalidate_fastlane
call, so this route works whether or not that method exists:
absent -> empty table + explicit backend_pending flag instead of a 500.
"""
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/fastlane", tags=["fastlane"])


def _reshape_row(row: dict) -> dict:
    """Map fastlane_census_snapshot()'s row shape onto the FE contract
    (FastLane.tsx, api.ts's FastLaneDiscoverable — declared as an
    open record whose "exact field set is owned elsewhere", i.e. here):
    ``classes`` (a dict of class -> count) becomes ``tag_classes_seen``, a
    sorted list of the class names seen; ``models`` (already a list)
    becomes ``models_seen`` unchanged. Every other field passes through
    as-is. Reshaping here, not in the manager, keeps
    fastlane_census_snapshot() byte-identical per its own docstring ("the
    API-layer route calls this and serializes the result").
    """
    reshaped = dict(row)
    classes = reshaped.pop("classes", {})
    reshaped["tag_classes_seen"] = sorted(classes)
    reshaped["models_seen"] = reshaped.pop("models", [])
    return reshaped


@router.get("/census")
async def get_fastlane_census(request: Request) -> dict:
    mgr = request.app.state.manager
    snapshot_fn = getattr(mgr, "fastlane_census_snapshot", None)
    if snapshot_fn is None:
        return {"backend_pending": True, "entries": []}
    snapshot = snapshot_fn()
    if isinstance(snapshot, dict):
        rest = {k: v for k, v in snapshot.items() if k != "rows"}
        entries = [_reshape_row(row) for row in snapshot.get("rows", [])]
        return {"backend_pending": False, "entries": entries, **rest}
    return {"backend_pending": False, "entries": snapshot}
