from fastapi import APIRouter
from fastapi.responses import Response

from src.controllers.dependencies import Actor, Admin, Db, Writer
from src.schemas.triage import (
    EventAck,
    JudgmentCreate,
    SchedulingCreate,
    TriageReviewCreate,
    WorkstreamCreate,
)
from src.services import triage
from src.utils.csv_files import write_rows

router = APIRouter(prefix="/triage")


@router.get("", status_code=200)
def list_all(session: Db, actor: Actor):
    return triage.list_workstreams(session, actor)


@router.post("", status_code=201)
def create(payload: WorkstreamCreate, session: Db, actor: Writer):
    return triage.create_workstream(session, payload, actor)


@router.get("/{workstream_id}")
def detail(workstream_id: int, session: Db, actor: Actor):
    return triage.detail_view(session, workstream_id, actor)


@router.post("/{workstream_id}/judgments", response_model=EventAck, status_code=201)
def judgment(workstream_id: int, payload: JudgmentCreate, session: Db, actor: Writer):
    return triage.add_judgment(session, workstream_id, payload, actor)


@router.post("/{workstream_id}/schedules", response_model=EventAck, status_code=201)
def schedule(workstream_id: int, payload: SchedulingCreate, session: Db, actor: Writer):
    return triage.schedule_batch(session, workstream_id, payload, actor)


@router.post("/{workstream_id}/review", response_model=EventAck, status_code=201)
def review(workstream_id: int, payload: TriageReviewCreate, session: Db, actor: Admin):
    return triage.submit_review(session, workstream_id, payload, actor)


@router.get("/{workstream_id}/thresholds/{version}")
def thresholds(workstream_id: int, version: int, session: Db, actor: Actor):
    return {"version": version, "candidates": triage.threshold_candidates(
        session, workstream_id, version, actor,
    )}


@router.get("/{workstream_id}/export/{key}")
def export(workstream_id: int, key: str, session: Db, actor: Actor):
    table = triage.export_table(session, workstream_id, key, actor)
    return Response(
        write_rows(table["columns"], table["rows"]), media_type="text/csv",
        headers={
            "Content-Disposition":
                f"attachment; filename=\"triage-{workstream_id}-{table['key']}.csv\""
        },
    )
