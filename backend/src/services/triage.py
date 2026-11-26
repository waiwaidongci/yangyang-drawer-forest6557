"""增量分诊工作流服务：事件持久化、状态重建与只读视图。

状态重建规则：
1. 用工作流建立时保存的 source_rows 与 settings 重新拟合森林（训练段与种子固定，
   得分可复现），得到初始 TriageState；
2. 按 seq 顺序回放 judgment / scheduling / review 事件；
3. 事件中带幂等键的重复请求在领域层去重，不新增记录。
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import HTTPException
from sqlalchemy import select

from src.constants.workflow import Decision, Role
from src.domain import triage as domain
from src.models.triage import TriageEvent, TriageWorkstream
from src.services.catalog import get_dataset
from src.services.runs import validate_settings
from src.utils.time import utc_stamp


def _settings_for_domain(settings: dict) -> dict:
    # runs.validate_settings 复用现有 SETTINGS（漏报/误报代价）口径。
    return validate_settings(settings)


def build_state(session, workstream: TriageWorkstream) -> domain.TriageState:
    state = domain.create_workstream(
        workstream.id, workstream.label, workstream.source_rows, workstream.settings,
    )
    events = session.scalars(
        select(TriageEvent)
        .where(TriageEvent.workstream_id == workstream.id)
        .order_by(TriageEvent.seq)
    ).all()
    for event in events:
        domain_event = domain.Event(
            seq=event.seq, kind=event.kind, key=event.idempotency_key,
            payload=event.payload, created_at=event.created_at,
        )
        if event.kind == domain.JUDGMENT:
            state.apply_judgment(domain_event)
        elif event.kind == domain.SCHEDULING:
            state.apply_scheduling(domain_event)
        elif event.kind == domain.REVIEW:
            state.apply_review(domain_event)
    return state


def workstream_info(workstream: TriageWorkstream, state: domain.TriageState | None = None) -> dict:
    info = {
        "id": workstream.id,
        "label": workstream.label,
        "dataset_id": workstream.dataset_id,
        "created_at": workstream.created_at,
        "settings": workstream.settings,
        "error": workstream.error,
        "review": state.review if state else None,
    }
    if state is not None:
        info["metrics"] = state.holdout_metrics()
        info["threshold_versions"] = len(state.threshold_versions)
        info["scheduled_batches"] = sum(
            1 for batch in state.batches.values() if batch.scheduling_event is not None
        )
    return info


def list_workstreams(session, actor):
    statement = select(TriageWorkstream).order_by(TriageWorkstream.id.desc()).limit(100)
    items = []
    for workstream in session.scalars(statement):
        state = None
        if workstream.error is None:
            state = build_state(session, workstream)
            if actor.role == Role.VIEWER and (
                not state.review or state.review["decision"] != Decision.APPROVED
            ):
                continue
        elif actor.role == Role.VIEWER:
            continue
        items.append(workstream_info(workstream, state))
    return items


def get_workstream(session, workstream_id: int, actor) -> tuple[TriageWorkstream, domain.TriageState]:
    workstream = session.get(TriageWorkstream, workstream_id)
    if workstream is None or workstream.error is not None:
        raise LookupError("找不到可用的增量分诊工作流")
    state = build_state(session, workstream)
    if actor.role == Role.VIEWER and (
        not state.review or state.review["decision"] != Decision.APPROVED
    ):
        raise LookupError("找不到可查看的工作流")
    return workstream, state


def create_workstream(session, payload, actor) -> dict:
    dataset = get_dataset(session, payload.dataset_id)
    settings = _settings_for_domain(payload.settings)
    # 增量台账：把完整末检台账按 workstream 建立时点“挖空”待判件。
    source_rows = _mark_pending(dataset.rows, payload.pending_ids)
    created_at = utc_stamp()
    workstream = TriageWorkstream(
        label=payload.label, dataset_id=payload.dataset_id, created_by=actor.id,
        created_at=created_at, settings=settings, source_rows=source_rows, error=None,
    )
    try:
        state = domain.create_workstream(workstream.id or 0, payload.label, source_rows, settings)
    except ValueError as error:
        workstream.error = str(error)
        session.add(workstream)
        session.commit()
        raise HTTPException(422, str(error)) from error
    session.add(workstream)
    session.flush()
    event = TriageEvent(
        workstream_id=workstream.id, seq=1, kind=domain.CREATED, idempotency_key=None,
        payload={"threshold_versions": len(state.threshold_versions)}, created_at=created_at,
    )
    session.add(event)
    session.commit()
    return workstream_info(workstream, build_state(session, session.get(TriageWorkstream, workstream.id)))


def _mark_pending(rows: list[dict], pending_ids: list[str]) -> list[dict]:
    targets = set(pending_ids)
    unknown = targets - {row["id"] for row in rows}
    if unknown:
        raise HTTPException(422, f"待判总成编号不存在：{', '.join(sorted(unknown))}")
    result = []
    for row in rows:
        record = dict(row)
        if record["id"] in targets:
            record["failed"] = None
            record["confirmed_at"] = None
        result.append(record)
    return result


def _next_seq(session, workstream_id: int) -> int:
    last = session.scalar(
        select(TriageEvent.seq)
        .where(TriageEvent.workstream_id == workstream_id)
        .order_by(TriageEvent.seq.desc())
        .limit(1)
    )
    return (last or 1) + 1


def _load_with_events(session, workstream_id: int) -> TriageWorkstream:
    workstream = session.get(TriageWorkstream, workstream_id)
    if workstream is None or workstream.error is not None:
        raise LookupError("找不到可用的增量分诊工作流")
    _ = workstream.events  # 触发关系加载
    return workstream


def _domain_event(kind: str, key: str | None, payload: dict) -> domain.Event:
    return domain.Event(0, kind, key, payload, utc_stamp())


def add_judgment(session, workstream_id: int, payload, actor) -> dict:
    workstream = _load_with_events(session, workstream_id)
    state = build_state(session, workstream)
    key = payload.idempotency_key
    event_payload = {
        "id": payload.id, "failed": payload.failed, "confirmed_at": payload.confirmed_at,
    }
    try:
        state.validate_judgment(_domain_event(domain.JUDGMENT, key, event_payload))
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    duplicate = state._find_event(domain.JUDGMENT, key) if key else None
    if duplicate is not None:
        return {"deduplicated": True, "event_seq": duplicate.seq, "workstream_id": workstream_id}
    event = TriageEvent(
        workstream_id=workstream_id, seq=_next_seq(session, workstream_id),
        kind=domain.JUDGMENT, idempotency_key=key,
        payload=event_payload, created_at=utc_stamp(),
    )
    session.add(event)
    session.flush()
    outcome = state.apply_judgment(domain.Event(
        event.seq, event.kind, event.idempotency_key, event.payload, event.created_at,
    ))
    workstream.state_snapshot = _snapshot(state)
    session.commit()
    return {**outcome, "workstream_id": workstream_id}


def schedule_batch(session, workstream_id: int, payload, actor) -> dict:
    workstream = _load_with_events(session, workstream_id)
    state = build_state(session, workstream)
    key = payload.idempotency_key
    position = domain.next_scheduling_position(state)
    event_payload = {"batch_id": payload.batch_id, "position": position}
    try:
        state.validate_scheduling(_domain_event(domain.SCHEDULING, key, event_payload))
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    duplicate = state._find_event(domain.SCHEDULING, key) if key else None
    if duplicate is not None:
        return {"deduplicated": True, "event_seq": duplicate.seq, "workstream_id": workstream_id}
    event = TriageEvent(
        workstream_id=workstream_id, seq=_next_seq(session, workstream_id),
        kind=domain.SCHEDULING, idempotency_key=key,
        payload=event_payload, created_at=utc_stamp(),
    )
    session.add(event)
    session.flush()
    outcome = state.apply_scheduling(domain.Event(
        event.seq, event.kind, event.idempotency_key, event.payload, event.created_at,
    ))
    workstream.state_snapshot = _snapshot(state)
    session.commit()
    return {**outcome, "workstream_id": workstream_id, "position": position}


def submit_review(session, workstream_id: int, payload, actor) -> dict:
    workstream = _load_with_events(session, workstream_id)
    state = build_state(session, workstream)
    key = payload.idempotency_key
    event_payload = {"decision": payload.decision, "note": payload.note or ""}
    try:
        state.validate_review(_domain_event(domain.REVIEW, key, event_payload))
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    duplicate = state._find_event(domain.REVIEW, key) if key else None
    if duplicate is not None:
        return {"deduplicated": True, "event_seq": duplicate.seq, "workstream_id": workstream_id}
    event = TriageEvent(
        workstream_id=workstream_id, seq=_next_seq(session, workstream_id),
        kind=domain.REVIEW, idempotency_key=key,
        payload=event_payload, created_at=utc_stamp(),
    )
    session.add(event)
    session.flush()
    state.apply_review(domain.Event(
        event.seq, event.kind, event.idempotency_key, event.payload, event.created_at,
    ))
    workstream.state_snapshot = _snapshot(state)
    session.commit()
    return {"deduplicated": False, "event_seq": event.seq, "workstream_id": workstream_id}


def detail_view(session, workstream_id: int, actor) -> dict:
    workstream, state = get_workstream(session, workstream_id, actor)
    return {
        **workstream_info(workstream, state),
        "events": [
            {
                "seq": event.seq, "kind": event.kind, "idempotency_key": event.idempotency_key,
                "payload": event.payload, "created_at": event.created_at,
            }
            for event in sorted(workstream.events, key=lambda item: item.seq)
        ],
        "tables": {
            "split_batches": state.split_rows(),
            "threshold_versions": state.threshold_rows(),
            "batch_audit": state.batch_audit_rows(),
            "retest_queue": state.queue_rows(),
            "feature_importance": state.feature_importance_rows(),
        },
        "warnings": state.warnings,
    }


def threshold_candidates(session, workstream_id: int, version: int, actor) -> list[dict]:
    _, state = get_workstream(session, workstream_id, actor)
    return state.threshold_candidate_rows(version)


def export_table(session, workstream_id: int, key: str, actor):
    workstream, state = get_workstream(session, workstream_id, actor)
    if not state.review or state.review["decision"] != Decision.APPROVED:
        raise HTTPException(409, "复核通过后可下载正式 CSV")
    tables = {
        "retest_queue": ("留出批次返测顺序单", state.queue_rows()),
        "batch_audit": ("留出批次误报与漏报对账", state.batch_audit_rows()),
        "threshold_versions": ("定阈版本依据", state.threshold_rows()),
    }
    if key not in tables:
        raise LookupError("找不到结果表")
    title, rows = tables[key]
    columns = list(rows[0].keys()) if rows else []
    return {"key": key, "title": title, "columns": columns, "labels": columns, "rows": rows}


def _snapshot(state: domain.TriageState) -> dict:
    return {
        "threshold_versions": [asdict(version) for version in state.threshold_versions],
        "review": state.review,
        "warnings": state.warnings,
    }
