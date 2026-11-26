"""增量返测分诊：末检未判齐时的事件化定阈、排单与代价区间。

与归档回放（domain.project）不同，本模块允许验证定阈批和留出评估批中存在
末检待判件（failed 为 None）。工作流以事件为事实来源，状态可由初始台账与
事件序列完整重建：

* ``created``      —— 工作流建立，训练段冻结、森林只拟合一次；
* ``judgment``     —— 末检补判到达（允许乱序），同键重试不新增记录；
* ``scheduling``   —— 交班排单，按整批冻结当时的定阈版本与返测顺序；
* ``review``       —— 复核负责人批准或退回。

定阈只使用验证批中已确认的样本；重算得到的新阈值版本仅对未排单的留出批
生效。已排单批次永久保留排单时的阈值和返测决定，后续补判只把该批的代价
区间收窄到实际值。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from sklearn.metrics import roc_auc_score

from .chronology import _cut_positions
from .forest import FEATURES, failure_scores, fit_forest
from .policy import Costs, parse_costs, select_threshold
from .records import LOCAL_MINUTE, MEASUREMENTS, _measurement

PENDING: None = None
JUDGMENT = "judgment"
SCHEDULING = "scheduling"
REVIEW = "review"
CREATED = "created"


def _outcome(flagged: bool, failed: int) -> str:
    if flagged:
        return "tp" if failed else "fp"
    return "fn" if failed else "tn"


OUTCOME_LABELS = {
    "tp": "正确优先返测（TP）",
    "fp": "误报（FP）",
    "tn": "正确常规末检（TN）",
    "fn": "漏报（FN）",
}


@dataclass(frozen=True)
class ThresholdVersion:
    """一次定阈计算的完整依据。"""

    version: int
    reason: str
    event_key: str | None
    chosen_threshold: float | None
    chosen_cost: int | None
    validation_confirmed: int
    validation_pending: int
    candidates: tuple[dict, ...]
    warning: str | None = None


@dataclass
class RowState:
    record: dict
    stage: str
    score: float
    failed: int | None
    confirmed_at: str | None
    judgment_event: str | None = None


@dataclass
class BatchState:
    batch_id: str
    stage: str
    first_measurement: str
    last_confirmation: str | None
    rows: list[RowState] = field(default_factory=list)
    # 排单后冻结的阈值版本号与排队位次；未排单为 None。
    threshold_version: int | None = None
    queue_position: int | None = None
    scheduling_event: str | None = None


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    key: str | None
    payload: dict
    created_at: str


class TriageError(ValueError):
    """增量分诊业务校验失败。"""


def _check_record_schema(row: dict, line: int) -> dict:
    required = {"id", "batch_id", "measured_at", *MEASUREMENTS.keys()}
    missing = sorted(required - row.keys())
    if missing:
        raise TriageError(f"第 {line} 行缺少列 {', '.join(missing)}")
    if not isinstance(row["id"], str) or not row["id"].strip():
        raise TriageError(f"第 {line} 行滑轨总成编号不能为空")
    if not isinstance(row["batch_id"], str) or not row["batch_id"].strip():
        raise TriageError(f"第 {line} 行装配批次不能为空")
    measured_at = row["measured_at"]
    if not isinstance(measured_at, str) or not measured_at.strip():
        raise TriageError(f"第 {line} 行装配测读时间不能为空")
    measured_at = measured_at.strip()
    measurements = {
        key: _measurement(row[key], *spec) for key, spec in MEASUREMENTS.items()
    }
    record = {
        "id": row["id"].strip(), "batch_id": row["batch_id"].strip(),
        "measured_at": measured_at, **measurements,
    }
    failed = row.get("failed", PENDING)
    if failed is not None and failed not in (0, 1):
        raise TriageError(f"第 {line} 行末检失败须为 0、1 或留空待判")
    record["failed"] = failed
    confirmed_at = row.get("confirmed_at")
    if confirmed_at is not None:
        if not isinstance(confirmed_at, str) or not LOCAL_MINUTE.fullmatch(confirmed_at.strip()):
            raise TriageError(f"第 {line} 行末检确认时间须为有效工厂本地时间（YYYY-MM-DD HH:MM）")
        confirmed_at = confirmed_at.strip()
        try:
            datetime.fromisoformat(confirmed_at)
        except ValueError as exc:
            raise TriageError(f"第 {line} 行末检确认时间须为有效工厂本地时间") from exc
        if confirmed_at < measured_at:
            raise TriageError("第 {line} 行末检确认时间早于装配测读时间")
    record["confirmed_at"] = confirmed_at
    return record


def prepare_rows(rows: list[dict]) -> list[dict]:
    """校验增量台账：测量字段必填，末检结果允许缺失。"""

    if not isinstance(rows, list) or len(rows) < 24:
        raise TriageError("至少需要 24 条装配记录才能建立增量分诊")
    records: list[dict] = []
    seen: set[str] = set()
    batches: set[str] = set()
    for line, row in enumerate(rows, 2):
        if not isinstance(row, dict):
            raise TriageError(f"第 {line} 行须为一条装配记录")
        record = _check_record_schema(row, line)
        if record["id"] in seen:
            raise TriageError(f"滑轨总成编号重复（{record['id']}）")
        seen.add(record["id"])
        batches.add(record["batch_id"])
        records.append(record)
    if len(batches) < 6:
        raise TriageError("至少需要 6 个不同装配批次")
    return records


def _partition(records: list[dict]):
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["batch_id"]].append(record)
    ordered = []
    for batch_id, members in grouped.items():
        confirmed = [row["confirmed_at"] for row in members if row["confirmed_at"]]
        ordered.append({
            "batch_id": batch_id,
            "first_measurement": min(row["measured_at"] for row in members),
            "last_confirmation": max(confirmed) if confirmed else None,
            "rows": tuple(sorted(members, key=lambda row: (row["measured_at"], row["id"]))),
        })
    ordered.sort(key=lambda item: (item["first_measurement"], item["batch_id"]))
    first_cut, second_cut = _cut_positions([
        type("CutBatch", (), {
            "first_measurement": item["first_measurement"],
            "last_confirmation": item["last_confirmation"] or "",
            "rows": item["rows"],
        })
        for item in ordered
    ])
    stages = (
        ("训练", ordered[:first_cut]),
        ("验证定阈", ordered[first_cut:second_cut]),
        ("留出评估", ordered[second_cut:]),
    )
    return stages


def _candidate_dicts(decision) -> tuple[dict, ...]:
    return tuple(
        {
            "threshold": point.threshold,
            "rule": "全不标记" if point.threshold is None else "得分大于或等于阈值",
            "tp": point.true_positive,
            "fp": point.false_positive,
            "tn": point.true_negative,
            "fn": point.false_negative,
            "cost": point.cost,
        }
        for point in decision.candidates
    )


@dataclass
class TriageState:
    workstream_id: int
    label: str
    costs: Costs
    records: list[dict]
    stages: tuple
    forest: object
    scores: dict[str, float]
    rows: dict[str, RowState]
    batches: dict[str, BatchState]
    batch_order: list[str]
    threshold_versions: list[ThresholdVersion]
    events: list[Event] = field(default_factory=list)
    review: dict | None = None
    warnings: list[str] = field(default_factory=list)

    # -- 定阈 ----------------------------------------------------------------

    def _validation_confirmed(self) -> list[RowState]:
        return [
            state for state in self.rows.values()
            if state.stage == "验证定阈" and state.failed is not None
        ]

    def recompute_threshold(self, reason: str, event_key: str | None) -> ThresholdVersion | None:
        """用验证批已确认件重算阈值，返回新版本；样本不足时只记警告。"""

        confirmed = self._validation_confirmed()
        pending = sum(
            1 for state in self.rows.values()
            if state.stage == "验证定阈" and state.failed is None
        )
        labels = {state.failed for state in confirmed}
        version_no = len(self.threshold_versions) + 1
        if labels != {0, 1}:
            warning = "验证定阈批已确认样本未同时包含通过与失败件，暂不产生新阈值，未排单留出批沿用既有定阈。"
            self.warnings.append(warning)
            version = ThresholdVersion(
                version_no, reason, event_key, None, None, len(confirmed), pending, (), warning,
            )
            self.threshold_versions.append(version)
            return version
        decision = select_threshold(
            [state.score for state in confirmed],
            [state.failed for state in confirmed],
            self.costs,
        )
        version = ThresholdVersion(
            version_no,
            reason,
            event_key,
            decision.chosen.threshold,
            decision.chosen.cost,
            len(confirmed),
            pending,
            _candidate_dicts(decision),
        )
        self.threshold_versions.append(version)
        return version

    def active_version(self) -> ThresholdVersion | None:
        """最新一个有效定阈版本（warning 为 None 表示阈值可用），供未排单留出批使用。"""

        for version in reversed(self.threshold_versions):
            if version.warning is None:
                return version
        return None

    def _policy_threshold(self, batch: BatchState) -> float | None:
        if batch.threshold_version is not None:
            return self.threshold_versions[batch.threshold_version - 1].chosen_threshold
        version = self.active_version()
        return version.chosen_threshold if version else None

    def _flags(self, state: RowState, batch: BatchState) -> bool:
        threshold = self._policy_threshold(batch)
        return threshold is not None and state.score >= threshold

    # -- 事件应用 ------------------------------------------------------------

    def validate_judgment(self, event: Event) -> None:
        """纯校验，不修改状态。供服务层在持久化事件前拦截非法补判。"""

        payload = event.payload
        record_id = payload["id"]
        state = self.rows.get(record_id)
        if state is None:
            raise TriageError(f"总成 {record_id} 不属于本工作流台账")
        if state.stage == "训练":
            raise TriageError("训练段只使用工作流建立时已确认的整批，训练件不接受补判")
        duplicate = self._find_event(JUDGMENT, event.key) if event.key else None
        if duplicate is not None:
            if duplicate.payload["failed"] != payload["failed"] or duplicate.payload["id"] != record_id:
                raise TriageError("幂等键已被不同补判内容占用")
            return
        if state.failed is not None:
            raise TriageError(f"总成 {record_id} 已完成末检确认，补判不得改写既判结果")

    def apply_judgment(self, event: Event) -> dict:
        payload = event.payload
        record_id = payload["id"]
        state = self.rows[record_id]
        duplicate = self._find_event(JUDGMENT, event.key) if event.key else None
        if duplicate is not None:
            return {"deduplicated": True, "event_seq": duplicate.seq}
        self.validate_judgment(event)
        state.failed = payload["failed"]
        state.confirmed_at = payload["confirmed_at"]
        state.judgment_event = event.key
        batch = self.batches[state.record["batch_id"]]
        confirmed_times = [
            row.confirmed_at for row in batch.rows if row.confirmed_at is not None
        ]
        batch.last_confirmation = max(confirmed_times) if confirmed_times else None
        self.events.append(event)
        if state.stage == "验证定阈":
            self.recompute_threshold("验证批补判到达后重算定阈", event.key)
        return {"deduplicated": False, "event_seq": event.seq}

    def validate_scheduling(self, event: Event) -> None:
        """纯校验，不修改状态。"""

        batch_id = event.payload["batch_id"]
        batch = self.batches.get(batch_id)
        if batch is None or batch.stage != "留出评估":
            raise TriageError("排单只能针对留出评估批")
        duplicate = self._find_event(SCHEDULING, event.key) if event.key else None
        if duplicate is not None:
            if duplicate.payload["batch_id"] != batch_id:
                raise TriageError("幂等键已被其他排单占用")
            return
        if batch.scheduling_event is not None:
            raise TriageError(f"批次 {batch_id} 已排单，一批只认同一次定阈，不得重复排单")
        if self.active_version() is None:
            raise TriageError("尚无有效定阈，无法排单")

    def apply_scheduling(self, event: Event) -> dict:
        payload = event.payload
        batch_id = payload["batch_id"]
        batch = self.batches[batch_id]
        duplicate = self._find_event(SCHEDULING, event.key) if event.key else None
        if duplicate is not None:
            return {"deduplicated": True, "event_seq": duplicate.seq}
        self.validate_scheduling(event)
        version = self.active_version()
        batch.threshold_version = version.version
        batch.queue_position = payload["position"]
        batch.scheduling_event = event.key
        self.events.append(event)
        return {"deduplicated": False, "event_seq": event.seq, "threshold_version": version.version}

    def validate_review(self, event: Event) -> None:
        duplicate = self._find_event(REVIEW, event.key) if event.key else None
        if duplicate is not None and duplicate.payload != event.payload:
            raise TriageError("幂等键已被不同复核意见占用")

    def apply_review(self, event: Event) -> None:
        duplicate = self._find_event(REVIEW, event.key) if event.key else None
        if duplicate is not None:
            return
        self.validate_review(event)
        self.review = {
            "decision": event.payload["decision"],
            "note": event.payload.get("note", ""),
            "event_key": event.key,
            "created_at": event.created_at,
        }
        self.events.append(event)

    def _find_event(self, kind: str, key: str) -> Event | None:
        for event in self.events:
            if event.kind == kind and event.key == key:
                return event
        return None

    # -- 视图 ----------------------------------------------------------------

    def queue_rows(self) -> list[dict]:
        scheduled = [batch for batch in self.batches.values() if batch.scheduling_event is not None]
        scheduled.sort(key=lambda batch: (batch.queue_position, batch.batch_id))
        rows = []
        rank = 0
        for batch in scheduled:
            members = sorted(
                batch.rows,
                key=lambda state: (-state.score, state.record["measured_at"], state.record["id"]),
            )
            for state in members:
                rank += 1
                flagged = self._flags(state, batch)
                rows.append({
                    "rank": rank,
                    "id": state.record["id"],
                    "batch_id": batch.batch_id,
                    "measured_at": state.record["measured_at"],
                    "score": state.score,
                    "threshold_version": batch.threshold_version,
                    "action": "优先返测" if flagged else "常规末检",
                    "failed": state.failed,
                    "actual": "待判" if state.failed is None else ("失败" if state.failed else "通过"),
                    "confirmed_at": state.confirmed_at,
                })
        return rows

    def batch_audit_rows(self) -> list[dict]:
        result = []
        for batch_id in self.batch_order:
            batch = self.batches[batch_id]
            confirmed_counts = Counter()
            pending_count = 0
            pending_flagged = 0
            for state in batch.rows:
                flagged = self._flags(state, batch)
                if state.failed is None:
                    pending_count += 1
                    pending_flagged += int(flagged)
                else:
                    confirmed_counts[_outcome(flagged, state.failed)] += 1
            actual_fn = confirmed_counts["fn"]
            actual_fp = confirmed_counts["fp"]
            actual_cost = self.costs.total(actual_fn, actual_fp)
            # 待判件的代价上下界：最坏按其当前分诊动作对应的最贵错误计。
            worst_extra = 0
            best_extra = 0
            for state in batch.rows:
                if state.failed is not None:
                    continue
                flagged = self._flags(state, batch)
                worst_extra += self.costs.unnecessary_retest if flagged else self.costs.missed_failure
            result.append({
                "batch_id": batch_id,
                "stage": batch.stage,
                "count": len(batch.rows),
                "confirmed": len(batch.rows) - pending_count,
                "pending": pending_count,
                "threshold_version": batch.threshold_version,
                "scheduled": "是" if batch.scheduling_event else "否",
                "tp": confirmed_counts["tp"],
                "fp": confirmed_counts["fp"],
                "tn": confirmed_counts["tn"],
                "fn": confirmed_counts["fn"],
                "actual_cost": actual_cost,
                "cost_lower": actual_cost + best_extra,
                "cost_upper": actual_cost + worst_extra,
                "last_confirmation": batch.last_confirmation,
            })
        return result

    def threshold_rows(self) -> list[dict]:
        rows = []
        for version in self.threshold_versions:
            rows.append({
                "version": version.version,
                "reason": version.reason,
                "event_key": version.event_key,
                "threshold": version.chosen_threshold,
                "validation_confirmed": version.validation_confirmed,
                "validation_pending": version.validation_pending,
                "validation_cost": version.chosen_cost,
                "warning": version.warning,
                "candidate_count": len(version.candidates),
            })
        return rows

    def threshold_candidate_rows(self, version: int) -> list[dict]:
        if not 1 <= version <= len(self.threshold_versions):
            raise TriageError("阈值版本不存在")
        current = self.threshold_versions[version - 1]
        return [
            {
                "version": version,
                "selected": "是"
                if candidate["threshold"] == current.chosen_threshold
                and candidate["cost"] == current.chosen_cost
                else "否",
                **candidate,
            }
            for candidate in current.candidates
        ]

    def holdout_metrics(self) -> dict:
        holdout_states = [state for state in self.rows.values() if state.stage == "留出评估"]
        confirmed = [state for state in holdout_states if state.failed is not None]
        labels = [state.failed for state in confirmed]
        scores = [state.score for state in confirmed]
        auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
        tp = fp = tn = fn = 0
        worst_pending = 0
        for state in holdout_states:
            batch = self.batches[state.record["batch_id"]]
            flagged = self._flags(state, batch)
            if state.failed is None:
                worst_pending += self.costs.unnecessary_retest if flagged else self.costs.missed_failure
                continue
            outcome = _outcome(flagged, state.failed)
            tp += outcome == "tp"
            fp += outcome == "fp"
            tn += outcome == "tn"
            fn += outcome == "fn"
        actual = self.costs.total(fn, fp)
        failures = tp + fn
        return {
            "holdout_total": len(holdout_states),
            "holdout_confirmed": len(confirmed),
            "holdout_pending": len(holdout_states) - len(confirmed),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "recall": round(tp / failures, 4) if failures else None,
            "roc_auc": round(auc, 4) if auc is not None else None,
            "actual_cost": actual,
            "cost_lower": actual,
            "cost_upper": actual + worst_pending,
            "active_threshold_version": self.active_version().version if self.active_version() else None,
        }

    def feature_importance_rows(self) -> list[dict]:
        contributions = sorted(
            zip(FEATURES, self.forest.feature_importances_.tolist(), strict=True),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [{"feature": FEATURES[key], "importance": importance} for key, importance in contributions]

    def split_rows(self) -> list[dict]:
        rows = []
        for stage_name, batches in self.stages:
            for item in batches:
                members = [self.rows[row["id"]] for row in item["rows"]]
                rows.append({
                    "stage": stage_name,
                    "batch_id": item["batch_id"],
                    "first_measurement": item["first_measurement"],
                    "last_confirmation": item["last_confirmation"],
                    "count": len(members),
                    "confirmed": sum(state.failed is not None for state in members),
                    "pending": sum(state.failed is None for state in members),
                })
        return rows


def create_workstream(workstream_id: int, label: str, rows: list[dict], settings: dict) -> TriageState:
    """建立工作流：校验、整批时间切分、训练冻结并完成首次定阈。"""

    costs = parse_costs(settings)
    records = prepare_rows(rows)
    stages = _partition(records)
    training_records = [row for item in stages[0][1] for row in item["rows"]]
    if any(row["failed"] is None for row in training_records):
        raise TriageError("训练段必须全部为已确认整批，请核对末检回填后再建立工作流")
    forest = fit_forest(training_records)
    all_records = [row for _, batches in stages for item in batches for row in item["rows"]]
    score_map = dict(zip(
        [row["id"] for row in all_records], failure_scores(forest, all_records), strict=True,
    ))
    row_states = {}
    batch_states = {}
    batch_order = []
    for stage_name, batches in stages:
        for item in batches:
            batch_id = item["batch_id"]
            batch_order.append(batch_id)
            batch = BatchState(
                batch_id=batch_id, stage=stage_name,
                first_measurement=item["first_measurement"],
                last_confirmation=item["last_confirmation"],
            )
            for row in item["rows"]:
                state = RowState(
                    record=row, stage=stage_name, score=score_map[row["id"]],
                    failed=row["failed"], confirmed_at=row["confirmed_at"],
                )
                row_states[row["id"]] = state
                batch.rows.append(state)
            batch_states[batch_id] = batch
    state = TriageState(
        workstream_id=workstream_id, label=label, costs=costs, records=records,
        stages=stages, forest=forest, scores=score_map, rows=row_states,
        batches=batch_states, batch_order=batch_order, threshold_versions=[],
    )
    state.recompute_threshold("工作流建立时按验证批已确认件首次定阈", None)
    return state


def next_scheduling_position(state: TriageState) -> int:
    return 1 + sum(1 for batch in state.batches.values() if batch.scheduling_event is not None)
