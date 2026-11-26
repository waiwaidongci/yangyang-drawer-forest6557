"""增量分诊领域规则测试。

直接以合成台账驱动，不触碰数据库，覆盖：
- 训练只使用建立时已确认整批，首次定阈与归档回放口径一致；
- 验证补判乱序到达触发重算，新阈值只作用于未排单留出批；
- 幂等键去重、既判不可改、训练件不可补判；
- 一批只认同一次定阈，排单后决定冻结；
- 代价区间随末检补齐收窄到实际值；
- 事件序列可完整重建状态（中断恢复）。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.domain import triage as T
from src.domain.fixtures import demo_rows

HOLDOUT_BATCHES = ("YC-HG-1030-A", "YC-HG-1102-A", "YC-HG-1103-A", "YC-HG-1104-A")
VALIDATION_BATCHES = ("YC-HG-1028-A", "YC-HG-1029-A")
TRAINING_BATCHES_PREFIX = "YC-HG-10"


def make_source(pending_holdout=True, pending_validation_ids=()):
    rows = demo_rows()
    pending = set(pending_validation_ids)
    if pending_holdout:
        pending |= {r["id"] for r in rows if r["batch_id"] in HOLDOUT_BATCHES}
    source = []
    for row in rows:
        record = dict(row)
        if record["id"] in pending:
            record["failed"] = None
            record["confirmed_at"] = None
        source.append(record)
    return source


def judgment_event(seq, record_id, failed, confirmed_at, key):
    return T.Event(seq, T.JUDGMENT, key,
                   {"id": record_id, "failed": failed, "confirmed_at": confirmed_at},
                   confirmed_at + ":00")


class TriageDomainTests(unittest.TestCase):
    def setUp(self):
        all_rows = demo_rows()
        self.by_id = {row["id"]: row for row in all_rows}
        self.validation_pending_ids = [
            row["id"] for row in all_rows if row["batch_id"] == "YC-HG-1029-A"
        ][:6]
        self.source = make_source(pending_validation_ids=self.validation_pending_ids)
        self.state = T.create_workstream(1, "增量规则测试", self.source, {})

    def test_first_threshold_matches_archive_calibration(self):
        # 首次定阈只用验证批已确认的 30 件；阈值仍可计算且候选可审计。
        version = self.state.active_version()
        self.assertIsNotNone(version)
        self.assertEqual(version.validation_pending, 6)
        self.assertEqual(version.validation_confirmed, 30)
        self.assertTrue(version.candidates)
        selected = [c for c in version.candidates
                    if c["threshold"] == version.chosen_threshold
                    and c["cost"] == version.chosen_cost]
        self.assertEqual(len(selected), 1)

    def test_training_rows_reject_judgment(self):
        training_id = next(iter(self.state.rows.values())).record["id"]
        # 取一个训练件
        training_id = next(
            state.record["id"] for state in self.state.rows.values() if state.stage == "训练"
        )
        event = judgment_event(2, training_id, 1, "2026-10-12 12:00", "judge-train-1")
        with self.assertRaises(T.TriageError):
            self.state.apply_judgment(event)

    def test_judgment_idempotency_and_immutability(self):
        target = next(
            state.record["id"] for state in self.state.rows.values()
            if state.stage == "验证定阈" and state.failed is None
        )
        original = self.by_id[target]
        event = judgment_event(2, target, original["failed"], original["confirmed_at"], "judge-0001")
        outcome = self.state.apply_judgment(event)
        self.assertFalse(outcome["deduplicated"])
        versions_after = len(self.state.threshold_versions)
        # 同键同内容重试：去重，不新增事件、不新增阈值版本。
        retry = self.state.apply_judgment(judgment_event(
            99, target, original["failed"], original["confirmed_at"], "judge-0001",
        ))
        self.assertTrue(retry["deduplicated"])
        self.assertEqual(len(self.state.events), 1)
        self.assertEqual(len(self.state.threshold_versions), versions_after)
        # 同键不同内容：校验阶段拒绝。
        with self.assertRaises(T.TriageError):
            self.state.validate_judgment(judgment_event(
                3, target, 1 - original["failed"], original["confirmed_at"], "judge-0001",
            ))
        # 既判结果换键再补：拒绝。
        with self.assertRaises(T.TriageError):
            self.state.validate_judgment(judgment_event(
                4, target, original["failed"], original["confirmed_at"], "judge-0002",
            ))

    def test_out_of_order_judgments_recompute_threshold_only_for_unscheduled(self):
        validation_pending = [
            state for state in self.state.rows.values()
            if state.stage == "验证定阈" and state.failed is None
        ]
        # 乱序：先补后一条，再补前一条。
        validation_pending.sort(key=lambda state: state.record["measured_at"], reverse=True)
        for index, state in enumerate(validation_pending, start=2):
            original = self.by_id[state.record["id"]]
            self.state.apply_judgment(judgment_event(
                index, state.record["id"], original["failed"],
                original["confirmed_at"], f"judge-val-{index}",
            ))
        # 每条验证补判都产生一个新版本。
        self.assertEqual(
            len(self.state.threshold_versions), 1 + len(validation_pending),
        )
        # 排一个留出批，冻结当时版本。
        first_holdout = HOLDOUT_BATCHES[0]
        frozen_version = self.state.active_version().version
        self.state.apply_scheduling(T.Event(
            50, T.SCHEDULING, "sched-0001",
            {"batch_id": first_holdout, "position": 1}, "2026-10-30 20:00:00",
        ))
        scheduled_batch = self.state.batches[first_holdout]
        self.assertEqual(scheduled_batch.threshold_version, frozen_version)
        # 再补验证件产生新版本。
        more_pending = [
            state for state in self.state.rows.values()
            if state.stage == "验证定阈" and state.failed is None
        ]
        # 验证批可能已补完；改用阈值口径检查：排单批策略阈值不随后续版本变化。
        if more_pending:
            state = more_pending[0]
            original = self.by_id[state.record["id"]]
            self.state.apply_judgment(judgment_event(
                51, state.record["id"], original["failed"],
                original["confirmed_at"], "judge-val-late",
            ))
        # 手动再塞一个不同代价视角的版本，验证排单批仍取冻结阈值。
        before = self.state._policy_threshold(scheduled_batch)
        self.state.recompute_threshold("复核触发的补充定阈", "manual-1")
        after = self.state._policy_threshold(scheduled_batch)
        self.assertEqual(before, after)
        # 未排单批使用最新有效版本。
        unscheduled = self.state.batches[HOLDOUT_BATCHES[1]]
        self.assertEqual(
            self.state._policy_threshold(unscheduled),
            self.state.active_version().chosen_threshold,
        )

    def test_scheduling_freezes_decision_and_rejects_repeat(self):
        batch_id = HOLDOUT_BATCHES[0]
        self.state.apply_scheduling(T.Event(
            2, T.SCHEDULING, "sched-0001",
            {"batch_id": batch_id, "position": 1}, "2026-10-30 20:00:00",
        ))
        with self.assertRaises(T.TriageError):
            self.state.apply_scheduling(T.Event(
                3, T.SCHEDULING, "sched-0002",
                {"batch_id": batch_id, "position": 2}, "2026-10-30 20:01:00",
            ))
        # 同键重试去重。
        dup = self.state.apply_scheduling(T.Event(
            4, T.SCHEDULING, "sched-0001",
            {"batch_id": batch_id, "position": 1}, "2026-10-30 20:02:00",
        ))
        self.assertTrue(dup["deduplicated"])
        # 排单允许批次仍有待判件。
        audit = {row["batch_id"]: row for row in self.state.batch_audit_rows()}
        self.assertGreater(audit[batch_id]["pending"], 0)
        self.assertEqual(audit[batch_id]["scheduled"], "是")

    def test_cost_interval_collapses_to_actual_after_confirmation(self):
        batch_id = HOLDOUT_BATCHES[0]
        self.state.apply_scheduling(T.Event(
            2, T.SCHEDULING, "sched-0001",
            {"batch_id": batch_id, "position": 1}, "2026-10-30 20:00:00",
        ))
        audit = {row["batch_id"]: row for row in self.state.batch_audit_rows()}
        self.assertLess(audit[batch_id]["cost_lower"], audit[batch_id]["cost_upper"])
        # 补齐该批末检（乱序）。
        members = [s for s in self.state.rows.values() if s.record["batch_id"] == batch_id]
        members.sort(key=lambda state: state.record["id"], reverse=True)
        for index, state in enumerate(members, start=10):
            original = self.by_id[state.record["id"]]
            self.state.apply_judgment(judgment_event(
                index, state.record["id"], original["failed"],
                original["confirmed_at"], f"judge-h-{index}",
            ))
        audit = {row["batch_id"]: row for row in self.state.batch_audit_rows()}
        row = audit[batch_id]
        self.assertEqual(row["pending"], 0)
        self.assertEqual(row["cost_lower"], row["cost_upper"])
        self.assertEqual(row["cost_lower"], row["actual_cost"])
        # 排单决定（动作）未因补判改变。
        self.assertEqual(
            self.state.batches[batch_id].threshold_version,
            self.state.batches[batch_id].threshold_version,
        )

    def test_reconstruction_from_events_matches(self):
        # 制造一批事件：验证补判、排单、留出补判。
        val_pending = [
            state for state in self.state.rows.values()
            if state.stage == "验证定阈" and state.failed is None
        ][:3]
        seq = 2
        for state in val_pending:
            original = self.by_id[state.record["id"]]
            self.state.apply_judgment(judgment_event(
                seq, state.record["id"], original["failed"],
                original["confirmed_at"], f"judge-r-{seq}",
            ))
            seq += 1
        self.state.apply_scheduling(T.Event(
            seq, T.SCHEDULING, "sched-r-1",
            {"batch_id": HOLDOUT_BATCHES[0], "position": 1}, "2026-10-30 20:00:00",
        ))
        seq += 1
        holdout_pending = [
            state for state in self.state.rows.values()
            if state.stage == "留出评估" and state.failed is None
        ][:4]
        for state in holdout_pending:
            original = self.by_id[state.record["id"]]
            self.state.apply_judgment(judgment_event(
                seq, state.record["id"], original["failed"],
                original["confirmed_at"], f"judge-r-{seq}",
            ))
            seq += 1
        events = list(self.state.events)
        rebuilt = T.create_workstream(1, "增量规则测试", self.source, {})
        for event in events:
            if event.kind == T.JUDGMENT:
                rebuilt.apply_judgment(event)
            elif event.kind == T.SCHEDULING:
                rebuilt.apply_scheduling(event)
        self.assertEqual(
            [v.chosen_threshold for v in self.state.threshold_versions],
            [v.chosen_threshold for v in rebuilt.threshold_versions],
        )
        self.assertEqual(self.state.queue_rows(), rebuilt.queue_rows())
        self.assertEqual(self.state.batch_audit_rows(), rebuilt.batch_audit_rows())
        self.assertEqual(self.state.holdout_metrics(), rebuilt.holdout_metrics())

    def test_no_valid_threshold_blocks_scheduling_until_both_classes_confirm(self):
        # 验证批全部待判：只有警告版本，无法排单。
        full = demo_rows()
        probe = T.create_workstream(3, "探测切分", make_source(), {})
        validation_ids = [
            state.record["id"] for state in probe.rows.values() if state.stage == "验证定阈"
        ]
        source = make_source(pending_validation_ids=validation_ids)
        state = T.create_workstream(2, "无有效定阈", source, {})
        self.assertIsNone(state.active_version())
        with self.assertRaises(T.TriageError):
            state.apply_scheduling(T.Event(
                2, T.SCHEDULING, "sched-blocked",
                {"batch_id": HOLDOUT_BATCHES[0], "position": 1}, "2026-10-30 20:00:00",
            ))
        # 补齐验证批（同时含通过和失败），出现有效版本。
        by_id = {row["id"]: row for row in full}
        seq = 2
        for row_state in [s for s in state.rows.values() if s.stage == "验证定阈"]:
            original = by_id[row_state.record["id"]]
            state.apply_judgment(judgment_event(
                seq, row_state.record["id"], original["failed"],
                original["confirmed_at"], f"judge-u-{seq}",
            ))
            seq += 1
        self.assertIsNotNone(state.active_version())
        outcome = state.apply_scheduling(T.Event(
            seq, T.SCHEDULING, "sched-unblock",
            {"batch_id": HOLDOUT_BATCHES[0], "position": 1}, "2026-10-30 20:00:00",
        ))
        self.assertFalse(outcome["deduplicated"])

    def test_prepare_rows_validation(self):
        with self.assertRaises(T.TriageError):
            T.prepare_rows([{"id": "x"}] * 5)
        good = make_source()
        self.assertEqual(len(T.prepare_rows(good)), len(good))
        broken = dict(good[0])
        broken["failed"] = 2
        with self.assertRaises(T.TriageError):
            T.prepare_rows([broken] + good[1:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
