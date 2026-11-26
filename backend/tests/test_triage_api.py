"""增量分诊 API 与事件持久化集成测试（内存 SQLite，无需外部服务）。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import src.config.database as database

# 必须在导入任何模型之前替换 engine，模型注册到 database.Base.metadata，
# 建表时使用当前绑定的 engine。
_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
database.engine = _engine
database.Session.configure(bind=_engine)

from src.constants.workflow import Role  # noqa: E402
from src.domain.fixtures import demo_rows  # noqa: E402
from src.main import app  # noqa: E402
from src.models.account import Account  # noqa: E402
from src.services.catalog import add_dataset  # noqa: E402
from src.utils.security import hash_password, token_for  # noqa: E402


class TriageApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from migrations.bootstrap import bootstrap

        bootstrap()
        with database.Session() as session:
            accounts = {}
            for role in Role:
                account = Account(
                    email=f"{role.value}@t.local", role=role,
                    password_hash=hash_password("pw-12345"),
                )
                session.add(account)
                session.flush()
                accounts[role] = account
            cls.dataset = add_dataset(session, "demo-triage.csv", demo_rows(), accounts[Role.ANALYST].id)
            session.commit()
            cls.dataset_id = cls.dataset.id
            cls.tokens = {role: token_for(accounts[role].id) for role in Role}
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        _engine.dispose()

    def _headers(self, role):
        return {"Authorization": f"Bearer {self.tokens[role]}"}

    def test_full_incremental_workflow_and_idempotency(self):
        client = self.client
        rows = demo_rows()
        holdout_batches = sorted({r["batch_id"] for r in rows})[-4:]
        holdout_ids = [r["id"] for r in rows if r["batch_id"] in holdout_batches]
        validation_batches = sorted({r["batch_id"] for r in rows})[11:14]
        pending_val = [r["id"] for r in rows if r["batch_id"] in validation_batches][:6]

        # viewer 不能建立工作流。
        denied = client.post(
            "/api/triage",
            headers=self._headers(Role.VIEWER),
            json={"label": "权限测试", "dataset_id": self.dataset_id},
        )
        self.assertEqual(denied.status_code, 403)

        created = client.post(
            "/api/triage",
            headers=self._headers(Role.ANALYST),
            json={
                "label": "增量分诊-集成",
                "dataset_id": self.dataset_id,
                "pending_ids": holdout_ids + pending_val,
            },
        )
        self.assertEqual(created.status_code, 201, created.text)
        workstream_id = created.json()["id"]
        self.assertGreaterEqual(created.json()["threshold_versions"], 1)

        # 待判件 id 不存在返回 422。
        bad = client.post(
            "/api/triage",
            headers=self._headers(Role.ANALYST),
            json={"label": "坏编号", "dataset_id": self.dataset_id, "pending_ids": ["NOPE-01"]},
        )
        self.assertEqual(bad.status_code, 422)

        by_id = {r["id"]: r for r in rows}

        # 乱序补判验证批：先发后半条，再发前半条。
        for record_id in reversed(pending_val):
            original = by_id[record_id]
            payload = {
                "id": record_id, "failed": original["failed"],
                "confirmed_at": original["confirmed_at"],
                "idempotency_key": f"judge-{record_id}",
            }
            first = client.post(
                f"/api/triage/{workstream_id}/judgments",
                headers=self._headers(Role.ANALYST), json=payload,
            )
            self.assertEqual(first.status_code, 201, first.text)
            self.assertFalse(first.json()["deduplicated"])
            # 中断恢复：同键重试不新增记录。
            retry = client.post(
                f"/api/triage/{workstream_id}/judgments",
                headers=self._headers(Role.ANALYST), json=payload,
            )
            self.assertEqual(retry.status_code, 201)
            self.assertTrue(retry.json()["deduplicated"])
            self.assertEqual(retry.json()["event_seq"], first.json()["event_seq"])

        detail = client.get(
            f"/api/triage/{workstream_id}", headers=self._headers(Role.ANALYST),
        ).json()
        version_count = detail["metrics"]["active_threshold_version"]
        self.assertIsNotNone(version_count)

        # 同键不同内容冲突。
        conflict = client.post(
            f"/api/triage/{workstream_id}/judgments",
            headers=self._headers(Role.ANALYST),
            json={
                "id": pending_val[0], "failed": 1 - by_id[pending_val[0]]["failed"],
                "confirmed_at": by_id[pending_val[0]]["confirmed_at"],
                "idempotency_key": f"judge-{pending_val[0]}",
            },
        )
        self.assertEqual(conflict.status_code, 422)

        # 排第一个留出批（仍有待判件也允许，决定先冻结）。
        first_batch = holdout_batches[0]
        sched_payload = {"batch_id": first_batch, "idempotency_key": "sched-batch-1"}
        sched = client.post(
            f"/api/triage/{workstream_id}/schedules",
            headers=self._headers(Role.ANALYST), json=sched_payload,
        )
        self.assertEqual(sched.status_code, 201, sched.text)
        frozen_version = sched.json()["threshold_version"]
        # 同批重复排单（新键）被拒；同键重试去重。
        repeat = client.post(
            f"/api/triage/{workstream_id}/schedules",
            headers=self._headers(Role.ANALYST),
            json={"batch_id": first_batch, "idempotency_key": "sched-batch-again"},
        )
        self.assertEqual(repeat.status_code, 422)
        dup = client.post(
            f"/api/triage/{workstream_id}/schedules",
            headers=self._headers(Role.ANALYST), json=sched_payload,
        )
        self.assertTrue(dup.json()["deduplicated"])

        # 排单后再补验证件产生新版本，排单批阈值保持冻结。
        late_val = [
            r["id"] for r in rows
            if r["batch_id"] in validation_batches and r["id"] not in pending_val
        ][:2]
        for record_id in late_val:
            original = by_id[record_id]
            client.post(
                f"/api/triage/{workstream_id}/judgments",
                headers=self._headers(Role.ANALYST),
                json={
                    "id": record_id, "failed": original["failed"],
                    "confirmed_at": original["confirmed_at"],
                    "idempotency_key": f"judge-late-{record_id}",
                },
            )
        detail = client.get(
            f"/api/triage/{workstream_id}", headers=self._headers(Role.ANALYST),
        ).json()
        batch_audit = {row["batch_id"]: row for row in detail["tables"]["batch_audit"]}
        self.assertEqual(batch_audit[first_batch]["threshold_version"], frozen_version)

        # 补齐已排单批末检，代价区间收窄到实际值。
        members = [r for r in rows if r["batch_id"] == first_batch]
        self.assertGreater(batch_audit[first_batch]["cost_upper"], batch_audit[first_batch]["cost_lower"])
        for original in reversed(members):
            client.post(
                f"/api/triage/{workstream_id}/judgments",
                headers=self._headers(Role.ANALYST),
                json={
                    "id": original["id"], "failed": original["failed"],
                    "confirmed_at": original["confirmed_at"],
                    "idempotency_key": f"judge-h-{original['id']}",
                },
            )
        detail = client.get(
            f"/api/triage/{workstream_id}", headers=self._headers(Role.ANALYST),
        ).json()
        row = {r["batch_id"]: r for r in detail["tables"]["batch_audit"]}[first_batch]
        self.assertEqual(row["pending"], 0)
        self.assertEqual(row["cost_lower"], row["cost_upper"])
        self.assertEqual(row["cost_lower"], row["actual_cost"])
        # 顺序单冻结版本不变，且按得分降序。
        queue = detail["tables"]["retest_queue"]
        self.assertEqual(len(queue), len(members))
        self.assertTrue(all(queue[i]["score"] >= queue[i + 1]["score"] for i in range(len(queue) - 1)))
        self.assertTrue(all(item["threshold_version"] == frozen_version for item in queue))

        # 未批准前 viewer 不可见、不可导出。
        self.assertEqual(
            client.get(f"/api/triage/{workstream_id}", headers=self._headers(Role.VIEWER)).status_code,
            404,
        )
        self.assertEqual(
            client.get(
                f"/api/triage/{workstream_id}/export/retest_queue",
                headers=self._headers(Role.ANALYST),
            ).status_code,
            409,
        )

        # analyst 不能复核；admin 批准（带幂等键，重试不新增）。
        review_payload = {
            "decision": "approved", "note": "确认增量定阈依据，批准交接",
            "idempotency_key": "review-0001",
        }
        self.assertEqual(
            client.post(
                f"/api/triage/{workstream_id}/review",
                headers=self._headers(Role.ANALYST), json=review_payload,
            ).status_code,
            403,
        )
        approved = client.post(
            f"/api/triage/{workstream_id}/review",
            headers=self._headers(Role.ADMIN), json=review_payload,
        )
        self.assertEqual(approved.status_code, 201, approved.text)
        review_retry = client.post(
            f"/api/triage/{workstream_id}/review",
            headers=self._headers(Role.ADMIN), json=review_payload,
        )
        self.assertTrue(review_retry.json()["deduplicated"])

        # 批准后 viewer 可见并可下载 CSV。
        visible = client.get(
            f"/api/triage/{workstream_id}", headers=self._headers(Role.VIEWER),
        )
        self.assertEqual(visible.status_code, 200)
        exported = client.get(
            f"/api/triage/{workstream_id}/export/retest_queue",
            headers=self._headers(Role.VIEWER),
        )
        self.assertEqual(exported.status_code, 200)
        self.assertIn("text/csv", exported.headers["content-type"])
        self.assertEqual(len(exported.text.strip().splitlines()), len(members) + 1)

        # 定阈依据：每个版本可查候选。
        versions = visible.json()["tables"]["threshold_versions"]
        self.assertTrue(versions)
        candidates = client.get(
            f"/api/triage/{workstream_id}/thresholds/{versions[0]['version']}",
            headers=self._headers(Role.VIEWER),
        )
        self.assertEqual(candidates.status_code, 200)
        self.assertTrue(candidates.json()["candidates"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
