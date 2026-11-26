from sqlalchemy import JSON, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.config.database import Base


class TriageWorkstream(Base):
    """增量分诊工作流。事件为事实来源，state_snapshot 是可重建的缓存。"""

    __tablename__ = "triage_workstreams"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(140))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    created_at: Mapped[str] = mapped_column(String(32))
    settings: Mapped[dict] = mapped_column(JSON)
    source_rows: Mapped[list] = mapped_column(JSON)
    state_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text)

    events: Mapped[list["TriageEvent"]] = relationship(
        back_populates="workstream", order_by="TriageEvent.seq",
    )


class TriageEvent(Base):
    """工作流事件日志。幂等键保证中断恢复时同一补判/排单重试不新增记录。"""

    __tablename__ = "triage_events"
    __table_args__ = (
        UniqueConstraint("workstream_id", "kind", "idempotency_key", name="uq_triage_event_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workstream_id: Mapped[int] = mapped_column(ForeignKey("triage_workstreams.id"), index=True)
    seq: Mapped[int] = mapped_column()
    kind: Mapped[str] = mapped_column(String(16), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(80), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(32))

    workstream: Mapped["TriageWorkstream"] = relationship(back_populates="events")
