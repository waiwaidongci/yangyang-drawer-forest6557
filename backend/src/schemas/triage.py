from pydantic import BaseModel, ConfigDict, Field

from src.constants.workflow import Decision


class WorkstreamCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    label: str = Field(min_length=2, max_length=140)
    dataset_id: int = Field(gt=0)
    settings: dict[str, float] = Field(default_factory=dict, max_length=20)
    pending_ids: list[str] = Field(default_factory=list, max_length=500)


class JudgmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1, max_length=80)
    failed: int = Field(ge=0, le=1)
    confirmed_at: str = Field(min_length=16, max_length=32)
    idempotency_key: str = Field(min_length=4, max_length=80)


class SchedulingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    batch_id: str = Field(min_length=1, max_length=80)
    idempotency_key: str = Field(min_length=4, max_length=80)


class TriageReviewCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    decision: Decision
    note: str = Field(default="", max_length=500)
    idempotency_key: str = Field(min_length=4, max_length=80)


class EventAck(BaseModel):
    deduplicated: bool
    event_seq: int
    workstream_id: int
    position: int | None = None
    threshold_version: int | None = None
