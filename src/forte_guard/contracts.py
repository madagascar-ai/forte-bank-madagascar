"""Versioned interchange contracts. Change these together before changing module boundaries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Box(Contract):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Box must have positive area")
        return self

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def contains(self, point: tuple[float, float]) -> bool:
        return self.x1 <= point[0] <= self.x2 and self.y1 <= point[1] <= self.y2


class Zone(Contract):
    name: Literal["cashier_area", "client_area", "counter_area", "storage_area", "exit_area"]
    box: Box


class ZoneProfile(Contract):
    source: str = "not_configured"
    zones: list[Zone] = Field(default_factory=list)


class Frame(Contract):
    frame_id: str
    timestamp_sec: float = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    image_path: str


class Detection(Contract):
    track_id: str
    label: str
    box: Box
    confidence: float = Field(ge=0, le=1)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ObjectFrame(Contract):
    frame_id: str
    timestamp_sec: float = Field(ge=0)
    detections: list[Detection] = Field(default_factory=list)


class ObjectJournal(Contract):
    schema_version: str = "1.0"
    backend: str
    supported_classes: list[str]
    frames: list[ObjectFrame]
    warnings: list[str] = Field(default_factory=list)


class Action(Contract):
    event_id: str
    type: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    actor_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    evidence_frame_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    visibility: Literal["clear", "uncertain"] = "uncertain"
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec < self.start_sec:
            raise ValueError("Event interval is reversed")
        return self


class State(Contract):
    state_id: str
    name: str
    value: bool | str | None
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    entity_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    evidence_frame_ids: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    visibility: Literal["clear", "uncertain"] = "uncertain"

    @model_validator(mode="after")
    def ordered(self):
        if self.end_sec < self.start_sec:
            raise ValueError("State interval is reversed")
        return self


class Batch(Contract):
    batch_id: str
    session_id: str | None = None
    parent_batch_ids: list[str] = Field(default_factory=list)
    provenance: Literal["observed", "unknown"] = "unknown"


class ActionJournal(Contract):
    schema_version: str = "1.0"
    duration_sec: float
    sample_fps: float
    actions: list[Action] = Field(default_factory=list)
    states: list[State] = Field(default_factory=list)
    batches: list[Batch] = Field(default_factory=list)
    roles: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    unsupported_actions: list[str] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class Rule(Contract):
    id: str
    title: str
    kind: Literal["action", "omission", "state"]
    enabled: bool = True
    source_status: Literal["verified", "needs_review"] = "needs_review"
    engine: dict[str, Any]
    document: dict[str, str]
    conditions: str = ""
    # What is visible, in plain words, for the reviewer; {t} is the timecode in seconds.
    observation: str = ""
    required_capabilities: list[str] = Field(default_factory=list)


class Candidate(Contract):
    candidate_id: str
    rule_id: str
    kind: Literal["action", "omission", "state"]
    event_ids: list[str] = Field(default_factory=list)
    state_ids: list[str] = Field(default_factory=list)
    actor_ids: list[str] = Field(default_factory=list)
    object_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    timecode_sec: float = Field(ge=0)
    sources: list[Literal["rule_engine", "llm"]]
    evidence_frame_ids: list[str]
    reasoning: str
    status: Literal["candidate", "accepted", "rejected"] = "candidate"
    verification: str = "not_run"


class Evidence(Contract):
    timestamp_sec: float
    image_base64: str


class Violation(Contract):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[Evidence]
    rule_id: str
    sources: list[str]
    document: dict[str, str]


class AnalysisResult(Contract):
    job_id: str
    status: Literal["done"] = "done"
    verdict: Literal["violation", "no_violation"]
    violations: list[Violation]
    warnings: list[str] = Field(default_factory=list)
    verification_mode: str = "structural_only"
    timings_sec: dict[str, float] = Field(default_factory=dict)
    stats: dict[str, int | float | str] = Field(default_factory=dict)
