"""Evaluate domain: Benchmark, EvalRun, EvalResult."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from surogate.core.db.base import Base, UUIDMixin, TimestampMixin


# ── Enums ──────────────────────────────────────────────────────────────


class BenchmarkCategory(enum.Enum):
    reasoning = "reasoning"
    language = "language"
    knowledge = "knowledge"
    coding = "coding"
    safety = "safety"
    chat = "chat"
    instruction = "instruction"
    custom = "custom"


class EvalRunStatus(enum.Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    queued = "queued"


class EvalResultStatus(enum.Enum):
    completed = "completed"
    pending = "pending"
    failed = "failed"


# ── Association table ──────────────────────────────────────────────────

eval_run_benchmarks = sa.Table(
    "eval_run_benchmarks",
    Base.metadata,
    sa.Column(
        "eval_run_id",
        sa.String(36),
        sa.ForeignKey("eval_runs.id"),
        primary_key=True,
    ),
    sa.Column(
        "benchmark_id",
        sa.String(36),
        sa.ForeignKey("benchmarks.id"),
        primary_key=True,
    ),
)


# ── Models ─────────────────────────────────────────────────────────────


class Benchmark(Base):
    __tablename__ = "benchmarks"

    id: Mapped[str] = mapped_column(sa.String(255), primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(255))
    category: Mapped[BenchmarkCategory] = mapped_column(sa.Enum(BenchmarkCategory))
    description: Mapped[str] = mapped_column(sa.Text, default="")
    sample_count: Mapped[int] = mapped_column(sa.Integer)
    metric: Mapped[str] = mapped_column(sa.String(128))
    built_in: Mapped[bool] = mapped_column(sa.Boolean, default=False)


class EvalRun(UUIDMixin, Base):
    __tablename__ = "eval_runs"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    model_id: Mapped[str] = mapped_column(sa.ForeignKey("deployed_models.id"))
    compare_model_id: Mapped[Optional[str]] = mapped_column(
        sa.ForeignKey("deployed_models.id"), nullable=True
    )
    status: Mapped[EvalRunStatus] = mapped_column(sa.Enum(EvalRunStatus))
    compute: Mapped[str] = mapped_column(sa.String(64))
    gpu: Mapped[str] = mapped_column(sa.String(128), default="")
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    duration: Mapped[Optional[str]] = mapped_column(sa.String(64), nullable=True)
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))

    benchmarks: Mapped[list[Benchmark]] = relationship(
        secondary=eval_run_benchmarks
    )
    results: Mapped[list[EvalResult]] = relationship(back_populates="eval_run")
    created_by: Mapped["User"] = relationship()  # noqa: F821


class EvalResult(UUIDMixin, Base):
    __tablename__ = "eval_results"

    eval_run_id: Mapped[str] = mapped_column(
        sa.ForeignKey("eval_runs.id"), index=True
    )
    benchmark_id: Mapped[str] = mapped_column(sa.ForeignKey("benchmarks.id"))
    score: Mapped[float] = mapped_column(sa.Float)
    previous_score: Mapped[Optional[float]] = mapped_column(
        sa.Float, nullable=True
    )
    delta: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    sample_results: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    status: Mapped[EvalResultStatus] = mapped_column(sa.Enum(EvalResultStatus))

    eval_run: Mapped[EvalRun] = relationship(back_populates="results")
    benchmark: Mapped[Benchmark] = relationship()


__all__ = [
    "BenchmarkCategory",
    "EvalRunStatus",
    "EvalResultStatus",
    "eval_run_benchmarks",
    "Benchmark",
    "EvalRun",
    "EvalResult",
]
