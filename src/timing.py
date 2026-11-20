"""
Run-level stage timing for the pipeline (Phase 6) and a progress hook for a future UI.

Use `stage(name)` as a nestable context manager. It logs start/end through the
`crypto_risk` logger (visible at INFO) and records wall-clock times via
time.perf_counter. An optional `on_stage(event, name, elapsed, payload)` callback
(event is "start" or "end") lets a UI show "working on X" and preview cheap
payloads such as a DataFrame head.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional
import csv
import logging
import time
import uuid

LOGGER_NAME = "crypto_risk"
logger = logging.getLogger(LOGGER_NAME)

# Stage names must match the node labels in docs/APP_FLOW.md (before the time suffix).
STAGE_DATA_PULL = "data pull"
STAGE_RETURNS = "returns"
STAGE_FITS = "fits"
STAGE_RISK_METRICS = "risk metrics"
STAGE_BAYES = "bayes"
STAGE_PLOTS = "plots"
STAGE_BACKTEST = "backtest"
STAGE_BACKTEST_LOOP = "backtest loop"
STAGE_ROLLING_BAYES = "rolling bayes"
STAGE_BACKTEST_TESTS = "backtest tests"
STAGE_ROLLING_VAR_PLOT = "rolling var plot"

OnStage = Callable[[str, str, float, Any], None]


@dataclass
class StageRecord:
    name: str
    parent: Optional[str]
    seconds: float = 0.0


@dataclass
class StageContext:
    """Yielded by `stage()` so the caller can attach a payload before exit."""

    name: str
    payload: Any = None


@dataclass
class TimingCollector:
    """Collects nestable stage timings for one run."""

    records: List[StageRecord] = field(default_factory=list)
    on_stage: Optional[OnStage] = None
    _stack: List[str] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, payload: Any = None) -> Iterator[StageContext]:
        parent = self._stack[-1] if self._stack else None
        record = StageRecord(name=name, parent=parent, seconds=0.0)
        self.records.append(record)
        self._stack.append(name)
        ctx = StageContext(name=name, payload=payload)
        logger.info("start: %s", name)
        if self.on_stage is not None:
            self.on_stage("start", name, 0.0, ctx.payload)
        t0 = time.perf_counter()
        try:
            yield ctx
        finally:
            record.seconds = time.perf_counter() - t0
            self._stack.pop()
            logger.info("end: %s (%.3f s)", name, record.seconds)
            if self.on_stage is not None:
                self.on_stage("end", name, record.seconds, ctx.payload)

    def summary_rows(self) -> List[StageRecord]:
        """Records in start order (parent appears before its children)."""
        return list(self.records)

    def top_level_total(self) -> float:
        return sum(r.seconds for r in self.records if r.parent is None)

    def print_summary(self, run_id: Optional[str] = None) -> None:
        rows = self.summary_rows()
        header = "Timing summary" + (f"  run_id={run_id}" if run_id else "")
        print(f"\n{header}  (seconds)")
        if not rows:
            print("  (no stages recorded)")
            return
        name_w = max(5, max(len(r.name) + (2 if r.parent else 0) for r in rows))
        parent_w = max(6, max(len(r.parent or "") for r in rows))
        print(f"{'stage':<{name_w}}  {'parent':<{parent_w}}  {'seconds':>10}")
        for r in rows:
            label = ("  " + r.name) if r.parent else r.name
            print(f"{label:<{name_w}}  {(r.parent or ''):<{parent_w}}  {r.seconds:10.3f}")
        print(f"{'top-level total':<{name_w}}  {'':<{parent_w}}  {self.top_level_total():10.3f}")

    def to_csv(self, path: Path, run_id: str, settings: Optional[dict] = None) -> None:
        """Write one row per stage: run_id, stage, parent, seconds, then any run settings."""
        settings = settings or {}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run_id", "stage", "parent", "seconds", *settings.keys()]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.summary_rows():
                row = {
                    "run_id": run_id,
                    "stage": r.name,
                    "parent": r.parent or "",
                    "seconds": r.seconds,
                    **{k: settings[k] for k in settings},
                }
                writer.writerow(row)


_current: Optional[TimingCollector] = None


def get_collector() -> TimingCollector:
    global _current
    if _current is None:
        _current = TimingCollector()
    return _current


def reset_collector(on_stage: Optional[OnStage] = None) -> TimingCollector:
    """Start a fresh run-level collector (call once at the beginning of main)."""
    global _current
    _current = TimingCollector(on_stage=on_stage)
    return _current


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


@contextmanager
def stage(name: str, payload: Any = None) -> Iterator[StageContext]:
    """Record a named stage on the current collector. Nestable."""
    with get_collector().stage(name, payload=payload) as ctx:
        yield ctx


def configure_logging(level: str = "WARNING") -> None:
    """Set the crypto_risk logger level. INFO shows per-stage start/end lines.

    Only this logger is changed, so PyMC / matplotlib stay quiet at default WARNING.
    """
    level_name = level.upper()
    level_num = getattr(logging, level_name, None)
    if not isinstance(level_num, int):
        raise ValueError(f"invalid log level {level!r}")
    logger.setLevel(level_num)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    for handler in logger.handlers:
        handler.setLevel(level_num)
    logger.propagate = False
