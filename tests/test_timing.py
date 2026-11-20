"""Tests for the Phase 6 timing collector, progress hook, and --methods default parsing."""

from pathlib import Path
import csv
import time

import numpy as np
import pandas as pd
import pytest

from src.main import DEFAULT_METHODS, build_parser, parse_methods
from src.timing import TimingCollector, new_run_id, reset_collector
from src.backtest import rolling_backtest


def test_collector_nesting_and_summary():
    collector = TimingCollector()
    with collector.stage("backtest"):
        with collector.stage("backtest loop"):
            time.sleep(0.01)
        with collector.stage("rolling bayes"):
            time.sleep(0.005)
        with collector.stage("backtest tests"):
            pass
    rows = collector.summary_rows()
    names = [r.name for r in rows]
    assert names == ["backtest", "backtest loop", "rolling bayes", "backtest tests"]
    by_name = {r.name: r for r in rows}
    assert by_name["backtest"].parent is None
    assert by_name["backtest loop"].parent == "backtest"
    assert by_name["rolling bayes"].parent == "backtest"
    assert by_name["backtest tests"].parent == "backtest"
    assert by_name["backtest loop"].seconds >= 0.01
    assert by_name["rolling bayes"].seconds >= 0.005
    # Parent wall-clock includes its children
    assert by_name["backtest"].seconds >= (
        by_name["backtest loop"].seconds
        + by_name["rolling bayes"].seconds
        + by_name["backtest tests"].seconds
    )
    assert collector.top_level_total() == pytest.approx(by_name["backtest"].seconds)


def test_timing_csv_columns(tmp_path: Path):
    collector = TimingCollector()
    with collector.stage("data pull"):
        pass
    with collector.stage("returns"):
        pass
    run_id = "test-run-id"
    settings = {"coin_symbol": "btc", "alpha": 0.05, "backtest": True}
    path = tmp_path / "btc-timings.csv"
    collector.to_csv(path, run_id, settings)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [c for c in rows[0].keys()][:4] == ["run_id", "stage", "parent", "seconds"]
    assert "coin_symbol" in rows[0] and "alpha" in rows[0] and "backtest" in rows[0]
    assert [r["stage"] for r in rows] == ["data pull", "returns"]
    assert all(r["run_id"] == run_id for r in rows)
    assert rows[0]["parent"] == ""
    assert rows[0]["coin_symbol"] == "btc"
    assert float(rows[0]["seconds"]) >= 0.0


def test_progress_hook_called_in_order():
    events = []

    def hook(event, name, elapsed, payload):
        events.append((event, name, elapsed, payload))

    collector = TimingCollector(on_stage=hook)
    with collector.stage("backtest", payload="outer"):
        with collector.stage("backtest loop", payload="inner"):
            time.sleep(0.002)
    pairs = [(e, n) for e, n, _, _ in events]
    assert pairs == [
        ("start", "backtest"),
        ("start", "backtest loop"),
        ("end", "backtest loop"),
        ("end", "backtest"),
    ]
    assert events[0][2] == 0.0 and events[1][2] == 0.0
    assert events[2][2] > 0.0 and events[3][2] >= events[2][2]
    assert events[0][3] == "outer" and events[2][3] == "inner"


def test_progress_hook_payload_can_be_set_during_stage():
    events = []

    def hook(event, name, elapsed, payload):
        events.append((event, payload))

    collector = TimingCollector(on_stage=hook)
    with collector.stage("data pull") as st:
        st.payload = "downloaded-head"
    assert events[0] == ("start", None)
    assert events[1] == ("end", "downloaded-head")


def test_rolling_backtest_emits_stage_hooks_in_order():
    events = []

    def hook(event, name, elapsed, payload):
        events.append((event, name))

    reset_collector(on_stage=hook)
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "return": 0.001 + 0.02 * rng.standard_normal(n),
    })
    rolling_backtest(
        df, window_size=10, alpha=0.05,
        methods=["normal", "historical"],
        on_stage=hook,
        return_refits=True,
    )
    assert events[0] == ("start", "backtest loop")
    names = [n for e, n in events]
    # start/end for loop then tests; no rolling bayes when bayes is not in methods
    assert names == [
        "backtest loop", "backtest loop",
        "backtest tests", "backtest tests",
    ]
    assert events[1] == ("end", "backtest loop")
    assert events[2] == ("start", "backtest tests")
    assert events[3] == ("end", "backtest tests")


def test_cli_methods_default_is_parsed_list():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.methods == parse_methods(DEFAULT_METHODS) == ["normal", "t"]
    assert isinstance(args.methods, list)
    args = parser.parse_args(["--methods", "normal,t,hist"])
    assert args.methods == ["normal", "t", "hist"]


def test_new_run_id_is_unique():
    assert new_run_id() != new_run_id()
