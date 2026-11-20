"""Parallel rolling backtest == serial, warm-start path runs, hook still fires."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.backtest import rolling_backtest
from src.distributions import DF_FLOOR, fit_student_t
from src.timing import reset_collector


def _tiny_returns(n=80, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "return": 0.001 + 0.02 * rng.standard_t(df=5, size=n),
    })


def test_parallel_matches_serial_on_small_series():
    df = _tiny_returns()
    serial, s_serial = rolling_backtest(
        df, window_size=20, alpha=0.05,
        methods=["normal", "student_t", "historical"],
        workers=1,
    )
    parallel, s_parallel = rolling_backtest(
        df, window_size=20, alpha=0.05,
        methods=["normal", "student_t", "historical"],
        workers=2,
    )
    pd.testing.assert_frame_equal(serial, parallel, rtol=0, atol=0)
    pd.testing.assert_frame_equal(s_serial, s_parallel, rtol=0, atol=0)


def test_parallel_emits_stage_hooks():
    events = []

    def hook(event, name, elapsed, payload):
        events.append((event, name))

    reset_collector(on_stage=hook)
    rolling_backtest(
        _tiny_returns(n=50), window_size=15, alpha=0.05,
        methods=["normal", "historical"],
        workers=2, on_stage=hook,
    )
    names = [n for _, n in events]
    assert names[0] == "backtest loop" and events[0][0] == "start"
    assert ("end", "backtest loop") in events
    assert ("start", "backtest tests") in events


def test_t_warm_start_runs_and_stays_finite():
    df = _tiny_returns()
    results, summary = rolling_backtest(
        df, window_size=20, alpha=0.05,
        methods=["student_t"],
        workers=1, t_warm_start=True,
    )
    assert np.isfinite(results["student_t_var"]).all()
    assert np.isfinite(results["student_t_cvar"]).all()
    assert summary.loc["student_t", "breaches"] >= 0


def test_t_warm_start_forces_serial_with_warning():
    df = _tiny_returns(n=50)
    with pytest.warns(RuntimeWarning, match="t_warm_start"):
        results, _ = rolling_backtest(
            df, window_size=15, alpha=0.05,
            methods=["student_t"],
            workers=2, t_warm_start=True,
        )
    assert len(results)


def test_fit_student_t_accepts_start_values():
    rng = np.random.default_rng(1)
    x = pd.Series(0.001 + 0.02 * rng.standard_t(df=5, size=400))
    cold = fit_student_t(x)
    warm = fit_student_t(x, start=cold)
    assert warm[0] >= DF_FLOOR
    assert np.isfinite(warm).all()
    # starting at the MLE should stay very close
    assert warm == pytest.approx(cold, rel=1e-4, abs=1e-6)
