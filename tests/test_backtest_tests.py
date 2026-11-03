import math
import numpy as np
import pandas as pd
import pytest

from src.backtest_tests import kupiec_pof, christoffersen_independence, christoffersen_cc
from src.backtest import rolling_backtest


def flags(T, idx):
    b = np.zeros(T, dtype=bool)
    b[list(idx)] = True
    return b


def test_kupiec_exact_coverage_is_zero():
    r = kupiec_pof(flags(100, range(5)), 0.05)
    assert r["lr_pof"] == pytest.approx(0.0, abs=1e-12)
    assert r["p_pof"] == pytest.approx(1.0)


def test_kupiec_known_value_10_of_100():
    # LR = -2[90 ln(.95/.90) + 10 ln(.05/.10)] = 4.130843
    r = kupiec_pof(flags(100, range(10)), 0.05)
    assert r["lr_pof"] == pytest.approx(4.130843, abs=1e-5)
    assert r["p_pof"] < 0.05


def test_kupiec_zero_breaches_no_nan():
    # LR = -2 * 100 ln(0.95) = 10.258658
    r = kupiec_pof(np.zeros(100, dtype=bool), 0.05)
    assert r["lr_pof"] == pytest.approx(-200 * math.log(0.95), abs=1e-9)
    assert np.isfinite(r["p_pof"])


def test_kupiec_all_breaches_no_nan():
    r = kupiec_pof(np.ones(20, dtype=bool), 0.05)
    assert r["lr_pof"] == pytest.approx(-40 * math.log(0.05), abs=1e-9)


def test_independence_zero_breaches():
    r = christoffersen_independence(np.zeros(50, dtype=bool))
    assert r["lr_ind"] == 0.0 and r["p_ind"] == 1.0


def test_independence_single_last_breach_no_transitions_out():
    r = christoffersen_independence(flags(50, [49]))
    assert r["n10"] + r["n11"] == 0
    assert np.isfinite(r["lr_ind"])


def test_independence_clustered_rejects():
    # 10 consecutive breaches in 200 obs: n00=188,n01=1,n10=1,n11=9
    b = flags(200, range(100, 110))
    r = christoffersen_independence(b)
    assert (r["n00"], r["n01"], r["n10"], r["n11"]) == (188, 1, 1, 9)
    n00, n01, n10, n11 = 188, 1, 1, 9
    pi0, pi1, pi = n01 / 189, n11 / 10, 10 / 199
    ll0 = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
    ll1 = n00 * math.log(1 - pi0) + n01 * math.log(pi0) + n10 * math.log(1 - pi1) + n11 * math.log(pi1)
    assert r["lr_ind"] == pytest.approx(-2 * (ll0 - ll1), rel=1e-12)
    assert r["p_ind"] < 0.001


def test_cc_is_sum_and_df2():
    b = flags(200, range(100, 110))
    r = christoffersen_cc(b, 0.05)
    assert r["lr_cc"] == pytest.approx(r["lr_pof"] + r["lr_ind"])
    assert r["expected_breaches"] == pytest.approx(10.0)
    assert r["reject_cc_5pct"] is True
    assert r["reject_pof_5pct"] is False


def test_rolling_backtest_summary_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=80), "return": rng.normal(0, 0.02, 80)})
    _, s = rolling_backtest(df, window_size=30, alpha=0.05, methods=["normal", "historical"])
    for c in ["breaches", "expected_breaches", "breach_rate", "lr_pof", "p_pof", "lr_ind", "p_ind", "lr_cc", "p_cc", "reject_cc_5pct"]:
        assert c in s.columns
    assert not s[["lr_pof", "lr_ind", "lr_cc"]].isna().any().any()
