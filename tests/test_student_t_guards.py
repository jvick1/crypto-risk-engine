import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import t

from src import distributions
from src.distributions import DF_FLOOR, DF_NORMAL_FALLBACK, fit_student_t
from src.risk_metrics import compute_cvar_student_t, compute_var_student_t
from src.backtest import rolling_backtest


def test_tiny_sample_does_not_crash():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, loc, scale = fit_student_t(pd.Series([0.01, -0.02, 0.005]))
    assert df >= DF_FLOOR and np.isfinite(loc) and scale > 0


def test_constant_sample_does_not_crash():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, loc, scale = fit_student_t(pd.Series([0.0] * 50))
    assert df >= DF_FLOOR and np.isfinite(loc) and scale > 0
    assert np.isfinite(compute_cvar_student_t(df, loc, scale, 0.05))


def test_df_respects_floor_on_cauchy_data():
    x = t.rvs(0.8, size=2000, random_state=np.random.default_rng(0))
    df, _, _ = fit_student_t(pd.Series(x))
    assert df >= DF_FLOOR


def test_normal_data_unchanged():
    x = pd.Series(t.rvs(5, scale=0.02, size=3000, random_state=np.random.default_rng(1)))
    assert fit_student_t(x) == pytest.approx(t.fit(x))


def test_cvar_matches_numeric_integral_df4():
    df, alpha = 4.0, 0.05
    q = t.ppf(alpha, df)
    numeric = integrate.quad(lambda z: z * t.pdf(z, df), -np.inf, q)[0] / alpha
    assert compute_cvar_student_t(df, 0.0, 1.0, alpha) == pytest.approx(numeric, rel=1e-6)


def test_cvar_finite_near_floor():
    cvar = compute_cvar_student_t(DF_FLOOR, 0.0, 0.02, 0.05)
    assert np.isfinite(cvar) and cvar < compute_var_student_t(DF_FLOOR, 0.0, 0.02, 0.05)


@pytest.mark.parametrize("df", [1.0, 0.5, np.nan])
def test_cvar_raises_for_invalid_df(df):
    with pytest.raises(ValueError):
        compute_cvar_student_t(df, 0.0, 1.0, 0.05)


def test_cvar_raises_for_invalid_alpha():
    with pytest.raises(ValueError):
        compute_cvar_student_t(5.0, 0.0, 1.0, 1.5)


def test_fallback_fires_when_fit_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("optimiser failed")
    monkeypatch.setattr(distributions.t, "fit", boom)
    x = pd.Series([0.01, -0.02, 0.03, -0.01])
    with pytest.warns(RuntimeWarning, match="falling back to Normal"):
        df, loc, scale = fit_student_t(x)
    assert df == DF_NORMAL_FALLBACK
    assert loc == pytest.approx(x.mean()) and scale == pytest.approx(np.std(x))


def test_fallback_fires_on_invalid_params(monkeypatch):
    monkeypatch.setattr(distributions.t, "fit", lambda *a, **k: (np.nan, 0.0, 1.0))
    with pytest.warns(RuntimeWarning):
        df, _, _ = fit_student_t(pd.Series([0.01, -0.02, 0.03]))
    assert df == DF_NORMAL_FALLBACK


def test_backtest_student_t_finite_on_heavy_tails():
    x = t.rvs(0.9, scale=0.01, size=80, random_state=np.random.default_rng(2))
    d = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=80), "return": x})
    res, _ = rolling_backtest(d, window_size=30, alpha=0.05, methods=["student_t"])
    assert np.isfinite(res["student_t_var"]).all() and np.isfinite(res["student_t_cvar"]).all()
