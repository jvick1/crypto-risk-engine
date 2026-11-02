"""Basic sanity tests for VaR / CVaR sign conventions and ordering (synthetic, seeded data)."""

import numpy as np
import pandas as pd
import pytest

from src.distributions import fit_normal, fit_student_t
from src.risk_metrics import (
    compute_var_normal, compute_cvar_normal,
    compute_var_student_t, compute_cvar_student_t,
    compute_var_historical, compute_cvar_historical,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    return pd.Series(0.001 + 0.03 * rng.standard_t(df=5, size=2000))


def test_normal_var_is_negative_for_alpha_5pct():
    assert compute_var_normal(0.0, 0.03, 0.05) < 0


def test_cvar_is_at_or_below_var(returns):
    mu, sigma = fit_normal(returns)
    df, loc, scale = fit_student_t(returns)
    assert compute_cvar_normal(mu, sigma, 0.05) < compute_var_normal(mu, sigma, 0.05)
    assert compute_cvar_student_t(df, loc, scale, 0.05) < compute_var_student_t(df, loc, scale, 0.05)
    assert compute_cvar_historical(returns, 0.05) <= compute_var_historical(returns, 0.05)


def test_smaller_alpha_gives_more_extreme_var(returns):
    assert compute_var_historical(returns, 0.01) < compute_var_historical(returns, 0.05)
    mu, sigma = fit_normal(returns)
    assert compute_var_normal(mu, sigma, 0.01) < compute_var_normal(mu, sigma, 0.05)


def test_student_t_fit_has_positive_scale_and_df(returns):
    df, loc, scale = fit_student_t(returns)
    assert df > 0 and scale > 0
