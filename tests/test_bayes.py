"""Fast tests for the PyMC Bayesian Student-t (tiny draws on seeded synthetic t data)."""

import argparse
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from src.bayes import (
    PARAMS, check_diagnostics, fit_bayes_student_t, hdi, predictive_var_cvar, run_bayes, summarize,
)
from src.risk_metrics import compute_cvar_student_t, compute_var_student_t
from src.main import parse_methods

TINY = dict(draws=150, tune=150, chains=2, cores=1)


@pytest.fixture(scope="module")
def returns():
    return pd.Series(0.001 + 0.02 * t.rvs(4, size=400, random_state=np.random.default_rng(7)))


@pytest.fixture(scope="module")
def idata(returns):
    return fit_bayes_student_t(returns, seed=123, **TINY)


@pytest.fixture(scope="module")
def summary(idata, returns):
    return summarize(idata, alpha=0.05, n_obs=len(returns), seed=123)


def test_posterior_shapes(idata):
    for p in PARAMS:
        assert idata.posterior[p].values.shape == (TINY["chains"], TINY["draws"])


def test_nu_above_two(idata):
    assert (idata.posterior["nu"].values > 2).all()


def test_summary_layout_and_finite(summary):
    assert list(summary.index) == ["mu", "sigma", "nu", "var", "cvar"]
    num = summary[["mean", "sd", "median", "hdi_low", "hdi_high"]].to_numpy()
    assert np.isfinite(num).all()
    assert np.isfinite(summary.loc[["var", "cvar"], "predictive"]).all()
    assert np.isfinite(summary.loc[PARAMS, ["r_hat", "ess_bulk", "ess_tail"]].to_numpy()).all()
    assert (summary["hdi_low"] <= summary["median"]).all() and (summary["median"] <= summary["hdi_high"]).all()


def test_cvar_worse_than_var(summary):
    assert summary.loc["cvar", "mean"] < summary.loc["var", "mean"] < 0
    assert summary.loc["cvar", "predictive"] < summary.loc["var", "predictive"] < 0
    assert summary.loc["cvar", "hdi_high"] < summary.loc["var", "hdi_high"]


def test_same_seed_reproducible(idata, returns):
    again = fit_bayes_student_t(returns, seed=123, **TINY)
    np.testing.assert_array_equal(idata.posterior["nu"].values, again.posterior["nu"].values)
    a = summarize(idata, seed=123)
    b = summarize(again, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_hdi_contains_mass():
    x = np.random.default_rng(0).normal(size=10_000)
    lo, hi = hdi(x, 0.94)
    assert np.mean((x >= lo) & (x <= hi)) == pytest.approx(0.94, abs=0.01)
    assert lo == pytest.approx(-1.88, abs=0.1) and hi == pytest.approx(1.88, abs=0.1)


def test_predictive_single_draw_matches_closed_form():
    var, cvar = predictive_var_cvar(np.array([0.001]), np.array([0.02]), np.array([4.0]), 0.05)
    assert var == pytest.approx(compute_var_student_t(4.0, 0.001, 0.02, 0.05), rel=1e-8)
    assert cvar == pytest.approx(compute_cvar_student_t(4.0, 0.001, 0.02, 0.05), rel=1e-8)


def test_predictive_mixture_matches_simulation():
    rng = np.random.default_rng(3)
    mu, sigma, nu = rng.normal(0, 0.001, 50), rng.uniform(0.015, 0.025, 50), rng.uniform(4, 8, 50)
    var, cvar = predictive_var_cvar(mu, sigma, nu, 0.05)
    y = np.repeat(mu, 20_000) + np.repeat(sigma, 20_000) * t.rvs(np.repeat(nu, 20_000), random_state=rng)
    assert var == pytest.approx(np.quantile(y, 0.05), rel=0.01)
    assert cvar == pytest.approx(y[y <= np.quantile(y, 0.05)].mean(), rel=0.02)


def test_check_diagnostics_warns_on_bad_rhat(summary):
    bad = summary.copy()
    bad.loc["nu", "r_hat"] = 1.2
    with pytest.warns(RuntimeWarning, match="R-hat for nu"):
        problems = check_diagnostics(bad)
    assert any("R-hat for nu" in p for p in problems)


def test_parse_methods():
    assert parse_methods("normal,t,hist,bayes") == ["normal", "t", "hist", "bayes"]
    assert parse_methods(" T , normal,t") == ["t", "normal"]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_methods("normal,garch")


@pytest.mark.slow
def test_recovers_parameters_on_larger_sample():
    """Slower (~10-20 s): default-ish sampler settings recover the true t parameters."""
    x = 0.0005 + 0.03 * t.rvs(5, size=3000, random_state=np.random.default_rng(11))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s, _ = run_bayes(x, draws=500, tune=500, chains=2, cores=2, seed=1)
    assert s.loc["sigma", "hdi_low"] < 0.03 < s.loc["sigma", "hdi_high"]
    assert s.loc["nu", "hdi_low"] < 5 < s.loc["nu", "hdi_high"]
    assert (s.loc[PARAMS, "r_hat"] < 1.05).all()  # loose: only 2 x 500 draws
