"""Fast tests for the rolling Bayesian backtest (Phase 4). MCMC tests are tiny and marked slow."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

import src.rolling_bayes as rb
from src.backtest import rolling_backtest
from src.risk_metrics import compute_cvar_student_t, compute_var_student_t


def synthetic(n=330, seed=11):
    rng = np.random.default_rng(seed)
    return 0.0005 + 0.03 * t.rvs(4, size=n, random_state=rng)


def as_df(x):
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(x), freq="D"), "return": x})


# ---------------------------------------------------------------- refit schedule

def test_refit_schedule_k1_is_every_day():
    s = rb.refit_schedule(262, 252, 1)
    assert np.array_equal(s, np.arange(252, 262))


def test_refit_schedule_blocks_and_never_after_forecast_day():
    s = rb.refit_schedule(265, 252, 5)
    assert list(s) == [252] * 5 + [257] * 5 + [262] * 3
    days = np.arange(252, 265)
    assert (s <= days).all() and (days - s < 5).all()


def test_refit_schedule_rejects_bad_input():
    with pytest.raises(ValueError):
        rb.refit_schedule(300, 252, 0)
    with pytest.raises(ValueError):
        rb.refit_schedule(252, 252, 1)


def test_default_refit_every():
    assert rb.DEFAULT_REFIT_EVERY == {"mcmc": 21, "conjugate": 1}


# ---------------------------------------------------------------- conjugate math

def test_nig_posterior_matches_hand_calculation():
    # x = [0.01, -0.02, 0.03], prior m0=0, kappa0=1, a0=1, b0=1e-4
    # n=3, xbar=0.02/3, SS = 1.2666...e-3
    # kappa_n = 4, m_n = 0.02/4 = 0.005, a_n = 1 + 3/2 = 2.5
    # b_n = 1e-4 + 0.5*SS + 1*3*xbar^2/(2*4) = 1e-4 + 6.3333e-4 + 1.6667e-5 = 7.5e-4
    m_n, kappa_n, a_n, b_n = rb.nig_posterior([0.01, -0.02, 0.03], m0=0.0, kappa0=1.0, a0=1.0, b0=1e-4)
    assert kappa_n == pytest.approx(4.0)
    assert m_n == pytest.approx(0.005)
    assert a_n == pytest.approx(2.5)
    assert b_n == pytest.approx(7.5e-4)
    # predictive t: df = 2 a_n = 5, scale = sqrt(b_n (kappa_n + 1) / (a_n kappa_n)) = sqrt(3.75e-4)
    df, loc, scale = rb.nig_predictive(m_n, kappa_n, a_n, b_n)
    assert df == pytest.approx(5.0)
    assert loc == pytest.approx(0.005)
    assert scale == pytest.approx(np.sqrt(3.75e-4))


def test_conjugate_var_cvar_is_closed_form_t():
    x = [0.01, -0.02, 0.03]
    v, c = rb.conjugate_var_cvar(x, 0.05, m0=0.0, kappa0=1.0, a0=1.0, b0=1e-4)
    s = np.sqrt(3.75e-4)
    assert v == pytest.approx(compute_var_student_t(5.0, 0.005, s, 0.05))
    assert c == pytest.approx(compute_cvar_student_t(5.0, 0.005, s, 0.05))
    assert v == pytest.approx(0.005 + s * t.ppf(0.05, 5))


def test_nig_predictive_matches_simulation():
    # sigma^2 ~ InvGamma(a_n, b_n), mu | sigma^2 ~ N(m_n, sigma^2 / kappa_n), x ~ N(mu, sigma^2)
    m_n, kappa_n, a_n, b_n = rb.nig_posterior(synthetic(252))
    rng = np.random.default_rng(0)
    sig2 = b_n / rng.gamma(a_n, 1.0, size=400_000)
    mu = rng.normal(m_n, np.sqrt(sig2 / kappa_n))
    sim = rng.normal(mu, np.sqrt(sig2))
    df, loc, scale = rb.nig_predictive(m_n, kappa_n, a_n, b_n)
    assert np.quantile(sim, 0.05) == pytest.approx(loc + scale * t.ppf(0.05, df), rel=0.02)


def test_nig_posterior_rejects_bad_input():
    with pytest.raises(ValueError):
        rb.nig_posterior([0.01, np.nan])
    with pytest.raises(ValueError):
        rb.nig_posterior([0.01], b0=0.0)


# ---------------------------------------------------------------- no look-ahead / schedule (conjugate)

@pytest.mark.parametrize("k", [1, 5, 21])
def test_no_look_ahead_conjugate(k):
    x = synthetic()
    w, p = 252, 300
    v0, c0, _ = rb.rolling_bayes(x, w, 0.05, engine="conjugate", refit_every=k, progressbar=False)
    y = x.copy()
    y[p:] = y[p:] * 5 - 0.2  # wreck the future
    v1, c1, _ = rb.rolling_bayes(y, w, 0.05, engine="conjugate", refit_every=k, progressbar=False)
    past = slice(0, p - w + 1)  # forecast days w .. p use data before day p only
    np.testing.assert_array_equal(v0[past], v1[past])
    np.testing.assert_array_equal(c0[past], c1[past])
    assert not np.allclose(v0[p - w + k + 1:], v1[p - w + k + 1:])  # later days do change


def test_conjugate_values_follow_refit_schedule():
    x = synthetic()
    w, k = 252, 7
    v, c, refits = rb.rolling_bayes(x, w, 0.05, engine="conjugate", refit_every=k, progressbar=False)
    sched = rb.refit_schedule(len(x), w, k)
    assert len(refits) == len(np.unique(sched)) == int(np.ceil((len(x) - w) / k))
    for j, r in enumerate(sched):
        ev, ec = rb.conjugate_var_cvar(x[r - w:r], 0.05)
        assert v[j] == pytest.approx(ev) and c[j] == pytest.approx(ec)
    assert (refits["fit_end_pos"] == refits["refit_pos"] - 1).all()
    assert (refits["n_obs"] == w).all()


# ---------------------------------------------------------------- backtest integration

def test_backtest_existing_methods_unchanged_by_bayes():
    df = as_df(synthetic())
    base_r, base_s = rolling_backtest(df, 252, 0.05)
    r, s, refits = rolling_backtest(df, 252, 0.05, methods=["normal", "student_t", "historical", "bayes"],
                                    bayes_engine="conjugate", return_refits=True)
    pd.testing.assert_frame_equal(r[base_r.columns], base_r)
    pd.testing.assert_frame_equal(s.loc[base_s.index], base_s)
    assert list(r.columns[-3:]) == ["bayes_var", "bayes_cvar", "bayes_breach"]
    assert "bayes" in s.index and s.loc["bayes", "total_periods"] == len(r)
    assert (r["bayes_breach"] == (r["actual_return"] < r["bayes_var"])).all()
    assert (r["bayes_cvar"] < r["bayes_var"]).all()
    assert list(refits["refit_date"][:2]) == ["2020-09-09", "2020-09-10"]  # day 252, 253
    assert refits["fit_end_date"].iloc[0] == "2020-09-08"


def test_backtest_without_bayes_returns_no_refits():
    r, s, refits = rolling_backtest(as_df(synthetic(270)), 252, 0.05, methods=["normal"], return_refits=True)
    assert refits is None and "bayes_var" not in r.columns


# ---------------------------------------------------------------- MCMC driver with a fake sampler (fast)

@pytest.fixture
def fake_sampler(monkeypatch):
    calls = []

    def fake(model, window, draws, tune, chains, seed, init):
        calls.append({"seed": seed, "init": init, "end": float(window[-1])})
        m, sd = float(np.mean(window)), float(np.std(window))
        rng = np.random.default_rng(seed)
        size = draws * chains
        return {"mu": m + 1e-4 * rng.standard_normal(size), "sigma": np.full(size, 0.8 * sd),
                "nu": np.full(size, 4.0), "r_hat": np.array([1.0, 1.0, 1.02 if seed % 2 else 1.0]),
                "ess_bulk": np.array([500.0] * 3), "ess_tail": np.array([500.0] * 3),
                "divergences": np.array(0)}

    monkeypatch.setattr(rb, "sample_window", fake)
    monkeypatch.setattr(rb, "build_rolling_model", lambda w: object())
    return calls


KW = dict(window_size=252, alpha=0.05, engine="mcmc", refit_every=21, draws=50, tune=50, chains=2,
          seed=42, progressbar=False)


def test_cache_miss_then_hit(tmp_path, fake_sampler):
    x = synthetic()
    v0, c0, ref0 = rb.rolling_bayes(x, cache_dir=tmp_path, **KW)
    n_refits = len(ref0)
    assert len(fake_sampler) == n_refits and not ref0["cached"].any()
    assert len(list(tmp_path.glob("*.npz"))) == n_refits
    v1, c1, ref1 = rb.rolling_bayes(x, cache_dir=tmp_path, **KW)
    assert len(fake_sampler) == n_refits  # no new sampling
    assert ref1["cached"].all()
    np.testing.assert_array_equal(v0, v1)
    np.testing.assert_array_equal(c0, c1)


def test_cache_invalidation(tmp_path, fake_sampler):
    x = synthetic()
    _, _, ref = rb.rolling_bayes(x, cache_dir=tmp_path, **KW)
    n = len(ref)
    # change the last return inside the final refit window only -> only that refit misses
    y = x.copy()
    last_r = int(ref["refit_pos"].iloc[-1])
    y[last_r - 1] += 0.01
    _, _, ref_y = rb.rolling_bayes(y, cache_dir=tmp_path, **KW)
    assert list(ref_y["cached"]) == [True] * (n - 1) + [False]
    # different sampler settings or seed -> every refit misses
    _, _, ref_d = rb.rolling_bayes(x, cache_dir=tmp_path, **{**KW, "draws": 60})
    assert not ref_d["cached"].any()
    _, _, ref_s = rb.rolling_bayes(x, cache_dir=tmp_path, **{**KW, "seed": 7})
    assert not ref_s["cached"].any()
    # corrupt cache file -> treated as a miss, refit, rewritten
    victim = sorted(tmp_path.glob("*.npz"))[0]
    victim.write_bytes(b"not a zip")
    assert rb.load_cached(tmp_path, victim.stem) is None


def test_cache_key_covers_inputs():
    w = synthetic(252)
    base = rb.cache_key(w, 500, 500, 2, 1, None)
    assert base == rb.cache_key(w.copy(), 500, 500, 2, 1, None)
    w2 = w.copy(); w2[100] += 1e-12
    others = [rb.cache_key(w2, 500, 500, 2, 1, None), rb.cache_key(w, 501, 500, 2, 1, None),
              rb.cache_key(w, 500, 501, 2, 1, None), rb.cache_key(w, 500, 500, 4, 1, None),
              rb.cache_key(w, 500, 500, 2, 2, None),
              rb.cache_key(w, 500, 500, 2, 1, {"mu_z": 0.0, "sigma_z": 0.7, "nu_minus_2": 2.0})]
    assert len({base, *others}) == len(others) + 1


def test_no_cache_dir_writes_nothing(tmp_path, fake_sampler, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rb.rolling_bayes(synthetic(), cache_dir=None, **KW)
    assert not any(tmp_path.rglob("*.npz"))


def test_warm_start_from_previous_posterior_means(fake_sampler):
    x = synthetic()
    _, _, ref = rb.rolling_bayes(x, cache_dir=None, **KW)
    assert fake_sampler[0]["init"] is None
    for j in range(1, len(ref)):
        r = int(ref["refit_pos"].iloc[j])
        prev = {"mu": ref["mu_mean"].iloc[j - 1], "sigma": ref["sigma_mean"].iloc[j - 1],
                "nu": ref["nu_mean"].iloc[j - 1]}
        assert fake_sampler[j]["init"] == rb.warm_init(prev, float(np.std(x[r - 252:r])))
        assert fake_sampler[j]["seed"] == 42 + r
        assert fake_sampler[j]["end"] == x[r - 1]  # last point fitted is the day before the refit


def test_diagnostic_failures_counted_not_warned(fake_sampler, recwarn):
    _, _, ref = rb.rolling_bayes(synthetic(), cache_dir=None, **KW)
    expected = int((ref["seed"] % 2 == 1).sum())  # fake R-hat 1.02 on odd seeds
    assert int((~ref["passed"]).sum()) == expected > 0
    assert (ref.loc[~ref["passed"], "problems"] == "rhat").all()
    assert f"refits failing diagnostics {expected}/{len(ref)}" in rb.refit_report(ref)
    assert not [w for w in recwarn if "diagnostic" in str(w.message).lower()]


def test_diagnostic_problems_thresholds():
    ok = {"r_hat": np.array([1.0, 1.005, 1.01]), "ess_bulk": np.array([200.0] * 3),
          "ess_tail": np.array([200.0] * 3), "divergences": np.array(0)}
    assert rb.diagnostic_problems(ok, chains=2) == []
    bad = {**ok, "r_hat": np.array([1.0, 1.02, 1.0]), "ess_tail": np.array([199.0, 300, 300]),
           "divergences": np.array(3)}
    assert rb.diagnostic_problems(bad, chains=2) == ["rhat", "ess", "divergences"]


def test_rolling_bayes_rejects_bad_engine():
    with pytest.raises(ValueError):
        rb.rolling_bayes(synthetic(), engine="gibbs")


# ---------------------------------------------------------------- real MCMC (tiny, slow)

@pytest.mark.slow
def test_mcmc_tiny_no_look_ahead_and_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    x = 0.02 * t.rvs(4, size=80, random_state=rng)
    kw = dict(window_size=60, alpha=0.05, engine="mcmc", refit_every=10, draws=100, tune=100,
              chains=2, seed=5, progressbar=False)
    v0, c0, ref0 = rb.rolling_bayes(x, cache_dir=tmp_path, **kw)
    assert len(ref0) == 2 and np.isfinite(v0).all() and (c0 < v0).all()
    assert {"r_hat_max", "ess_bulk_min", "ess_tail_min", "divergences", "passed"} <= set(ref0.columns)
    assert (ref0["nu_mean"] > 2).all()
    # refits are on days 60 and 70 and use data before day 70 only: wrecking x[70:] changes nothing
    y = x.copy(); y[70:] = -0.5
    v1, _, ref1 = rb.rolling_bayes(y, cache_dir=tmp_path, **kw)
    np.testing.assert_array_equal(v0, v1)
    assert ref1["cached"].all()  # same windows + settings -> cache hits
    # same inputs without the cache reproduce the cached draws exactly (seeded sampler)
    v2, _, _ = rb.rolling_bayes(x, cache_dir=None, **kw)
    np.testing.assert_array_equal(v0, v2)
