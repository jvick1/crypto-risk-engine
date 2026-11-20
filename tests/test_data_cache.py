"""Raw-data cache merge and delta-fetch (mocked; no network)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.api import (
    YFRateLimitError,
    load_or_fetch_raw,
    merge_raw_prices,
    save_raw_data,
)


def _raw(dates, prices):
    return pd.DataFrame({
        "snapped_at": pd.to_datetime(dates, utc=True),
        "price": prices,
    })


def test_merge_newer_rows_win_and_new_days_are_appended():
    cached = _raw(["2020-01-01", "2020-01-02", "2020-01-03"], [1.0, 2.0, 3.0])
    incoming = _raw(["2020-01-02", "2020-01-03", "2020-01-04"], [2.5, 3.5, 4.0])
    out = merge_raw_prices(cached, incoming)
    assert list(out["price"]) == [1.0, 2.5, 3.5, 4.0]
    assert [d.date() for d in out["snapped_at"]] == [
        date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)
    ]


def test_merge_empty_incoming_keeps_cache():
    cached = _raw(["2020-01-01"], [1.0])
    incoming = _raw([], [])
    out = merge_raw_prices(cached, incoming)
    assert list(out["price"]) == [1.0]


def test_cache_data_delta_fetch_starts_five_days_before_last(monkeypatch, tmp_path: Path):
    path = tmp_path / "btc-usd-max.csv"
    cached = _raw(
        [f"2020-01-{d:02d}" for d in range(1, 11)],
        [float(i) for i in range(1, 11)],
    )
    save_raw_data(cached, path)
    calls = []

    def fake_fetch(ticker, start=None):
        calls.append((ticker, start))
        return _raw(["2020-01-08", "2020-01-10", "2020-01-11"], [80.0, 100.0, 11.0])

    monkeypatch.setattr("src.api.fetch_historical_data", fake_fetch)
    out = load_or_fetch_raw("BTC-USD", path, cache_data=True, refresh=False)
    assert calls == [("BTC-USD", date(2020, 1, 5))]  # last 2020-01-10 minus 5 days
    # unchanged days kept; 01-10 overwritten; 01-11 added
    by_day = {d.date(): p for d, p in zip(out["snapped_at"], out["price"])}
    assert by_day[date(2020, 1, 1)] == 1.0
    assert by_day[date(2020, 1, 10)] == 100.0
    assert by_day[date(2020, 1, 11)] == 11.0


def test_refresh_forces_full_download(monkeypatch, tmp_path: Path):
    path = tmp_path / "btc-usd-max.csv"
    save_raw_data(_raw(["2020-01-01"], [1.0]), path)
    calls = []

    def fake_fetch(ticker, start=None):
        calls.append(start)
        return _raw(["2019-01-01", "2020-01-01"], [0.5, 1.5])

    monkeypatch.setattr("src.api.fetch_historical_data", fake_fetch)
    out = load_or_fetch_raw("BTC-USD", path, cache_data=True, refresh=True)
    assert calls == [None]
    assert list(out["price"]) == [0.5, 1.5]


def test_default_path_does_full_download_even_if_file_exists(monkeypatch, tmp_path: Path):
    path = tmp_path / "btc-usd-max.csv"
    save_raw_data(_raw(["2020-01-01"], [1.0]), path)
    calls = []

    def fake_fetch(ticker, start=None):
        calls.append(start)
        return _raw(["2020-01-01"], [9.0])

    monkeypatch.setattr("src.api.fetch_historical_data", fake_fetch)
    out = load_or_fetch_raw("BTC-USD", path, cache_data=False)
    assert calls == [None]
    assert list(out["price"]) == [9.0]


def test_rate_limit_falls_back_to_cache(monkeypatch, tmp_path: Path):
    path = tmp_path / "btc-usd-max.csv"
    cached = _raw(["2020-01-01", "2020-01-02"], [1.0, 2.0])
    save_raw_data(cached, path)

    def boom(ticker, start=None):
        raise YFRateLimitError()

    monkeypatch.setattr("src.api.fetch_historical_data", boom)
    with pytest.warns(RuntimeWarning, match="using cached raw data"):
        out = load_or_fetch_raw("BTC-USD", path, cache_data=True)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        cached.reset_index(drop=True),
        check_dtype=False,
    )


def test_normalize_flattens_multilevel_columns():
    import pandas as pd
    from src.api import _normalize_downloaded
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="Date", tz="UTC")
    cols = pd.MultiIndex.from_tuples([("Close", "BTC-USD")], names=["Price", "Ticker"])
    raw = pd.DataFrame([[1.25], [2.5]], index=idx, columns=cols)
    out = _normalize_downloaded(raw)
    assert list(out.columns) == ["snapped_at", "price"]
    assert not isinstance(out.columns, pd.MultiIndex)
    assert list(out["price"]) == [1.25, 2.5]
    assert out["snapped_at"].dt.date.tolist()[0].isoformat() == "2020-01-01"
