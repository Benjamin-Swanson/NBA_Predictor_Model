"""stats.nba.com is flaky: downloads retry, and daily files fall back to yesterday's copy. No network used."""
import pandas as pd
import pytest
import requests

from pipeline import fetch_data as fd


def boom():
    raise requests.exceptions.ReadTimeout("simulated")


def test_daily_falls_back_to_previous_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, "RAW", tmp_path)
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "rosters_2000-01-01.parquet")
    assert fd._daily("rosters_{}.parquet", boom)["x"].tolist() == [1]  # down: yesterday's copy
    assert fd._daily("rosters_{}.parquet", lambda: pd.DataFrame({"x": [2]}))["x"].tolist() == [2]  # up: saved
    assert fd._daily("rosters_{}.parquet", boom)["x"].tolist() == [2]  # today's file now cached
    with pytest.raises(requests.exceptions.ReadTimeout):  # nothing to fall back to: fail loudly
        fd._daily("schedule_{}.parquet", boom)


def test_get_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(fd.time, "sleep", lambda s: None)
    calls = []

    class Flaky:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if len(calls) < 3:
                raise requests.exceptions.ConnectionError("reset")

        def get_data_frames(self):
            return [pd.DataFrame({"ok": [1]})]

    assert fd._get(Flaky)["ok"].tolist() == [1]
    assert len(calls) == 3 and calls[0]["timeout"] == 60
