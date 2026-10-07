"""Fetch team game logs per season, cached to data/raw/. Never re-fetches a saved completed season.

Run from the project root:  python -m pipeline.fetch_data
"""
import time
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamelog, leaguehustlestatsplayer

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2015, 2026)]  # 2015-16 .. 2025-26
CURRENT_SEASON = "2026-27"  # always re-fetched: still in progress


def game_log(season: str, kind: str = "T", sleep: float = 1.5) -> pd.DataFrame:
    """kind "T": one row per team-game; "P": one row per player-game (players who sat out have no row)."""
    path = RAW / f"{'team' if kind == 'T' else 'player'}_gamelog_{season}.parquet"
    if path.exists() and season != CURRENT_SEASON:
        return pd.read_parquet(path)
    df = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation=kind,
        timeout=60,
    ).get_data_frames()[0]
    time.sleep(sleep)  # be polite to stats.nba.com
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def all_team_game_logs() -> pd.DataFrame:
    return pd.concat([game_log(s, "T") for s in SEASONS], ignore_index=True)


def all_player_game_logs() -> pd.DataFrame:
    return pd.concat([game_log(s, "P") for s in SEASONS], ignore_index=True)


HUSTLE_SEASONS = SEASONS[1:]  # 2015-16 regular season has only 2 tracked games


def hustle_snapshot(season: str, date_to: pd.Timestamp, sleep: float = 1.5) -> pd.DataFrame:
    """Season-to-date hustle totals for games played on or before date_to."""
    path = RAW / "hustle" / f"{season}_{date_to:%Y-%m-%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    for wait in (10, 30, 90, None):  # stats.nba.com times out under load: back off and retry
        try:
            df = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
                season=season, per_mode_time="Totals", date_to_nullable=f"{date_to:%m/%d/%Y}", timeout=60,
            ).get_data_frames()[0]
            break
        except requests.exceptions.RequestException:
            if wait is None:
                raise
            print(f"timeout on {season} {date_to:%Y-%m-%d}, retrying in {wait}s", flush=True)
            time.sleep(wait)
    time.sleep(sleep)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def snapshot_dates(season: str) -> list[pd.Timestamp]:
    """Every Sunday inside the season, plus the final day. A game uses the latest snapshot BEFORE its date."""
    dates = pd.to_datetime(game_log(season, "T")["GAME_DATE"])
    sundays = pd.date_range(dates.min(), dates.max(), freq="W-SUN")
    return sorted(set(sundays) | {dates.max()})


def all_hustle_snapshots() -> pd.DataFrame:
    return pd.concat([hustle_snapshot(s, d).assign(SEASON_STR=s, DATE_TO=d)
                      for s in HUSTLE_SEASONS for d in snapshot_dates(s)], ignore_index=True)


def validate(df: pd.DataFrame) -> None:
    g = df.groupby("GAME_ID")
    assert (g.size() == 2).all(), "game without exactly 2 team rows"
    # 1 home team ("vs."), or 0 for neutral-site games (NBA Cup Vegas, Paris, Mexico City from 2024-25 on)
    homes = g["MATCHUP"].apply(lambda m: m.str.contains("vs.", regex=False).sum())
    assert homes.isin([0, 1]).all(), "game with two home teams"
    assert df["WL"].isin(["W", "L"]).all()
    assert df[["FGA", "FTA", "OREB", "TOV", "PTS", "MIN"]].notna().all().all()


if __name__ == "__main__":
    df = all_team_game_logs()
    validate(df)
    players = all_player_game_logs()
    assert set(players["GAME_ID"]) == set(df["GAME_ID"]), "player and team logs cover different games"
    print(f"player rows: {len(players)}")
    print(f"hustle snapshot rows: {len(all_hustle_snapshots())}")
    print(df.groupby("SEASON_ID").size())
    print("ok:", df["GAME_ID"].nunique(), "games")
