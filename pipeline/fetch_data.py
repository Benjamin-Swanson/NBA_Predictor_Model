"""Fetch from nba_api, cached to data/raw/. Never re-fetches a completed season.
The current season's logs are re-fetched when older than CURRENT_MAX_AGE; its hustle snapshots are weekly.

Run from the project root:  python -m pipeline.fetch_data
"""
import time
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import (commonteamroster, leaguegamelog, leaguehustlestatsplayer, leaguestandingsv3,
                                     scheduleleaguev2)
from nba_api.stats.static import teams

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2015, 2026)]  # 2015-16 .. 2025-26
CURRENT_SEASON = "2026-27"  # in progress: re-fetched when stale
CURRENT_MAX_AGE = pd.Timedelta(hours=6)


def _fresh(path: Path) -> bool:
    return path.exists() and pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s") < CURRENT_MAX_AGE


def game_log(season: str, kind: str = "T", sleep: float = 1.5) -> pd.DataFrame:
    """kind "T": one row per team-game; "P": one row per player-game (players who sat out have no row)."""
    path = RAW / f"{'team' if kind == 'T' else 'player'}_gamelog_{season}.parquet"
    if (path.exists() and season != CURRENT_SEASON) or (season == CURRENT_SEASON and _fresh(path)):
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


def _concat(frames) -> pd.DataFrame:
    # An empty frame (current season before opening night) would turn numeric columns into object dtype
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def all_team_game_logs() -> pd.DataFrame:
    return _concat(game_log(s, "T") for s in SEASONS + [CURRENT_SEASON])


def all_player_game_logs() -> pd.DataFrame:
    return _concat(game_log(s, "P") for s in SEASONS + [CURRENT_SEASON])


HUSTLE_SEASONS = SEASONS[1:] + [CURRENT_SEASON]  # 2015-16 regular season has only 2 tracked games


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
    """Every Sunday inside the season, plus the final day. A game uses the latest snapshot BEFORE its date.
    Current season: only Sundays before today (each file stays valid forever), so glue lags up to a week."""
    dates = pd.to_datetime(game_log(season, "T")["GAME_DATE"])
    if dates.empty:
        return []
    if season == CURRENT_SEASON:
        return list(pd.date_range(dates.min(), pd.Timestamp.today().normalize() - pd.Timedelta(days=1), freq="W-SUN"))
    sundays = pd.date_range(dates.min(), dates.max(), freq="W-SUN")
    return sorted(set(sundays) | {dates.max()})


def all_hustle_snapshots() -> pd.DataFrame:
    return _concat(hustle_snapshot(s, d).assign(SEASON_STR=s, DATE_TO=d)
                   for s in HUSTLE_SEASONS for d in snapshot_dates(s))


def standings(season: str, sleep: float = 1.5) -> pd.DataFrame:
    """Official standings: conference, division, W/L, seed (PlayoffRank). Current season refreshed when stale."""
    path = RAW / f"standings_{season}.parquet"
    if (path.exists() and season != CURRENT_SEASON) or (season == CURRENT_SEASON and _fresh(path)):
        return pd.read_parquet(path)
    df = leaguestandingsv3.LeagueStandingsV3(season=season, timeout=60).get_data_frames()[0]
    time.sleep(sleep)
    df.to_parquet(path, index=False)
    return df


def postseason_log(season: str, kind: str = "Playoffs", sleep: float = 1.5) -> pd.DataFrame:
    """Team rows for "Playoffs" (game id 004...) or "PlayIn" (005...) games. Completed seasons only."""
    path = RAW / f"{kind.lower()}_gamelog_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = leaguegamelog.LeagueGameLog(season=season, season_type_all_star=kind,
                                     player_or_team_abbreviation="T", timeout=60).get_data_frames()[0]
    time.sleep(sleep)
    df.to_parquet(path, index=False)
    return df


def schedule() -> pd.DataFrame:
    """Current-season regular-season schedule, one row per game, cached per day."""
    path = RAW / f"schedule_{pd.Timestamp.today():%Y-%m-%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = scheduleleaguev2.ScheduleLeagueV2(season=CURRENT_SEASON, timeout=60).get_data_frames()[0]
    df = df[df["gameId"].str.startswith("002")]  # 001 = preseason, 006 = NBA Cup final placeholder
    df = pd.DataFrame({
        "GAME_ID": df["gameId"], "GAME_DATE": pd.to_datetime(df["gameDateEst"]).dt.tz_localize(None).dt.normalize(),
        "TIP_UTC": df["gameDateTimeUTC"], "HOME": df["homeTeam_teamTricode"], "AWAY": df["awayTeam_teamTricode"],
        "NEUTRAL": df["isNeutral"].astype(bool), "ARENA": df["arenaName"] + ", " + df["arenaCity"],
        "STATUS": df["gameStatusText"]})
    df.to_parquet(path, index=False)
    return df


def current_rosters(sleep: float = 1.0) -> pd.DataFrame:
    """Today's rosters for all 30 teams (30 calls), cached per day."""
    path = RAW / f"rosters_{pd.Timestamp.today():%Y-%m-%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    out = []
    for t in teams.get_teams():
        out.append(commonteamroster.CommonTeamRoster(team_id=t["id"], season=CURRENT_SEASON, timeout=60)
                   .get_data_frames()[0].assign(TEAM_ABBREVIATION=t["abbreviation"], TEAM_NAME=t["full_name"]))
        time.sleep(sleep)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(path, index=False)
    return df


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
