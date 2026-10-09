"""Season inputs as of any date: teams (conference/division), games played so far, games remaining.

Past seasons come from the game logs (the schedule was known in advance; results before `as_of` only).
The current season comes from the published schedule plus results so far. Until the Cup group stage
ends, 30 games (60 team slots: unscheduled games + knockout games with TBD teams) are missing; they're
filled as placeholder games vs a league-average opponent (AWAY_ID = 0) so every team plays 82.
"""
import numpy as np
import pandas as pd

from pipeline.fetch_data import CURRENT_SEASON, game_log, schedule, standings

GAMES_PER_TEAM = 82


def teams(season: str) -> pd.DataFrame:
    st = standings(season)
    return pd.DataFrame({"TEAM_ID": st["TeamID"].astype(int), "CONF": st["Conference"], "DIV": st["Division"],
                         "NAME": st["TeamCity"] + " " + st["TeamName"]}).sort_values("TEAM_ID").reset_index(drop=True)


def games(season: str) -> pd.DataFrame:
    """One row per regular-season game: GAME_ID, DATE, HOME_ID, AWAY_ID, NEUTRAL, HOME_WIN (NaN = not played)."""
    log = game_log(season, "T")
    played = pd.DataFrame()
    if not log.empty:
        log = log.assign(GAME_DATE=pd.to_datetime(log["GAME_DATE"]), IS_HOME=log["MATCHUP"].str.contains("vs.", regex=False))
        log["NEUTRAL"] = log.groupby("GAME_ID")["IS_HOME"].transform("sum").eq(0)
        # neutral games: lower TEAM_ID takes the home slot (same convention as features/build.py)
        lo = log.groupby("GAME_ID")["TEAM_ID"].transform("min")
        h = log[log["IS_HOME"] | (log["NEUTRAL"] & (log["TEAM_ID"] == lo))]
        a = log[~log.index.isin(h.index)]
        played = h[["GAME_ID", "GAME_DATE", "TEAM_ID", "NEUTRAL", "WL"]].merge(
            a[["GAME_ID", "TEAM_ID"]], on="GAME_ID", suffixes=("_H", "_A"))
        played = pd.DataFrame({"GAME_ID": played["GAME_ID"], "DATE": played["GAME_DATE"],
                               "HOME_ID": played["TEAM_ID_H"], "AWAY_ID": played["TEAM_ID_A"],
                               "NEUTRAL": played["NEUTRAL"], "HOME_WIN": (played["WL"] == "W").astype(float)})
    if season != CURRENT_SEASON:
        return played.sort_values(["DATE", "GAME_ID"]).reset_index(drop=True)

    from nba_api.stats.static import teams as static
    tid = {t["abbreviation"]: t["id"] for t in static.get_teams()}
    sch = schedule()
    # Cup knockout games (Dec) have no teams yet: skip them, the placeholder fill covers those slots
    sch = sch[~sch["GAME_ID"].isin(played.get("GAME_ID", [])) & sch["HOME"].notna() & sch["AWAY"].notna()]
    future = pd.DataFrame({"GAME_ID": sch["GAME_ID"], "DATE": sch["GAME_DATE"], "HOME_ID": sch["HOME"].map(tid),
                           "AWAY_ID": sch["AWAY"].map(tid), "NEUTRAL": sch["NEUTRAL"], "HOME_WIN": np.nan})
    return pd.concat([d for d in (played, future) if not d.empty]).sort_values(["DATE", "GAME_ID"]).reset_index(drop=True)


def as_of(season: str, date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(teams, played games before `date`, remaining games incl. placeholders). Results on/after `date` are hidden."""
    tm, g = teams(season), games(season)
    date = pd.Timestamp(date)
    played = g[(g["DATE"] < date) & g["HOME_WIN"].notna()]
    remaining = g[~g.index.isin(played.index)].assign(HOME_WIN=np.nan)
    count = pd.concat([g["HOME_ID"], g["AWAY_ID"]]).value_counts()
    # Past seasons are complete in the logs (2020-21 had 72 games); only the current one needs filling
    fill = [] if season != CURRENT_SEASON else [{"GAME_ID": f"TBD{t}_{i}", "DATE": pd.NaT, "HOME_ID": t, "AWAY_ID": 0, "NEUTRAL": True, "HOME_WIN": np.nan}
            for t in tm["TEAM_ID"] for i in range(GAMES_PER_TEAM - int(count.get(t, 0)))]
    if fill:
        remaining = pd.concat([remaining, pd.DataFrame(fill)], ignore_index=True)
    return tm, played.reset_index(drop=True), remaining.reset_index(drop=True)


def records(tm: pd.DataFrame, played: pd.DataFrame) -> pd.DataFrame:
    """W/L per team from played games."""
    w = pd.concat([played.loc[played["HOME_WIN"] == 1, "HOME_ID"], played.loc[played["HOME_WIN"] == 0, "AWAY_ID"]])
    l = pd.concat([played.loc[played["HOME_WIN"] == 0, "HOME_ID"], played.loc[played["HOME_WIN"] == 1, "AWAY_ID"]])  # noqa: E741
    return tm.assign(W=tm["TEAM_ID"].map(w.value_counts()).fillna(0).astype(int),
                     L=tm["TEAM_ID"].map(l.value_counts()).fillna(0).astype(int))


if __name__ == "__main__":
    # Verify against the official standings, then show the current season's inputs
    for season in ("2023-24", "2024-25", "2025-26"):
        tm, played, rem = as_of(season, "2100-01-01")
        rec = records(tm, played).merge(standings(season)[["TeamID", "WINS", "LOSSES"]], left_on="TEAM_ID", right_on="TeamID")
        assert (rec["W"] == rec["WINS"]).all() and (rec["L"] == rec["LOSSES"]).all(), season
        assert len(rem) == 0 and (rec["W"] + rec["L"] == GAMES_PER_TEAM).all()
        print(f"{season}: records match official standings for all 30 teams")
    tm, played, rem = as_of(CURRENT_SEASON, pd.Timestamp.today())
    n_fill = rem["GAME_ID"].str.startswith("TBD").sum()
    print(f"{CURRENT_SEASON}: {len(played)} played, {len(rem)} remaining ({n_fill} placeholder games for the unscheduled Cup slots)")
    per_team = pd.concat([rem["HOME_ID"], rem["AWAY_ID"][rem["AWAY_ID"] != 0]]).value_counts()
    assert (per_team == GAMES_PER_TEAM).all(), per_team[per_team != GAMES_PER_TEAM]
    print(tm.groupby("CONF").size().to_dict(), "| every team has 82 games")
