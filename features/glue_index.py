"""Glue guy index (user's own, unvalidated design). Must earn its place by ablation in models/train.py.

Player:  Glue = 0.22 Z(deflections) + 0.18 Z(screen ast) + 0.15 Z(loose balls) + 0.15 Z(contested shots)
              + 0.05 Z(charges) + 0.05 Z(box outs) + 0.20 Z(shrunk on/off)      (hustle stats per 36)
Team:    minutes-weighted mean Glue over the top 10 (by MPG) players who actually played the game.

Deviations from the handoff, all data-driven:
- On/off from box-score plus/minus (per 48), not teamplayeronoffsummary: same idea, zero API calls, exact dates.
- Hustle starts 2016-17 (2015-16 regular season has 2 tracked games); box outs start 2017-18 (Z=0 before).
- Weekly snapshots: a game uses the latest snapshot dated strictly before game day.
- Early season: blended with last season's final value, weight G / (G + PRIOR_GAMES).

Inspect:  python -m features.glue_index           (top glue players, latest season)
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "glue.parquet"
WEIGHTS = {"DEFLECTIONS": 0.22, "SCREEN_ASSISTS": 0.18, "LOOSE_BALLS_RECOVERED": 0.15,
           "CONTESTED_SHOTS": 0.15, "CHARGES_DRAWN": 0.05, "BOX_OUTS": 0.05, "ONOFF_SHRUNK": 0.20}
ONOFF_SHRINK_MIN = 1000  # tunable guess from the handoff
PRIOR_GAMES = 15  # tunable guess: games of current data worth as much as last season's final value
MIN_G, MIN_MPG, MIN_G_EARLY = 15, 10, 5
TOP_N = 10


def season_of(s: pd.Series) -> pd.Series:
    return s.astype(str).str[1:].astype(int)  # "22024" -> 2024


def onoff_to_date(pl: pd.DataFrame, tg: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date on/off (+/- per 48 on court minus off court) per player at each snapshot date."""
    team = tg[["GAME_ID", "TEAM_ID", "PLUS_MINUS", "MIN"]].rename(columns={"PLUS_MINUS": "TEAM_PM", "MIN": "TEAM_MIN"})
    pg = pl.merge(team, on=["GAME_ID", "TEAM_ID"])
    pg["ON_PM"], pg["ON_MIN"] = pg["PLUS_MINUS"], pg["MIN"]
    pg["OFF_PM"], pg["OFF_MIN"] = pg["TEAM_PM"] - pg["PLUS_MINUS"], pg["TEAM_MIN"] / 5 - pg["MIN"]
    pg = pg.sort_values("GAME_DATE")
    cols = ["ON_PM", "ON_MIN", "OFF_PM", "OFF_MIN"]
    pg[cols] = pg.groupby(["PLAYER_ID", "SEASON"])[cols].cumsum()
    m = pd.merge_asof(snaps.sort_values("DATE_TO"), pg[["PLAYER_ID", "SEASON", "GAME_DATE"] + cols],
                      left_on="DATE_TO", right_on="GAME_DATE", by=["PLAYER_ID", "SEASON"], direction="backward")
    on = m["ON_PM"] / m["ON_MIN"].where(m["ON_MIN"] > 0)
    off = m["OFF_PM"] / m["OFF_MIN"].where(m["OFF_MIN"] > 0)
    m["ONOFF"] = (48 * (on - off)).fillna(0.0)
    return m.drop(columns=["GAME_DATE"] + cols)


def player_glue(snaps: pd.DataFrame) -> pd.DataFrame:
    """Per player per snapshot: current-season Glue (z-scored vs qualified players that snapshot)."""
    s = snaps.copy()
    s["MPG"] = s["MIN"] / s["G"]
    for c in WEIGHTS:
        if c != "ONOFF_SHRUNK":
            s[c] = 36 * s[c].fillna(0) / s["MIN"].where(s["MIN"] > 0)
    s["ONOFF_SHRUNK"] = s["ONOFF"] * s["MIN"] / (s["MIN"] + ONOFF_SHRINK_MIN)
    s["GLUE_CUR"] = 0.0
    for _, idx in s.groupby("DATE_TO").groups.items():
        snap = s.loc[idx]
        qual = snap[(snap["G"] >= MIN_G) & (snap["MPG"] >= MIN_MPG)]
        if len(qual) < 100:  # early weeks: too few qualified players to z-score against
            continue
        glue = sum(w * ((snap[c] - qual[c].mean()) / qual[c].std()).fillna(0) if qual[c].std() > 0 else 0.0
                   for c, w in WEIGHTS.items())
        s.loc[idx, "GLUE_CUR"] = glue
    s["W_CUR"] = np.where((s["G"] >= MIN_G_EARLY) & (s["MPG"] >= MIN_MPG) & s["GLUE_CUR"].ne(0), s["G"], 0)
    return s


def blend_prior(s: pd.DataFrame) -> pd.DataFrame:
    """Blend each snapshot with last season's final value; add a pre-season snapshot holding only the prior."""
    final = s[s["DATE_TO"] == s.groupby("SEASON")["DATE_TO"].transform("max")]
    prior = final[["PLAYER_ID", "SEASON", "GLUE_CUR", "MPG"]].assign(SEASON=final["SEASON"] + 1)
    prior = prior.rename(columns={"GLUE_CUR": "GLUE_PRIOR", "MPG": "MPG_PRIOR"})
    s = s.merge(prior, on=["PLAYER_ID", "SEASON"], how="left")
    s["GLUE"] = (s["W_CUR"] * s["GLUE_CUR"] + PRIOR_GAMES * s["GLUE_PRIOR"].fillna(0)) / (s["W_CUR"] + PRIOR_GAMES)
    # Pre-season snapshot (dated before any game): prior only, so week-1 games still get a value
    starts = s.groupby("SEASON")["DATE_TO"].min() - pd.Timedelta(days=7)
    pre = prior[prior["SEASON"].isin(starts.index)].assign(DATE_TO=lambda d: d["SEASON"].map(starts))
    pre = pre.rename(columns={"GLUE_PRIOR": "GLUE", "MPG_PRIOR": "MPG"})
    return pd.concat([s[["PLAYER_ID", "SEASON", "DATE_TO", "PLAYER_NAME", "G", "MPG", "GLUE"]], pre],
                     ignore_index=True)


def team_glue(pl: pd.DataFrame, glue: pd.DataFrame) -> pd.DataFrame:
    """Per (GAME_ID, TEAM_ID): MPG-weighted Glue of the top TOP_N players who played, from snapshots before the game."""
    m = pd.merge_asof(pl.sort_values("GAME_DATE"), glue.drop(columns="PLAYER_NAME").sort_values("DATE_TO"),
                      left_on="GAME_DATE", right_on="DATE_TO", by=["PLAYER_ID", "SEASON"],
                      direction="backward", allow_exact_matches=False)
    m = m.dropna(subset=["MPG", "GLUE"])
    m = m.sort_values("MPG", ascending=False).groupby(["GAME_ID", "TEAM_ID"]).head(TOP_N)
    m["W"] = m["MPG"] * m["GLUE"]
    out = m.groupby(["GAME_ID", "TEAM_ID"])[["W", "MPG"]].sum()
    return (out["W"] / out["MPG"]).rename("GLUE").reset_index()


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    from pipeline.fetch_data import all_hustle_snapshots, all_player_game_logs, all_team_game_logs
    pl = all_player_game_logs()
    pl = pl.assign(GAME_DATE=pd.to_datetime(pl["GAME_DATE"]), SEASON=season_of(pl["SEASON_ID"]))
    snaps = all_hustle_snapshots()
    snaps = snaps.assign(SEASON=snaps["SEASON_STR"].str[:4].astype(int), DATE_TO=pd.to_datetime(snaps["DATE_TO"]))
    snaps = snaps.drop_duplicates(["PLAYER_ID", "DATE_TO"])  # one row per player per snapshot
    glue = blend_prior(player_glue(onoff_to_date(pl, all_team_game_logs(), snaps)))
    return glue, team_glue(pl, glue)


def load() -> pd.DataFrame:
    """Cached team-game glue; rebuilt whenever any raw file is newer than the cache."""
    from pipeline.fetch_data import RAW
    newest_raw = max(f.stat().st_mtime for f in RAW.rglob("*.parquet"))
    if not OUT.exists() or OUT.stat().st_mtime < newest_raw:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        build()[1].to_parquet(OUT, index=False)
    return pd.read_parquet(OUT)


if __name__ == "__main__":
    glue, team = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    team.to_parquet(OUT, index=False)
    last = glue[glue["DATE_TO"] == glue["DATE_TO"].max()]
    print(f"{len(team)} team-games with glue. Top 15 players, snapshot {last['DATE_TO'].iloc[0]:%Y-%m-%d}:")
    last = last[(last["G"] >= MIN_G) & (last["MPG"] >= MIN_MPG)]  # qualified only
    print(last.nlargest(15, "GLUE")[["PLAYER_NAME", "G", "MPG", "GLUE"]].round(2).to_string(index=False))
    print("\nBottom 5:")
    print(last.nsmallest(5, "GLUE")[["PLAYER_NAME", "G", "MPG", "GLUE"]].round(2).to_string(index=False))
