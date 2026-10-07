"""Turn raw team game logs into one row per game with pre-game features only.

Run from the project root:  python -m features.build
"""
from pathlib import Path

import numpy as np
import pandas as pd

from features import availability, glue_index
from pipeline.fetch_data import all_team_game_logs

OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "games.parquet"
WINDOW = 10

# Elo constants from FiveThirtyEight's NBA model, except HCA: 100 -> 60 (best train log loss among 40/60/80/100;
# home edge shrank after 2019). ponytail: K and carry-over untuned.
ELO_K, ELO_HCA, ELO_MEAN, ELO_CARRY = 20.0, 60.0, 1505.0, 0.75


def team_games(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, with opponent stats and per-game ratings."""
    df = raw.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON"] = df["SEASON_ID"].astype(str).str[1:].astype(int)  # "22024" -> 2024 (= 2024-25)
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    df["NEUTRAL"] = df.groupby("GAME_ID")["IS_HOME"].transform("sum").eq(0).astype(int)
    df["POSS_RAW"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]

    stats = ["TEAM_ID", "PTS", "FGM", "FGA", "FG3M", "FTA", "OREB", "DREB", "TOV", "POSS_RAW"]
    opp = df[["GAME_ID"] + stats].rename(columns={c: "OPP_" + c for c in stats})
    df = df.merge(opp, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]].copy()

    # Both teams share the same possession count: average the two estimates.
    df["POSS"] = (df["POSS_RAW"] + df["OPP_POSS_RAW"]) / 2
    df["PACE"] = df["POSS"] * 240 / df["MIN"]  # per 48 minutes (MIN is team minutes, 240 in regulation)
    df["ORTG"] = 100 * df["PTS"] / df["POSS"]
    df["DRTG"] = 100 * df["OPP_PTS"] / df["POSS"]
    df["NET"] = df["ORTG"] - df["DRTG"]
    df["MARGIN"] = df["PTS"] - df["OPP_PTS"]
    # Four Factors, offense and defense
    df["EFG"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]
    df["TOV_PCT"] = df["TOV"] / df["POSS"]
    df["ORB_PCT"] = df["OREB"] / (df["OREB"] + df["OPP_DREB"])
    df["FT_RATE"] = df["FTA"] / df["FGA"]
    df["OPP_EFG"] = (df["OPP_FGM"] + 0.5 * df["OPP_FG3M"]) / df["OPP_FGA"]
    df["OPP_TOV_PCT"] = df["OPP_TOV"] / df["POSS"]
    df["DRB_PCT"] = df["DREB"] / (df["DREB"] + df["OPP_OREB"])
    df["OPP_FT_RATE"] = df["OPP_FTA"] / df["OPP_FGA"]
    return df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)


ROLL_COLS = ["ORTG", "DRTG", "NET", "PACE", "MARGIN", "EFG", "TOV_PCT", "ORB_PCT", "FT_RATE",
             "OPP_EFG", "OPP_TOV_PCT", "DRB_PCT", "OPP_FT_RATE"]


def add_pregame_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Rolling form and rest. shift(1) makes every value use only games before this one.

    ponytail: rolling window runs across season boundaries (early-season games lean on last
    season's tail); Elo's carry-over handles roster churn better. Revisit with a season blend.
    """
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    by_team = tg.groupby("TEAM_ID")
    for c in ROLL_COLS:
        tg[f"R{WINDOW}_{c}"] = by_team[c].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=3).mean())

    prev = by_team["GAME_DATE"].shift(1)
    tg["REST"] = (tg["GAME_DATE"] - prev).dt.days.clip(upper=7).fillna(7)  # season opener -> 7
    tg["B2B"] = tg["REST"].eq(1).astype(int)
    # 3 games in 4 nights: this game plus 2 more in the previous 3 days
    tg["G3IN4"] = ((tg["GAME_DATE"] - by_team["GAME_DATE"].shift(2)).dt.days <= 3).astype(int)
    tg["GAMES_PLAYED"] = tg.groupby(["TEAM_ID", "SEASON"]).cumcount()
    return tg


def add_elo(tg: pd.DataFrame) -> pd.DataFrame:
    """Pre-game Elo per team-game (FiveThirtyEight style, margin-of-victory multiplier)."""
    elo: dict[int, float] = {}
    last_season: dict[int, int] = {}
    pre = {}
    games = tg[tg["IS_HOME"].eq(1) | (tg["NEUTRAL"].eq(1) & (tg["TEAM_ID"] < tg["OPP_TEAM_ID"]))]
    for g in games.sort_values(["GAME_DATE", "GAME_ID"]).itertuples():
        a, b = g.TEAM_ID, g.OPP_TEAM_ID
        for t in (a, b):
            if t in last_season and last_season[t] != g.SEASON:
                elo[t] = ELO_CARRY * elo[t] + (1 - ELO_CARRY) * ELO_MEAN
            last_season[t] = g.SEASON
        ea, eb = elo.get(a, ELO_MEAN), elo.get(b, ELO_MEAN)
        pre[(g.GAME_ID, a)], pre[(g.GAME_ID, b)] = ea, eb
        hca = 0.0 if g.NEUTRAL else ELO_HCA
        p_a = 1 / (1 + 10 ** (-(ea + hca - eb) / 400))
        won = 1.0 if g.MARGIN > 0 else 0.0
        winner_diff = (ea + hca - eb) if won else (eb - ea - hca)
        mult = (abs(g.MARGIN) + 3) ** 0.8 / (7.5 + 0.006 * winner_diff)
        delta = ELO_K * mult * (won - p_a)
        elo[a], elo[b] = ea + delta, eb - delta
    tg["ELO"] = [pre[(gid, t)] for gid, t in zip(tg["GAME_ID"], tg["TEAM_ID"])]
    return tg


def to_games(tg: pd.DataFrame) -> pd.DataFrame:
    """One row per game: home-team features, away-team features, and home-minus-away diffs."""
    feat = [f"R{WINDOW}_{c}" for c in ROLL_COLS] + ["REST", "B2B", "G3IN4", "GAMES_PLAYED", "ELO",
                                                   "MISSING_GMSC", "MISSING_TOP", "MISSING_N", "GLUE"]
    # Neutral games: the lower TEAM_ID plays the "home" slot, with NEUTRAL=1 telling the model there's no edge.
    is_h = tg["IS_HOME"].eq(1) | (tg["NEUTRAL"].eq(1) & (tg["TEAM_ID"] < tg["OPP_TEAM_ID"]))
    keep = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"] + feat
    h = tg[is_h][["GAME_DATE", "SEASON", "NEUTRAL", "PTS", "OPP_PTS"] + keep]
    a = tg[~is_h][keep]
    g = h.merge(a, on="GAME_ID", suffixes=("_H", "_A"))
    g = g.rename(columns={"PTS": "PTS_H", "OPP_PTS": "PTS_A"})
    g["HOME_WIN"] = (g["PTS_H"] > g["PTS_A"]).astype(int)
    g["MARGIN"] = g["PTS_H"] - g["PTS_A"]
    for c in feat:
        g[f"D_{c}"] = g[f"{c}_H"] - g[f"{c}_A"]
    return g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    tg = add_elo(add_pregame_features(team_games(all_team_game_logs())))
    av = availability.load().drop(columns="MISSING_NAMES")
    tg = tg.merge(av, on=["GAME_ID", "TEAM_ID"], how="left", validate="1:1")
    assert tg["MISSING_GMSC"].notna().all(), "team-game without availability row"
    tg = tg.merge(glue_index.load(), on=["GAME_ID", "TEAM_ID"], how="left", validate="1:1")  # NaN before 2016-17
    return tg, to_games(tg)


def check_no_leakage(tg: pd.DataFrame, n: int = 200) -> None:
    """Recompute rolling net rating from strictly earlier games for random rows."""
    rng = np.random.default_rng(0)
    for i in rng.choice(len(tg), n, replace=False):
        row = tg.iloc[i]
        hist = tg[(tg["TEAM_ID"] == row["TEAM_ID"]) & (tg["GAME_DATE"] < row["GAME_DATE"])]
        last = hist.sort_values("GAME_DATE")["NET"].tail(WINDOW)
        expected = last.mean() if len(last) >= 3 else np.nan
        got = row[f"R{WINDOW}_NET"]
        assert (np.isnan(expected) and np.isnan(got)) or np.isclose(expected, got), (row["GAME_ID"], expected, got)


if __name__ == "__main__":
    tg, games = build()
    check_no_leakage(tg)
    assert games["GAME_ID"].is_unique and len(games) == tg["GAME_ID"].nunique()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(OUT, index=False)
    print(f"leakage check ok; {len(games)} games -> {OUT}")
    print(games[["GAME_DATE", "TEAM_ABBREVIATION_H", "TEAM_ABBREVIATION_A", "PTS_H", "PTS_A",
                 "ELO_H", "ELO_A", "D_R10_NET", "REST_H", "REST_A"]].tail(5).to_string())
