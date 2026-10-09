"""Team strength for the season simulation: Elo as of a date, with offseason regression, an optional
roster-change adjustment, and a fast Elo → single-game win probability.

Known at `date` only: Elo uses games before `date`. Roster change compares last season's closing
rotation with the new one, both valued by last season's Game Score.
Backtest caveat: for past seasons the "new roster" is who played in a team's first 5 games, which
leaks a little (opening-night availability). Live, it's today's official rosters.
"""
import tomllib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from features import availability
from features.build import ELO_MEAN, add_elo, team_games
from pipeline.fetch_data import CURRENT_SEASON, all_player_game_logs, all_team_game_logs

CONFIG = Path(__file__).resolve().parent / "config.toml"
TOP_N, OLD_ROSTER_GAMES, NEW_ROSTER_GAMES, MIN_GAMES_FOR_VALUE = 8, 15, 5, 20


def load_config(**overrides) -> dict:
    return {**tomllib.loads(CONFIG.read_text()), **overrides}


@lru_cache(maxsize=8)
def _elo_games(regression: float) -> pd.DataFrame:
    tg = add_elo(team_games(all_team_game_logs()), carry=1 - regression)
    return tg[["TEAM_ID", "GAME_DATE", "SEASON", "ELO_POST"]].sort_values("GAME_DATE")


@lru_cache(maxsize=1)
def _players() -> pd.DataFrame:
    pl = all_player_game_logs()
    return pl.assign(GAME_DATE=pd.to_datetime(pl["GAME_DATE"]), GMSC=availability.game_score(pl),
                     SEASON=pl["SEASON_ID"].astype(str).str[1:].astype(int))


def roster_delta(year: int, team_ids) -> pd.Series:
    """Change in top-8 player value (last season's Game Score/game) from last season's roster to this one."""
    pl = _players()
    prev = pl[pl["SEASON"] == year - 1]
    gp = prev.groupby("PLAYER_ID")["GMSC"].agg(["mean", "size"])
    value = gp.loc[gp["size"] >= MIN_GAMES_FOR_VALUE, "mean"]

    def talent(ids) -> float:
        return float(value.reindex(list(set(ids))).dropna().nlargest(TOP_N).sum())

    def last_games(df, team, n, first=False):
        dates = sorted(df.loc[df["TEAM_ID"] == team, "GAME_DATE"].unique())
        keep = dates[:n] if first else dates[-n:]
        return df[(df["TEAM_ID"] == team) & df["GAME_DATE"].isin(keep)]["PLAYER_ID"]

    cur = pl[pl["SEASON"] == year]
    if year == int(CURRENT_SEASON[:4]) and cur.empty:
        from pipeline.fetch_data import current_rosters
        ros = current_rosters()
        new = {t: ros.loc[ros["TeamID"] == t, "PLAYER_ID"] for t in team_ids}
    else:
        new = {t: last_games(cur, t, NEW_ROSTER_GAMES, first=True) for t in team_ids}
    return pd.Series({t: talent(new[t]) - talent(last_games(prev, t, OLD_ROSTER_GAMES)) for t in team_ids})


def team_ratings(season: str, date, team_ids, cfg: dict) -> pd.DataFrame:
    """Per team: ELO (rating used by the sim), GAMES (played this season before date), ROSTER_ADJ."""
    year, date = int(season[:4]), pd.Timestamp(date)
    eg = _elo_games(cfg["regression"])
    rows = []
    for t in team_ids:
        hist = eg[(eg["TEAM_ID"] == t) & (eg["GAME_DATE"] < date)]
        last = hist.iloc[-1]
        elo = float(last["ELO_POST"])
        if last["SEASON"] < year:  # no games yet this season: apply the offseason regression now
            elo = (1 - cfg["regression"]) * elo + cfg["regression"] * ELO_MEAN
        rows.append({"TEAM_ID": t, "ELO_BASE": elo, "GAMES": int((hist["SEASON"] == year).sum())})
    r = pd.DataFrame(rows).set_index("TEAM_ID")
    r["ROSTER_ADJ"] = 0.0
    if cfg["roster_k"]:
        fade = np.clip(1 - r["GAMES"] / cfg["roster_fade_games"], 0, 1)
        r["ROSTER_ADJ"] = cfg["roster_k"] * roster_delta(year, list(r.index)) * fade
    r["ELO"] = r["ELO_BASE"] + r["ROSTER_ADJ"]
    return r


def injury_adjust(ratings: pd.DataFrame, missing_value: pd.Series, cfg: dict) -> pd.DataFrame:
    """Long-term absences only (live): Elo points off for the Game Score value of players out for weeks.
    ponytail: same Elo-per-value rate as roster changes; not backtestable (no historical injury reports)."""
    return ratings.assign(ELO=ratings["ELO"] - cfg["roster_k"] * missing_value.reindex(ratings.index).fillna(0))


def win_prob(elo_home, elo_away, neutral, home_elo: float):
    """Fast single-game home win probability (vectorized)."""
    hca = np.where(neutral, 0.0, home_elo)
    return 1 / (1 + 10 ** (-(np.asarray(elo_home) - np.asarray(elo_away) + hca) / 400))


if __name__ == "__main__":
    import joblib

    from models.baselines import score, split
    cfg = load_config()
    # 1) fast path vs the full game model on the held-out test seasons
    art = joblib.load("models/model.joblib")
    g = split(pd.read_parquet("data/processed/games.parquet").dropna(subset=art["cols"]), "test")
    fast = win_prob(g["ELO_H"], g["ELO_A"], g["NEUTRAL"].eq(1), cfg["home_elo"])
    full = art["model"].predict_proba(g[art["cols"]])[:, 1]
    y = g["HOME_WIN"].to_numpy()
    print(f"fast path vs full model on {len(g)} test games: mean |diff| {np.abs(fast - full).mean():.3f}, "
          f"corr {np.corrcoef(fast, full)[0, 1]:.3f}")
    print(f"  log loss fast {score(y, fast)['log_loss']:.4f} | full {score(y, full)['log_loss']:.4f}")
    # 2) preseason ratings now
    from season.inputs import teams
    tm = teams(CURRENT_SEASON)
    r = team_ratings(CURRENT_SEASON, pd.Timestamp.today(), tm["TEAM_ID"], load_config(roster_k=1.0))
    r = r.join(tm.set_index("TEAM_ID")["NAME"]).sort_values("ELO", ascending=False)
    print(f"\n{CURRENT_SEASON} preseason (roster_k=1 for display): mean Elo {r['ELO_BASE'].mean():.1f}")
    print(r[["NAME", "ELO_BASE", "ROSTER_ADJ", "GAMES"]].round(1).head(8).to_string(index=False))
    print("biggest roster changes (top-8 value delta):")
    print(r["ROSTER_ADJ"].round(1).rename(r["NAME"]).sort_values().iloc[[0, 1, 2, -3, -2, -1]].to_string())
