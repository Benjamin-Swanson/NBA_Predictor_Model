"""Backtest the season model on completed seasons, then tune its knobs.

For each season and checkpoint (opening night, ~20 games in, ~50 games in) the model only sees games before that date.
Outcomes come from the final standings and the playoff logs.

Seasons: 2016-17 .. 2025-26, minus 2019-20 (bubble: suspended season, seeding games, one-off play-in).
Play-in metrics use 2020-21 onward (the play-in didn't exist before).

Tuning is staged to stay fast:
  1. regression, roster_k : expected wins, no simulation needed (noise doesn't move the mean much)
  2. noise_preseason       : preseason checkpoint, simulated (coverage of the 10-90% win range, playoff log loss)
  3. noise_half_games      : ~20- and ~50-game checkpoints, simulated
Caveat: knobs are tuned and scored on the same 9 seasons (3 knobs, so overfitting is limited, not zero).

Run:  python -m season.backtest            (writes data/season_backtest.json)
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fetch_data import postseason_log, standings
from season import inputs
from season.ratings import load_config, team_ratings, win_prob
from season.simulate import run, summarize

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2026) if y != 2019]
OUT = Path(__file__).resolve().parent.parent / "data" / "season_backtest.json"


def checkpoints(season: str) -> dict:
    """Opening night, then the dates when the league has played ~20 and ~50 games per team
    (by games played rather than calendar date: 2020-21 started Dec 22)."""
    d = inputs.games(season)["DATE"].sort_values().reset_index(drop=True)
    return {"preseason": d.iloc[0], "g20": d.iloc[20 * 15], "g50": d.iloc[50 * 15]}


def outcomes(season: str) -> pd.DataFrame:
    st = standings(season)
    po = postseason_log(season, "Playoffs")
    rnd = po["GAME_ID"].str[7].astype(int)
    champ = po.groupby("TEAM_ID")["WL"].apply(lambda s: (s == "W").sum()).idxmax()
    out = pd.DataFrame({"TEAM_ID": st["TeamID"].astype(int), "WINS": st["WINS"], "SEED": st["PlayoffRank"].astype(int)})
    out["PLAYOFFS"] = out["TEAM_ID"].isin(po["TEAM_ID"]).astype(int)
    out["PLAYIN"] = (out["SEED"].between(7, 10) if int(season[:4]) >= 2020 else pd.Series(np.nan, out.index))
    out["CONF_FINALS"] = out["TEAM_ID"].isin(po.loc[rnd >= 3, "TEAM_ID"]).astype(int)
    out["FINALS"] = out["TEAM_ID"].isin(po.loc[rnd == 4, "TEAM_ID"]).astype(int)
    out["TITLE"] = (out["TEAM_ID"] == champ).astype(int)
    return out


def ll(y, p):
    p = np.clip(np.asarray(p, float), 0.001, 0.999)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_wins(season, date, cfg) -> pd.DataFrame:
    """Mean-only projection (no noise): current wins + sum of single-game win probabilities."""
    tm, played, rem = inputs.as_of(season, date)
    r = team_ratings(season, date, tm["TEAM_ID"], cfg)["ELO"]
    p = win_prob(rem["HOME_ID"].map(r), rem["AWAY_ID"].map(r), rem["NEUTRAL"].to_numpy(bool), cfg["home_elo"])
    ew = pd.concat([pd.Series(p, index=rem["HOME_ID"]), pd.Series(1 - p, index=rem["AWAY_ID"])]).groupby(level=0).sum()
    rec = inputs.records(tm, played).set_index("TEAM_ID")
    return (rec["W"] + ew.reindex(rec.index, fill_value=0)).rename("EXP_W").reset_index()


def simulate_scores(season, date, cfg) -> dict:
    tm, played, rem = inputs.as_of(season, date)
    ratings = team_ratings(season, date, tm["TEAM_ID"], cfg)
    df = summarize(run(tm, played, rem, ratings, cfg), tm, inputs.records(tm, played), ratings)
    df = df.merge(outcomes(season), on="TEAM_ID")
    pi = df.dropna(subset=["PLAYIN"])
    return {"win_mae": float((df["WINS_P50"] - df["WINS"]).abs().mean()),
            "win_cover_80": float(df["WINS"].between(df["WINS_P10"], df["WINS_P90"]).mean()),
            "playoffs_ll": ll(df["PLAYOFFS"], df["P_PLAYOFFS"]), "playoffs_brier": float(((df["P_PLAYOFFS"] - df["PLAYOFFS"]) ** 2).mean()),
            "playin_ll": ll(pi["PLAYIN"], pi["P_PLAYIN"]) if len(pi) else np.nan,
            "finals_ll": ll(df["FINALS"], df["P_FINALS"]),
            "title_ll": float(-np.log(max(df.loc[df["TITLE"] == 1, "P_TITLE"].iloc[0], 0.001))),
            "champ_p": float(df.loc[df["TITLE"] == 1, "P_TITLE"].iloc[0]),
            "top_pick": df.iloc[0]["NAME"], "champ": df.loc[df["TITLE"] == 1, "NAME"].iloc[0],
            "rows": df}


def mean_scores(rows: list[dict]) -> dict:
    keys = ["win_mae", "win_cover_80", "playoffs_ll", "playoffs_brier", "playin_ll", "finals_ll", "title_ll", "champ_p"]
    return {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in keys}


def main():
    base = load_config(n_sims=2000)
    cps = {s: checkpoints(s) for s in SEASONS}

    # ---- stage 1: regression and roster_k on expected-win error at preseason
    print("stage 1: preseason win MAE (expected wins, no noise)")
    actual = {s: outcomes(s).set_index("TEAM_ID")["WINS"] for s in SEASONS}
    grid1 = []
    for reg, rk in product([0.15, 0.25, 0.33, 0.4, 0.5], [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]):
        cfg = {**base, "regression": reg, "roster_k": rk}
        err = [(expected_wins(s, cps[s]["preseason"], cfg).set_index("TEAM_ID")["EXP_W"] - actual[s]).abs().mean()
               for s in SEASONS]
        grid1.append({"regression": reg, "roster_k": rk, "win_mae": float(np.mean(err))})
    g1 = pd.DataFrame(grid1).sort_values("win_mae")
    print(g1.head(6).round(3).to_string(index=False))
    naive = np.mean([(41 - actual[s]).abs().mean() for s in SEASONS])
    print(f"  (everyone-41-wins baseline: {naive:.2f})")
    best = {**base, **g1.iloc[0][["regression", "roster_k"]].to_dict()}

    # ---- stage 2: preseason noise
    print("\nstage 2: preseason noise (simulated)")
    grid2 = []
    for nz in [0, 30, 45, 60, 75, 90, 110]:
        cfg = {**best, "noise_preseason": float(nz)}
        grid2.append({"noise_preseason": nz, **mean_scores([simulate_scores(s, cps[s]["preseason"], cfg) for s in SEASONS])})
    g2 = pd.DataFrame(grid2)
    print(g2[["noise_preseason", "win_cover_80", "playoffs_ll", "finals_ll", "title_ll", "champ_p"]].round(3).to_string(index=False))
    # pick: playoff log loss (primary, per the spec) with 10-90% coverage close to 80%
    g2["score"] = g2["playoffs_ll"] + 0.5 * (g2["win_cover_80"] - 0.8).abs()
    best["noise_preseason"] = float(g2.sort_values("score").iloc[0]["noise_preseason"])

    # ---- stage 3: how fast noise shrinks in-season
    print("\nstage 3: in-season noise decay (~20 and ~50 games in)")
    grid3 = []
    for half in [5, 10, 20, 40, 80]:
        cfg = {**best, "noise_half_games": float(half)}
        rows = [simulate_scores(s, cps[s][c], cfg) for s in SEASONS for c in ("g20", "g50")]
        grid3.append({"noise_half_games": half, **mean_scores(rows)})
    g3 = pd.DataFrame(grid3)
    print(g3[["noise_half_games", "win_cover_80", "playoffs_ll", "finals_ll", "title_ll"]].round(3).to_string(index=False))
    g3["score"] = g3["playoffs_ll"] + 0.5 * (g3["win_cover_80"] - 0.8).abs()
    best["noise_half_games"] = float(g3.sort_values("score").iloc[0]["noise_half_games"])

    # ---- final report with the chosen config, full 10k sims
    final = {**best, "n_sims": 10000}
    print(f"\nchosen: regression {final['regression']}, roster_k {final['roster_k']}, "
          f"noise_preseason {final['noise_preseason']}, noise_half_games {final['noise_half_games']}")
    report, per_season = {}, []
    calib = None
    for c in ("preseason", "g20", "g50"):
        rows = [simulate_scores(s, cps[s][c], final) for s in SEASONS]
        report[c] = mean_scores(rows)
        if c == "preseason":
            per_season = [{"season": s, "favorite": r["top_pick"], "champion": r["champ"],
                           "champion_odds": round(r["champ_p"], 3), "win_mae": round(r["win_mae"], 2)}
                          for s, r in zip(SEASONS, rows)]
            # playoff calibration buckets, pooled over seasons
            pool = pd.concat([r["rows"] for r in rows])
            pool["bucket"] = pd.cut(pool["P_PLAYOFFS"], [0, .1, .3, .5, .7, .9, 1], include_lowest=True)
            calib = pool.groupby("bucket", observed=True).agg(teams=("PLAYOFFS", "size"),
                                                              predicted=("P_PLAYOFFS", "mean"), actual=("PLAYOFFS", "mean"))
    print(pd.DataFrame(report).T.round(3).to_string())
    print("\npreseason playoff calibration (pooled):")
    print(calib.round(3).to_string())
    print("\npreseason title favorite vs champion:")
    print(pd.DataFrame(per_season).to_string(index=False))
    print(f"\n(uniform 1/30 title log loss would be {np.log(30):.2f}; everyone-41 win MAE {naive:.2f})")

    OUT.write_text(json.dumps({
        "seasons": SEASONS, "config": {k: final[k] for k in ("regression", "roster_k", "noise_preseason", "noise_half_games", "home_elo")},
        "by_checkpoint": report, "naive_win_mae": round(float(naive), 2),
        "playoff_calibration": [{"bucket": str(b), **v} for b, v in calib.round(3).to_dict("index").items()],
        "per_season": per_season}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
