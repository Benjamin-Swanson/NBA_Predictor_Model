"""Predict any matchup from data/team_state.json + models/model.joblib. No nba_api calls.

Features are built exactly as in training (features/build.py, availability.py, glue_index.py).
Try:  python predict.py BOS NYK        (away team first, like "BOS @ NYK")
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from simulate import simulate

ROOT = Path(__file__).resolve().parent
MIN_MPG, GLUE_TOP_N = 10.0, 10  # same as features/availability.py MIN_MPG, glue_index.py TOP_N


def load():
    return json.loads((ROOT / "data" / "team_state.json").read_text()), joblib.load(ROOT / "models" / "model.joblib")


def team_levels(team: dict, out_ids=(), rest: int = 2) -> dict:
    out_ids = set(out_ids)
    missing = [p["value"] for p in team["players"] if p["id"] in out_ids
               and p["value"] is not None and (p["mpg_recent"] or 0) >= MIN_MPG]
    avail = [p for p in team["players"] if p["id"] not in out_ids and p["glue"] is not None and p["mpg_season"]]
    top = sorted(avail, key=lambda p: -p["mpg_season"])[:GLUE_TOP_N]
    glue = sum(p["mpg_season"] * p["glue"] for p in top) / sum(p["mpg_season"] for p in top) if top else 0.0
    lv = {f"R10_{k}": v for k, v in team["recent"].items()}
    lv.update(ELO=team["elo"], REST=min(rest, 7), B2B=int(rest == 1), G3IN4=0,
              MISSING_GMSC=sum(missing), MISSING_TOP=max(missing, default=0.0), MISSING_N=len(missing), GLUE=glue)
    return lv


def feature_row(state, home, away, out_home=(), out_away=(), rest_home=2, rest_away=2, neutral=False) -> pd.DataFrame:
    h = team_levels(state["teams"][home], out_home, rest_home)
    a = team_levels(state["teams"][away], out_away, rest_away)
    row = {"NEUTRAL": int(neutral)}
    for k in h:
        row[f"{k}_H"], row[f"{k}_A"], row[f"D_{k}"] = h[k], a[k], h[k] - a[k]
    return pd.DataFrame([row])


def why(art, x: pd.DataFrame) -> pd.Series:
    """Each feature group's push on the log-odds of a home win (+ favors home), from the logistic model."""
    scaler, lr = art["model"][0], art["model"][-1]
    contrib = pd.Series(lr.coef_[0] * (x[art["cols"]].to_numpy()[0] - scaler.mean_) / scaler.scale_, index=art["cols"])
    return pd.Series({g: contrib[[c for c in cols if c in contrib]].sum() for g, cols in art["groups"].items()})


def predict_matchup(state, art, home, away, out_home=(), out_away=(), rest_home=2, rest_away=2,
                    neutral=False, n=10_000, keep_draws=False) -> dict:
    x = feature_row(state, home, away, out_home, out_away, rest_home, rest_away, neutral)
    mu_h = float(art["points"]["H"]["model"].predict(x[art["points_cols"]])[0])
    mu_a = float(art["points"]["A"]["model"].predict(x[art["points_cols"]])[0])
    sim = simulate(mu_h, art["points"]["H"]["sigma"], mu_a, art["points"]["A"]["sigma"], n=n,
                   rho=art["rho"], seed=0, keep_draws=keep_draws)
    return {"home": home, "away": away, "mu_home": mu_h, "mu_away": mu_a,
            "logistic_home_win_prob": float(art["model"].predict_proba(x[art["cols"]])[0, 1]),
            "why": why(art, x), "features": x, **sim}


if __name__ == "__main__":
    away, home = (sys.argv[1:3] + ["BOS", "NYK"])[:2] if len(sys.argv) > 1 else ("BOS", "NYK")
    state, art = load()
    r = predict_matchup(state, art, home.upper(), away.upper())
    print(f"{r['away']} @ {r['home']}: {r['home']} {r['home_win_prob']:.1%} (logistic {r['logistic_home_win_prob']:.1%})")
    print(f"projected {r['away']} {r['mu_away']:.1f} - {r['home']} {r['mu_home']:.1f}, total 90% {r['total_pct'][5]}-{r['total_pct'][95]}")
    print("why (log-odds, + favors home):", r["why"].round(3).to_dict())
    # Two models, same test log loss; on test games they differ by 2.6 pts median, 13 max. Bigger = something broke.
    assert abs(r["home_win_prob"] - r["logistic_home_win_prob"]) < 0.15, "sim and logistic disagree"
