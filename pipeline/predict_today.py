"""Daily pipeline: refresh data → team state → injuries → predict the next game day → data/predictions.json.

Run from the project root:
    python -m pipeline.predict_today                 # next game day on/after today (US Eastern)
    python -m pipeline.predict_today --date 2026-10-22
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from features import injuries
from pipeline import team_state
from pipeline.fetch_data import schedule
from predict import load, predict_matchup

DATA = Path(__file__).resolve().parent.parent / "data"


def rest_info(sched: pd.DataFrame, team: str, date: pd.Timestamp) -> tuple[int, int]:
    """(rest days capped at 7, 3-games-in-4-nights flag) from the schedule, same as training."""
    dates = sorted(sched.loc[(sched["HOME"] == team) | (sched["AWAY"] == team), "GAME_DATE"])
    before = [d for d in dates if d < date]
    rest = min((date - before[-1]).days, 7) if before else 7
    g3in4 = int(len(before) >= 2 and (date - before[-2]).days <= 3)
    return rest, g3in4


def run(date: str | None = None) -> dict:
    state = team_state.build()
    (DATA / "team_state.json").write_text(json.dumps(state, indent=1))
    _, art = load()

    sched = schedule()
    today_et = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    target = pd.Timestamp(date) if date else sched.loc[sched["GAME_DATE"] >= today_et, "GAME_DATE"].min()
    games = sched[sched["GAME_DATE"] == target].sort_values("TIP_UTC")

    report = injuries.fetch()
    inj, unmatched = injuries.attach(state, report)

    out = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "date": f"{target:%Y-%m-%d}", "ratings_through": state["as_of"],
           "unmatched_injuries": unmatched, "games": []}
    for g in games.itertuples():
        (rest_h, g34_h), (rest_a, g34_a) = rest_info(sched, g.HOME, target), rest_info(sched, g.AWAY, target)
        out_h = [p["id"] for p in inj[g.HOME] if p["out"]]
        out_a = [p["id"] for p in inj[g.AWAY] if p["out"]]
        r = predict_matchup(state, art, g.HOME, g.AWAY, out_h, out_a, rest_h, rest_a, bool(g.NEUTRAL),
                            g3in4_home=g34_h, g3in4_away=g34_a)
        out["games"].append({
            "game_id": g.GAME_ID, "tip_utc": g.TIP_UTC, "tip_et": g.STATUS, "arena": g.ARENA,
            "home": g.HOME, "away": g.AWAY, "neutral": bool(g.NEUTRAL),
            "home_name": state["teams"][g.HOME]["name"], "away_name": state["teams"][g.AWAY]["name"],
            "home_win_prob": round(r["home_win_prob"], 4),
            "home_pts": round(r["mu_home"], 1), "away_pts": round(r["mu_away"], 1),
            "margin_pct": r["margin_pct"], "total_mean": round(r["total_mean"], 1), "total_pct": r["total_pct"],
            "rest_home": rest_h, "rest_away": rest_a,
            "injuries_home": [{k: p[k] for k in ("name", "status", "note", "out")} for p in inj[g.HOME]],
            "injuries_away": [{k: p[k] for k in ("name", "status", "note", "out")} for p in inj[g.AWAY]],
        })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (US Eastern game date); default: next game day")
    preds = run(ap.parse_args().date)
    text = json.dumps(preds, indent=1)
    (DATA / "predictions.json").write_text(text)
    # Keep each day's last pre-tip version so live accuracy can be scored later
    (DATA / "history").mkdir(exist_ok=True)
    (DATA / "history" / f"predictions_{preds['date']}.json").write_text(text)
    print(f"{len(preds['games'])} games on {preds['date']} (ratings through {preds['ratings_through']}):")
    for g in preds["games"]:
        fav, p = (g["home"], g["home_win_prob"]) if g["home_win_prob"] >= 0.5 else (g["away"], 1 - g["home_win_prob"])
        outs = [i["name"] for i in g["injuries_home"] + g["injuries_away"] if i["out"]]
        print(f"  {g['away']} @ {g['home']}: {fav} {p:.0%}, {g['away_pts']:.0f}-{g['home_pts']:.0f}"
              + (f"  (out: {', '.join(outs)})" if outs else ""))
    if preds["unmatched_injuries"]:
        print("unmatched injury names:", preds["unmatched_injuries"])
