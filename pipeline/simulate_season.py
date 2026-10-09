"""Season outlook job: ratings + standings + remaining schedule → 10,000 simulated seasons →
data/season_outlook.json and one row per team appended to data/season_outlook_history.csv.

Skips when no new games have finished since the last run (use --force to run anyway).
Run from the project root:  python -m pipeline.simulate_season [--force]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.static import teams as static_teams

from pipeline.fetch_data import CURRENT_SEASON
from season import inputs
from season.ratings import injury_adjust, load_config, team_ratings
from season.simulate import check, run, summarize

DATA = Path(__file__).resolve().parent.parent / "data"
OUT, HIST = DATA / "season_outlook.json", DATA / "season_outlook_history.csv"
LONG_TERM_DAYS = 14  # injuries expected to last longer than this lower a team's season rating


def long_term_missing(today: pd.Timestamp) -> pd.Series:
    """Per TEAM_ID: summed Game Score value of rotation players listed Out with a return date > 2 weeks away."""
    from features import injuries
    state = json.loads((DATA / "team_state.json").read_text())
    rep = injuries.fetch()
    found, _ = injuries.attach(state, rep)
    report = rep.drop_duplicates("PLAYER").set_index("PLAYER")["RETURN_DATE"]
    missing = {}
    for abbr, plist in found.items():
        players = {p["id"]: p for p in state["teams"][abbr]["players"]}
        total = 0.0
        for inj in plist:
            ret = pd.to_datetime(report.get(inj["name"]), errors="coerce")
            p = players[inj["id"]]
            if inj["out"] and pd.notna(ret) and (ret - today).days > LONG_TERM_DAYS \
                    and p["value"] is not None and (p["mpg_recent"] or 0) >= 10:
                total += p["value"]
        missing[state["teams"][abbr]["id"]] = total
    return pd.Series(missing)


def main(force: bool = False) -> None:
    cfg = load_config()
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    tm, played, rem = inputs.as_of(CURRENT_SEASON, today)
    if OUT.exists() and not force:
        prev = json.loads(OUT.read_text())
        if prev["games_played"] == len(played) and prev["config"] == {k: cfg[k] for k in prev["config"]}:
            prev["last_checked"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
            OUT.write_text(json.dumps(prev, indent=1))  # page freshness = last check, not last recalculation
            print(f"no new games since {prev['last_updated']} ({len(played)} played): skipping")
            return

    ratings = team_ratings(CURRENT_SEASON, today, tm["TEAM_ID"], cfg)
    try:
        ratings = injury_adjust(ratings, long_term_missing(today), cfg)
    except Exception as e:  # injury feed down: run without it rather than skip the day
        print(f"injury adjustment skipped: {e}")
    t0 = time.perf_counter()
    out = run(tm, played, rem, ratings, cfg)
    df = summarize(out, tm, inputs.records(tm, played), ratings)
    check(df, cfg["n_sims"])
    secs = time.perf_counter() - t0

    abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    idx = {t: i for i, t in enumerate(out["ids"])}
    teams = []
    for r in df.itertuples():
        w = out["wins"][:, idx[r.TEAM_ID]]
        hist = np.bincount(w, minlength=83)[:83] / len(w)
        teams.append({
            "abbr": abbr[r.TEAM_ID], "name": r.NAME, "conf": r.CONF, "div": r.DIV, "record": f"{r.W}-{r.L}",
            "rating": r.RATING, "wins_mean": round(r.WINS_MEAN, 1), "wins_p10": int(r.WINS_P10),
            "wins_p50": int(r.WINS_P50), "wins_p90": int(r.WINS_P90),
            "p_playoffs": round(r.P_PLAYOFFS, 4), "p_playin": round(r.P_PLAYIN, 4), "p_lottery": round(r.P_LOTTERY, 4),
            "p_win_r1": round(r.P_WIN_R1, 4), "p_conf_finals": round(r.P_CONF_FINALS, 4),
            "p_finals": round(r.P_FINALS, 4), "p_title": round(r.P_TITLE, 4),
            "p_seed": [round(getattr(r, f"P_SEED_{k}"), 4) for k in range(1, 16)],
            "wins_dist": [round(float(x), 5) for x in hist]})
    stamp = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    OUT.write_text(json.dumps({
        "last_updated": stamp, "season": CURRENT_SEASON, "as_of": f"{today:%Y-%m-%d}", "games_played": len(played),
        "games_remaining": len(rem), "config": {k: cfg[k] for k in ("n_sims", "home_elo", "regression", "roster_k",
                                                                       "roster_fade_games", "noise_preseason", "noise_half_games")},
        "teams": teams}, indent=1))
    hist_rows = pd.DataFrame({"date": f"{today:%Y-%m-%d}", "abbr": [t["abbr"] for t in teams],
                              "p_title": [t["p_title"] for t in teams], "p_playoffs": [t["p_playoffs"] for t in teams],
                              "wins_p50": [t["wins_p50"] for t in teams]})
    if HIST.exists():  # one row per team per day: a rerun replaces today's rows
        hist_rows = pd.concat([pd.read_csv(HIST).query("date != @hist_rows.date[0]"), hist_rows])
    hist_rows.to_csv(HIST, index=False)
    print(f"{cfg['n_sims']:,} seasons in {secs:.1f}s ({len(played)} games played) → {OUT.name}")
    for t in teams[:5]:
        print(f"  {t['abbr']}: title {t['p_title']:.1%}, wins {t['wins_p50']} ({t['wins_p10']}-{t['wins_p90']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
