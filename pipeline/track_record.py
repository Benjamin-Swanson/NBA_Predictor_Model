"""Live track record: score saved pre-game predictions (data/history/) against final results.

Run from the project root:  python -m pipeline.track_record      (also runs inside predict_today)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
HISTORY = DATA / "history"
BUCKETS = [0.5, 0.6, 0.7, 0.8, 1.0]


def save_history(preds: dict) -> None:
    """Write today's predictions, but never overwrite a game that had already tipped off when it was saved."""
    HISTORY.mkdir(exist_ok=True)
    path = HISTORY / f"predictions_{preds['date']}.json"
    now = pd.Timestamp(preds["generated_at"])
    old = {g["game_id"]: g for g in json.loads(path.read_text())["games"]} if path.exists() else {}
    games = [old[g["game_id"]] if g["game_id"] in old and pd.Timestamp(g["tip_utc"]) <= now else
             {**g, "predicted_at": preds["generated_at"]} for g in preds["games"]]
    path.write_text(json.dumps({**preds, "games": games}, indent=1))


def score(results: pd.DataFrame) -> dict:
    """results: one row per finished team-game from the team game log (GAME_ID, TEAM_ABBREVIATION, PTS)."""
    pts = results.set_index(["GAME_ID", "TEAM_ABBREVIATION"])["PTS"].to_dict()
    rows = []
    for f in sorted(HISTORY.glob("predictions_*.json")):
        for g in json.loads(f.read_text())["games"]:
            gid = g["game_id"]
            if (gid, g["home"]) not in pts:
                continue  # not played yet
            ph, pa = int(pts[(gid, g["home"])]), int(pts[(gid, g["away"])])
            p = g["home_win_prob"]
            won = int(ph > pa)
            rows.append({"date": f.stem.split("_")[1], "game_id": gid, "away": g["away"], "home": g["home"],
                         "home_win_prob": p, "home_won": won, "pick": g["home"] if p >= 0.5 else g["away"],
                         "confidence": max(p, 1 - p), "correct": int((p >= 0.5) == won),
                         "proj": f"{g['away_pts']:.0f}-{g['home_pts']:.0f}", "final": f"{pa}-{ph}",
                         "margin_err": (ph - pa) - (g["home_pts"] - g["away_pts"]),
                         "in_band": int(g["margin_pct"]["5"] <= ph - pa <= g["margin_pct"]["95"]),
                         "log_loss": -np.log(np.clip(p if won else 1 - p, 1e-6, 1))})
    if not rows:
        return {"n": 0, "games": [], "buckets": []}
    df = pd.DataFrame(rows).sort_values(["date", "game_id"], ascending=False)
    df["bucket"] = pd.cut(df["confidence"], BUCKETS, include_lowest=True,
                          labels=[f"{int(a * 100)}-{int(b * 100)}%" for a, b in zip(BUCKETS, BUCKETS[1:])])
    buckets = (df.groupby("bucket", observed=True)
               .agg(games=("correct", "size"), predicted=("confidence", "mean"), actual=("correct", "mean"))
               .reset_index().round(3))
    buckets["bucket"] = buckets["bucket"].astype(str)
    return {
        "n": len(df), "accuracy": round(df["correct"].mean(), 4), "log_loss": round(df["log_loss"].mean(), 4),
        "brier": round(float(np.mean((df["home_win_prob"] - df["home_won"]) ** 2)), 4),
        "margin_mae": round(df["margin_err"].abs().mean(), 2), "band_coverage": round(df["in_band"].mean(), 4),
        "buckets": buckets.to_dict("records"),
        "games": df.drop(columns=["bucket", "log_loss"]).round(3).to_dict("records"),
    }


def update() -> dict:
    from pipeline.fetch_data import CURRENT_SEASON, game_log
    rec = {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), **score(game_log(CURRENT_SEASON, "T"))}
    (DATA / "track_record.json").write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    r = update()
    print(f"{r['n']} graded games" + (f": {r['accuracy']:.1%} correct, log loss {r['log_loss']:.3f}" if r["n"] else ""))
