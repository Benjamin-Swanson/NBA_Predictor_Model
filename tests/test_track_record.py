"""Track record grading and the no-overwrite-after-tip rule, on fake files.  Run: pytest"""
import json

import pandas as pd

from pipeline import track_record as tr


def game(gid, home, away, p, tip, hp=112, ap=108):
    return {"game_id": gid, "home": home, "away": away, "home_win_prob": p, "tip_utc": tip,
            "home_pts": hp, "away_pts": ap, "margin_pct": {"5": -20, "95": 28}}


def test_history_freezes_after_tip_and_grades(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "HISTORY", tmp_path)
    tr.save_history({"date": "2026-10-20", "generated_at": "2026-10-20T15:00:00+00:00",
                     "games": [game("g1", "DET", "BOS", 0.63, "2026-10-20T19:00:00Z"),
                               game("g2", "NYK", "PHI", 0.72, "2026-10-20T23:00:00Z")]})
    # 20:00 UTC rerun: g1 already tipped (keep 0.63), g2 not yet (take the update)
    tr.save_history({"date": "2026-10-20", "generated_at": "2026-10-20T20:00:00+00:00",
                     "games": [game("g1", "DET", "BOS", 0.99, "2026-10-20T19:00:00Z"),
                               game("g2", "NYK", "PHI", 0.40, "2026-10-20T23:00:00Z")]})
    saved = {g["game_id"]: g["home_win_prob"] for g in json.loads((tmp_path / "predictions_2026-10-20.json").read_text())["games"]}
    assert saved == {"g1": 0.63, "g2": 0.40}

    results = pd.DataFrame([  # DET beat BOS; PHI beat NYK
        {"GAME_ID": "g1", "TEAM_ABBREVIATION": "DET", "MATCHUP": "DET vs. BOS", "PTS": 110},
        {"GAME_ID": "g1", "TEAM_ABBREVIATION": "BOS", "MATCHUP": "BOS @ DET", "PTS": 100},
        {"GAME_ID": "g2", "TEAM_ABBREVIATION": "NYK", "MATCHUP": "NYK vs. PHI", "PTS": 99},
        {"GAME_ID": "g2", "TEAM_ABBREVIATION": "PHI", "MATCHUP": "PHI @ NYK", "PTS": 104}])
    r = tr.score(results)
    assert r["n"] == 2 and r["accuracy"] == 1.0          # DET at 63% right; PHI (NYK 40%) right
    assert abs(r["brier"] - ((0.63 - 1) ** 2 + 0.40 ** 2) / 2) < 1e-4
