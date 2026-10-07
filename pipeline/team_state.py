"""Snapshot of every team right now: rating, recent form, current roster with player values.
The app reads data/team_state.json and never calls nba_api itself.

Run from the project root:  python -m pipeline.team_state
"""
import json
from pathlib import Path

import pandas as pd

from features import availability, glue_index
from features.build import (ELO_CARRY, ELO_MEAN, ROLL_COLS, WINDOW, add_elo, add_pregame_features,
                            team_games)
from pipeline.fetch_data import CURRENT_SEASON, all_player_game_logs, all_team_game_logs, current_rosters

OUT = Path(__file__).resolve().parent.parent / "data" / "team_state.json"


def build() -> dict:
    tg = add_elo(add_pregame_features(team_games(all_team_game_logs()))).sort_values("GAME_DATE")
    last_season = int(tg["SEASON"].max())
    new_season = int(CURRENT_SEASON[:4]) > last_season  # no games played yet this season

    pl = all_player_game_logs()
    pl = pl.assign(GAME_DATE=pd.to_datetime(pl["GAME_DATE"]), GMSC=availability.game_score(pl)).sort_values("GAME_DATE")
    by_p = pl.groupby("PLAYER_ID")
    # Same definitions as features/availability.py: value = last-20 Game Score, minutes = last-10 average
    pv = pd.DataFrame({"value": by_p["GMSC"].apply(lambda s: s.tail(availability.VALUE_GAMES).mean()),
                       "mpg_recent": by_p["MIN"].apply(lambda s: s.tail(availability.ROSTER_GAMES).mean())})
    # ponytail: uses the blended end-of-season glue, training's week 1 uses the unblended final value; nearly equal by April.
    glue = glue_index.build()[0].sort_values("DATE_TO").groupby("PLAYER_ID")[["GLUE", "MPG"]].last()
    pv = pv.join(glue.rename(columns={"GLUE": "glue", "MPG": "mpg_season"}), how="outer")

    rosters = current_rosters()
    state = {"as_of": f"{tg['GAME_DATE'].max():%Y-%m-%d}", "season": CURRENT_SEASON, "teams": {}}
    for abbr, ros in rosters.groupby("TEAM_ABBREVIATION"):
        team_id = int(ros["TeamID"].iloc[0])
        games = tg[tg["TEAM_ID"] == team_id]
        elo = float(games["ELO_POST"].iloc[-1])
        if new_season:
            elo = ELO_CARRY * elo + (1 - ELO_CARRY) * ELO_MEAN
        players = []
        for r in ros.itertuples():
            v = pv.loc[r.PLAYER_ID] if r.PLAYER_ID in pv.index else None
            num = lambda k: None if v is None or pd.isna(v[k]) else round(float(v[k]), 3)  # noqa: E731
            players.append({"id": int(r.PLAYER_ID), "name": r.PLAYER, "pos": r.POSITION,
                            "value": num("value"), "mpg_recent": num("mpg_recent"),
                            "glue": num("glue"), "mpg_season": num("mpg_season")})
        players.sort(key=lambda p: -(p["value"] or -99))
        state["teams"][abbr] = {
            "id": team_id, "name": ros["TEAM_NAME"].iloc[0], "elo": round(elo, 1),
            "recent": {c: round(float(games[c].tail(WINDOW).mean()), 4) for c in ROLL_COLS},
            "players": players,
        }
    return state


if __name__ == "__main__":
    state = build()
    OUT.write_text(json.dumps(state, indent=1))
    t = state["teams"]
    print(f"{len(t)} teams, data through {state['as_of']} -> {OUT}")
    for abbr in sorted(t, key=lambda a: -t[a]["elo"])[:5]:
        top = ", ".join(p["name"] for p in t[abbr]["players"][:3])
        print(f"  {abbr} elo {t[abbr]['elo']:.0f}  net(L10) {t[abbr]['recent']['NET']:+.1f}  top: {top}")
