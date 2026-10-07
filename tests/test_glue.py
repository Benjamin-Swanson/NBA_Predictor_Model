"""Known-answer checks for features.glue_index.  Run: pytest"""
import pandas as pd

from features.glue_index import onoff_to_date, team_glue

D = pd.Timestamp


def test_onoff_math():
    # One 48-min game, team +10. Player: 24 min at +8  -> on +16/48, off +2 in 24 -> +4/48. On/off = +12.
    pl = pd.DataFrame([dict(GAME_ID="g1", TEAM_ID=1, PLAYER_ID=7, PLUS_MINUS=8, MIN=24,
                            GAME_DATE=D("2024-11-01"), SEASON=2024)])
    tg = pd.DataFrame([dict(GAME_ID="g1", TEAM_ID=1, PLUS_MINUS=10, MIN=240)])
    snaps = pd.DataFrame([dict(PLAYER_ID=7, SEASON=2024, DATE_TO=D("2024-11-03"))])
    assert abs(onoff_to_date(pl, tg, snaps)["ONOFF"].iloc[0] - 12.0) < 1e-9


def test_team_glue_uses_only_earlier_snapshot_and_weights_by_mpg():
    glue = pd.DataFrame([
        dict(PLAYER_ID=1, SEASON=2024, DATE_TO=D("2024-11-03"), PLAYER_NAME="a", MPG=30.0, GLUE=1.0),
        dict(PLAYER_ID=2, SEASON=2024, DATE_TO=D("2024-11-03"), PLAYER_NAME="b", MPG=10.0, GLUE=-1.0),
        # same-day snapshot must NOT be used for a game on 11-10
        dict(PLAYER_ID=1, SEASON=2024, DATE_TO=D("2024-11-10"), PLAYER_NAME="a", MPG=30.0, GLUE=99.0),
    ])
    pl = pd.DataFrame([dict(GAME_ID="g", TEAM_ID=1, PLAYER_ID=p, GAME_DATE=D("2024-11-10"), SEASON=2024)
                       for p in (1, 2)])
    got = team_glue(pl, glue)["GLUE"].iloc[0]
    assert abs(got - (30 * 1.0 + 10 * -1.0) / 40) < 1e-9
