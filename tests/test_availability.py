"""Synthetic games with known answers for features.availability.compute.  Run: pytest"""
import pandas as pd

from features.availability import compute

BOX = dict(FGM=10, FGA=20, FTA=5, FTM=4, OREB=1, DREB=5, STL=1, AST=5, BLK=0, PF=2, TOV=2, MIN=30)


def rows(day, team, players, season="22024", gid=None):
    return [dict(BOX, GAME_ID=gid or f"{team}{day}", GAME_DATE=f"2024-11-{day:02d}", TEAM_ID=team,
                 SEASON_ID=season, PLAYER_ID=p, PLAYER_NAME=f"p{p}", PTS=30 if p == 1 else 10)
            for p in players]


def missing(av, team, day):
    r = av[av["GAME_ID"] == f"{team}{day}"]
    return set() if r.empty or not r.iloc[0]["MISSING_NAMES"] else {
        n.split(" (")[0] for n in r.iloc[0]["MISSING_NAMES"].split(", ")}


def test_star_sits_and_no_leakage():
    data = rows(1, "A", [1, 2, 3]) + rows(2, "A", [1, 2, 3]) + rows(3, "A", [2, 3])
    av = compute(pd.DataFrame(data))
    assert missing(av, "A", 1) == set()          # opener: no history, nobody can be missing
    assert missing(av, "A", 3) == {"p1"}         # star sat out game 3
    star_gmsc = 30 + 4 - 14 - 0.4 + 0.7 + 1.5 + 1 + 3.5 - 0.8 - 2
    assert abs(av.loc[av["GAME_ID"] == "A3", "MISSING_TOP"].iloc[0] - star_gmsc) < 1e-9


def test_traded_player_not_missing_for_old_team():
    data = (rows(1, "A", [1, 2]) + rows(2, "B", [1, 9])    # player 1 traded to B
            + rows(3, "A", [2]))
    assert missing(compute(pd.DataFrame(data)), "A", 3) == set()


def test_new_season_resets_roster():
    data = rows(1, "A", [1, 2], season="22023") + rows(2, "A", [2], season="22024")
    assert missing(compute(pd.DataFrame(data)), "A", 2) == set()


def test_long_absence_drops_off():
    data = rows(1, "A", [1, 2]) + [r for d in range(2, 14) for r in rows(d, "A", [2])]
    av = compute(pd.DataFrame(data))
    assert missing(av, "A", 2) == {"p1"}
    assert missing(av, "A", 13) == set()          # out more than 10 team games: off the roster
