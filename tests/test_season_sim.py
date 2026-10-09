"""Play-in and bracket rules on a made-up league with a known answer.  Run: pytest"""
import itertools

import numpy as np
import pandas as pd

from season.simulate import run, summarize

CFG = {"n_sims": 2000, "seed": 0, "home_elo": 60.0, "noise_preseason": 0.0, "noise_half_games": 20.0}


def league(strong_seed: int):
    """Two 15-team conferences; season over; in each conference team k beat every team after it (seed = k+1).
    The East team finishing at `strong_seed` is unbeatable."""
    tm = pd.DataFrame({"TEAM_ID": range(1, 31), "CONF": ["East"] * 15 + ["West"] * 15,
                       "DIV": "x", "NAME": [f"T{i}" for i in range(1, 31)]})
    played = pd.DataFrame([{"HOME_ID": c + i, "AWAY_ID": c + j, "NEUTRAL": False, "HOME_WIN": 1.0}
                           for c in (1, 16) for i, j in itertools.combinations(range(15), 2)])
    ratings = pd.DataFrame({"ELO": 1500.0, "GAMES": 82}, index=tm["TEAM_ID"])
    ratings.loc[strong_seed, "ELO"] = 4000.0
    return tm, played, ratings


def test_ten_seed_can_win_title_through_play_in():
    tm, played, ratings = league(strong_seed=10)
    remaining = pd.DataFrame(columns=["HOME_ID", "AWAY_ID", "NEUTRAL"])
    df = summarize(run(tm, played, remaining, ratings, CFG), tm, tm.assign(W=0, L=0), ratings).set_index("TEAM_ID")
    assert df.loc[10, "P_SEED_10"] == 1 and df.loc[10, "P_PLAYIN"] == 1
    assert df.loc[10, "P_TITLE"] == 1                       # won 9v10, won for the 8 seed, then 4 series
    assert df.loc[11, "P_PLAYOFFS"] == 0                    # 11th place never gets in
    assert np.isclose(df["P_TITLE"].sum(), 1) and np.isclose(df["P_PLAYOFFS"].sum(), 16)
    for k in range(1, 7):                                   # seeds 1-6 always make it
        assert df.loc[k, "P_PLAYOFFS"] == 1


def test_eleven_seed_is_out_even_if_unbeatable():
    tm, played, ratings = league(strong_seed=11)
    remaining = pd.DataFrame(columns=["HOME_ID", "AWAY_ID", "NEUTRAL"])
    df = summarize(run(tm, played, remaining, ratings, CFG), tm, tm.assign(W=0, L=0), ratings).set_index("TEAM_ID")
    assert df.loc[11, "P_PLAYOFFS"] == 0 and df.loc[11, "P_TITLE"] == 0
