"""Baselines every real model must beat, scored on date-based splits.

Run from the project root:  python -m models.baselines   (after python -m features.build)
"""
import numpy as np
import pandas as pd

from features.build import ELO_HCA, OUT as GAMES

SPLITS = {"train": (2015, 2022), "val": (2023, 2023), "test": (2024, 2025)}  # SEASON = start year


def split(games: pd.DataFrame, name: str) -> pd.DataFrame:
    lo, hi = SPLITS[name]
    return games[games["SEASON"].between(lo, hi)]


def score(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "brier": np.mean((p - y) ** 2),
        "accuracy": np.mean((p > 0.5) == y),
        "n": len(y),
    }


def home_always(g: pd.DataFrame, home_rate: float) -> np.ndarray:
    # Accuracy ignores the probability (it predicts home every time); log loss/Brier use the train home win rate.
    return np.where(g["NEUTRAL"].eq(1), 0.5, home_rate)


def elo_prob(g: pd.DataFrame) -> np.ndarray:
    hca = np.where(g["NEUTRAL"].eq(1), 0.0, ELO_HCA)
    return 1 / (1 + 10 ** (-(g["ELO_H"] - g["ELO_A"] + hca) / 400))


if __name__ == "__main__":
    games = pd.read_parquet(GAMES)
    train = split(games, "train")
    home_rate = train.loc[train["NEUTRAL"].eq(0), "HOME_WIN"].mean()
    print(f"train home win rate: {home_rate:.3f}\n")
    rows = []
    for name in ("val", "test"):
        g = split(games, name)
        y = g["HOME_WIN"].to_numpy()
        # home-always accuracy: predict home every game (neutral "home" slot included)
        r = score(y, home_always(g, home_rate)); r["accuracy"] = y.mean()
        rows.append({"split": name, "model": "home always wins", **r})
        rows.append({"split": name, "model": "Elo", **score(y, elo_prob(g))})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nhome win rate by season:")
    print(games[games["NEUTRAL"].eq(0)].groupby("SEASON")["HOME_WIN"].mean().round(3).to_string())
