"""Train win-probability and points models, evaluate on date-based splits, save the best.

Run from the project root:  python -m models.train   (after python -m features.build)
Train 2016-17..2022-23 (glue starts then; Elo still warms up from 2015), choose models on val (2023-24), report test (2024-25 + 2025-26) once.
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features.build import OUT as GAMES, ROLL_COLS, WINDOW
from models.baselines import elo_prob, score, split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
R = [f"R{WINDOW}_{c}" for c in ROLL_COLS]

# Feature groups mirror the prior-weight table so importance can be compared to it.
GROUPS = {
    "efficiency": [f"D_R{WINDOW}_{c}" for c in ("ORTG", "DRTG", "NET")],
    "recent_form": [f"D_R{WINDOW}_MARGIN"],
    "elo": ["D_ELO"],
    "four_factors": [f"D_R{WINDOW}_{c}" for c in ("EFG", "TOV_PCT", "ORB_PCT", "FT_RATE",
                                                  "OPP_EFG", "OPP_TOV_PCT", "DRB_PCT", "OPP_FT_RATE")],
    "rest": ["REST_H", "REST_A", "B2B_H", "B2B_A", "G3IN4_H", "G3IN4_A"],
    "pace": [f"D_R{WINDOW}_PACE"],
    "neutral_site": ["NEUTRAL"],
    "availability": ["MISSING_GMSC_H", "MISSING_GMSC_A", "MISSING_TOP_H", "MISSING_TOP_A"],
    "glue": ["D_GLUE"],  # exists from 2016-17, so training effectively starts there
}
FULL = [c for cols in GROUPS.values() for c in cols]
# Handoff's suggested first pass, without availability
MINIMAL = GROUPS["efficiency"] + ["REST_H", "REST_A", "NEUTRAL", "D_ELO"]
# Points models need each team's levels, not just differences
POINTS = [f"{c}_{s}" for c in R + ["ELO", "REST", "B2B", "MISSING_GMSC", "MISSING_TOP", "GLUE"]
          for s in ("H", "A")] + ["NEUTRAL"]


def logreg(cols):
    return cols, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def fit_lgbm(cols, tr, va):
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.01, num_leaves=15, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(tr[cols], tr["HOME_WIN"], eval_X=(va[cols],), eval_y=(va["HOME_WIN"],), eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


def group_importance(model, cols, df, n_repeats=10, seed=0):
    """Increase in log loss when a whole feature group is shuffled together (handles correlated cols within a group)."""
    rng = np.random.default_rng(seed)
    y = df["HOME_WIN"].to_numpy()
    base = score(y, model.predict_proba(df[cols])[:, 1])["log_loss"]
    out = {}
    for name, group in GROUPS.items():
        group = [c for c in group if c in cols]
        if not group:
            continue
        losses = []
        for _ in range(n_repeats):
            shuffled = df[cols].copy()
            shuffled[group] = df[group].to_numpy()[rng.permutation(len(df))]
            losses.append(score(y, model.predict_proba(shuffled)[:, 1])["log_loss"])
        out[name] = np.mean(losses) - base
    return pd.Series(out).sort_values(ascending=False)


def ablation(group: str, tr, va, te) -> None:
    """Logistic regression with vs without one feature group, same training rows."""
    extra = GROUPS[group]
    print(f"\n{group} ablation:")
    for name, cols in (("without", [c for c in FULL if c not in extra]), ("with", FULL)):
        cols, m = logreg(cols)
        m.fit(tr[cols], tr["HOME_WIN"])
        r = {s: score(df["HOME_WIN"].to_numpy(), m.predict_proba(df[cols])[:, 1]) for s, df in (("val", va), ("test", te))}
        print(f"  {name:8s} val log loss {r['val']['log_loss']:.4f} acc {r['val']['accuracy']:.4f} | "
              f"test log loss {r['test']['log_loss']:.4f} acc {r['test']['accuracy']:.4f}")


def main():
    games = pd.read_parquet(GAMES).dropna(subset=FULL + POINTS)  # drops 2015-16 (no glue) + early 2016-17
    tr, va, te = (split(games, s) for s in ("train", "val", "test"))
    print(f"train {len(tr)}, val {len(va)}, test {len(te)} games\n")

    models = {}
    for name, cols in (("logreg_minimal", MINIMAL), ("logreg_full", FULL)):
        cols, m = logreg(cols)
        models[name] = (cols, m.fit(tr[cols], tr["HOME_WIN"]))
    models["lgbm_full"] = (FULL, fit_lgbm(FULL, tr, va))
    print(f"lgbm early-stopped at {models['lgbm_full'][1].best_iteration_} trees")

    # Points: one ridge model per team score; sigma from val residuals feeds the Monte Carlo layer
    pts = {}
    for side in ("H", "A"):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(tr[POINTS], tr[f"PTS_{side}"])
        pts[side] = {"model": m, "sigma": float(np.std(va[f"PTS_{side}"] - m.predict(va[POINTS])))}
    margin_sigma = float(np.std(va["MARGIN"] - (pts["H"]["model"].predict(va[POINTS])
                                                - pts["A"]["model"].predict(va[POINTS]))))
    # Home and away score errors move together (pace, OT): simulate.py draws them jointly with this
    rho = float(np.corrcoef(va["PTS_H"] - pts["H"]["model"].predict(va[POINTS]),
                            va["PTS_A"] - pts["A"]["model"].predict(va[POINTS]))[0, 1])

    def predict(name, df):
        if name == "elo":
            return elo_prob(df)
        if name == "points_model":
            mu = pts["H"]["model"].predict(df[POINTS]) - pts["A"]["model"].predict(df[POINTS])
            return norm.cdf(mu / margin_sigma)
        cols, m = models[name]
        return m.predict_proba(df[cols])[:, 1]

    names = ["elo", *models, "points_model"]
    rows = []
    for split_name, df in (("val", va), ("test", te)):
        y = df["HOME_WIN"].to_numpy()
        rows += [{"split": split_name, "model": n, **score(y, predict(n, df))} for n in names]
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))

    for side in ("H", "A"):
        err = te[f"PTS_{side}"] - pts[side]["model"].predict(te[POINTS])
        print(f"\npoints {side}: test MAE {err.abs().mean():.2f}, val sigma {pts[side]['sigma']:.2f}")
    mu = pts["H"]["model"].predict(te[POINTS]) - pts["A"]["model"].predict(te[POINTS])
    print(f"margin: test MAE {np.abs(te['MARGIN'] - mu).mean():.2f}, val sigma {margin_sigma:.2f}, residual rho {rho:.3f}")

    for group in ("availability", "glue"):
        ablation(group, tr, va, te)

    # Model choice uses val only
    best = res[(res["split"] == "val") & (res["model"].isin(models))].sort_values("log_loss").iloc[0]["model"]
    cols, m = models[best]
    print(f"\nbest on val: {best}\n\ngroup importance (val log-loss increase when shuffled), {best}:")
    imp = group_importance(m, cols, va)
    print(pd.DataFrame({"logloss_increase": imp.round(5),
                        "share": (imp.clip(lower=0) / imp.clip(lower=0).sum() * 100).round(1)}).to_string())

    fig, ax = plt.subplots(figsize=(6, 6))
    y = te["HOME_WIN"]
    for n in ("elo", "logreg_full", "lgbm_full"):
        CalibrationDisplay.from_predictions(y, predict(n, te), n_bins=10, name=n, ax=ax)
    ax.set_title("Calibration, test 2024-25 + 2025-26")
    fig.savefig(HERE / "calibration.png", dpi=120, bbox_inches="tight")

    # ponytail: saved models are fit on train only; refit on all seasons before live use (step 8).
    joblib.dump({"name": best, "cols": cols, "model": m, "points": pts, "points_cols": POINTS,
                 "margin_sigma": margin_sigma, "rho": rho, "groups": GROUPS,
                 "results": res, "importance": imp, "train_seasons": (int(tr["SEASON"].min()), int(tr["SEASON"].max()))},
                HERE / "model.joblib")
    print(f"\nsaved {HERE / 'model.joblib'} and calibration.png")


if __name__ == "__main__":
    main()
