"""Monte Carlo layer: turn each team's expected points + spread into win probability and score ranges.

Scores are Normal, not Poisson (NBA scoring is high-volume and overdispersed). Home and away are drawn
jointly with correlation rho: fast/slow games and overtime push both scores the same way, and independent
draws would make the margin too wide and the win probability too close to 50%.

Validate on the test seasons:  python simulate.py
"""
import numpy as np

PCTS = (5, 25, 50, 75, 95)


def simulate(mu_home, sigma_home, mu_away, sigma_away, n=10_000, rho=0.0, seed=None) -> dict:
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal((2, n))
    home = mu_home + sigma_home * z1
    away = mu_away + sigma_away * (rho * z1 + np.sqrt(1 - rho**2) * z2)
    margin, total = home - away, home + away
    pct = lambda x: dict(zip(PCTS, np.percentile(x, PCTS).round(1).tolist()))  # noqa: E731
    return {
        "home_win_prob": float((margin > 0).mean()),  # continuous draws: ties have probability 0
        "margin_mean": float(margin.mean()),
        "margin_pct": pct(margin),
        "total_mean": float(total.mean()),
        "total_pct": pct(total),
        "home_pts_pct": pct(home),
        "away_pts_pct": pct(away),
    }


if __name__ == "__main__":
    import joblib
    import pandas as pd

    from models.baselines import score, split
    from models.train import FULL, POINTS

    art = joblib.load("models/model.joblib")
    te = split(pd.read_parquet("data/processed/games.parquet").dropna(subset=FULL + POINTS), "test")
    mu_h = art["points"]["H"]["model"].predict(te[POINTS])
    mu_a = art["points"]["A"]["model"].predict(te[POINTS])
    s_h, s_a, rho = art["points"]["H"]["sigma"], art["points"]["A"]["sigma"], art["rho"]
    y = te["HOME_WIN"].to_numpy()

    print(f"test {len(te)} games, sigma_home {s_h:.2f}, sigma_away {s_a:.2f}, rho {rho:.3f}\n")
    rows = {}
    for label, r in (("sim, rho", rho), ("sim, independent", 0.0)):
        sims = [simulate(h, s_h, a, s_a, rho=r, seed=i) for i, (h, a) in enumerate(zip(mu_h, mu_a))]
        p = np.array([s["home_win_prob"] for s in sims])
        lo = np.array([s["margin_pct"][5] for s in sims]); hi = np.array([s["margin_pct"][95] for s in sims])
        tlo = np.array([s["total_pct"][5] for s in sims]); thi = np.array([s["total_pct"][95] for s in sims])
        total = te["PTS_H"] + te["PTS_A"]
        rows[label] = {**score(y, p),
                       "margin_in_90%": np.mean((te["MARGIN"] >= lo) & (te["MARGIN"] <= hi)),
                       "total_in_90%": np.mean((total >= tlo) & (total <= thi))}
    p_lr = art["model"].predict_proba(te[art["cols"]])[:, 1]
    rows["logistic model"] = score(y, p_lr)
    print(pd.DataFrame(rows).T.round(4).to_string())
    print("\n(margin_in_90% / total_in_90% should be ~0.90 if the spreads are right)")

    g = te.iloc[-1]
    ex = simulate(mu_h[-1], s_h, mu_a[-1], s_a, rho=rho, seed=0)
    print(f"\nexample, {g['GAME_DATE']:%Y-%m-%d} {g['TEAM_ABBREVIATION_A']} @ {g['TEAM_ABBREVIATION_H']} "
          f"(actual {g['PTS_A']}-{g['PTS_H']}):")
    for k, v in ex.items():
        print(f"  {k:14s} {v if isinstance(v, dict) else round(v, 3)}")
