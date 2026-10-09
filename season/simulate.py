"""Vectorized season Monte Carlo: rest of the regular season → seeding → play-in → playoffs.

Every array has a leading axis of n_sims, so 10,000 seasons run as NumPy array operations.

Rules simulated (current NBA format, in place since 2020-21):
- Seeds 1-6 per conference qualify; 7-10 play in: 7v8 (winner = 7 seed), 9v10 (loser out),
  loser(7v8) v winner(9v10) for the 8 seed. Higher seed hosts each play-in game.
- Playoffs: 1v8, 4v5, 2v7, 3v6 bracket, best-of-7 with 2-2-1-1-1 home court (better seed hosts 1, 2, 5, 7).
  Finals home court goes to the better regular-season record.
- A series is decided by simulating all 7 games with home/away-specific probabilities; who wins 4 of 7
  is the same as who wins 4 first, so no early stopping is needed.

Tiebreakers (SIMPLIFIED, v1): teams tied on wins are ordered by their combined head-to-head record
against the other tied teams, then randomly. Real NBA rules also use division leader, division record,
conference record, etc. This only shuffles seeds between tied teams; it barely moves title odds.

Rating uncertainty: each simulated season draws every team's true strength once from
Normal(rating, sd), sd shrinking as games are played (see config.toml).
"""
import numpy as np
import pandas as pd

from season.ratings import win_prob


def _series(r, hi, lo, home_elo, rng):
    """Best-of-7 between team indices hi (home court) and lo, per sim. Returns winner indices."""
    s = np.arange(len(hi))
    p_hi_home = win_prob(r[s, hi], r[s, lo], False, home_elo)        # hi hosts games 1, 2, 5, 7
    p_hi_away = 1 - win_prob(r[s, lo], r[s, hi], False, home_elo)    # lo hosts games 3, 4, 6
    hi_home = np.array([1, 1, 0, 0, 1, 0, 1], bool)
    p = np.where(hi_home, p_hi_home[:, None], p_hi_away[:, None])
    wins_hi = (rng.random(p.shape) < p).sum(1)
    return np.where(wins_hi >= 4, hi, lo)


def _game(r, home, away, home_elo, rng):
    s = np.arange(len(home))
    return np.where(rng.random(len(home)) < win_prob(r[s, home], r[s, away], False, home_elo), home, away)


def run(tm: pd.DataFrame, played: pd.DataFrame, remaining: pd.DataFrame, ratings: pd.DataFrame, cfg: dict) -> dict:
    """tm: TEAM_ID/CONF; ratings: ELO/GAMES indexed by TEAM_ID. Returns per-sim arrays."""
    n, rng = cfg["n_sims"], np.random.default_rng(cfg["seed"])
    ids = tm["TEAM_ID"].to_numpy()
    idx = {t: i for i, t in enumerate(ids)}
    N = len(ids)

    # ---- team strength per sim (rating uncertainty)
    elo = ratings.loc[ids, "ELO"].to_numpy()
    sd = cfg["noise_preseason"] / np.sqrt(1 + ratings.loc[ids, "GAMES"].to_numpy() / cfg["noise_half_games"])
    r = elo + rng.standard_normal((n, N)) * sd                                           # (n, N)
    r_avg = np.concatenate([r, r.mean(1, keepdims=True)], axis=1)                        # col N = average opponent

    # ---- regular season: results so far + simulated remaining games
    h = remaining["HOME_ID"].map(idx).to_numpy()
    a = remaining["AWAY_ID"].map(idx).fillna(N).astype(int).to_numpy()                   # placeholder opp → N
    p = win_prob(r_avg[:, h], r_avg[:, a], remaining["NEUTRAL"].to_numpy(bool), cfg["home_elo"])
    home_won = rng.random(p.shape) < p                                                   # (n, G)
    onehot_h = np.eye(N + 1, dtype=np.float32)[h]                                        # (G, N+1)
    onehot_a = np.eye(N + 1, dtype=np.float32)[a]
    wins = (home_won @ onehot_h + (~home_won) @ onehot_a)[:, :N]
    w0 = pd.concat([played.loc[played["HOME_WIN"] == 1, "HOME_ID"], played.loc[played["HOME_WIN"] == 0, "AWAY_ID"]])
    wins += w0.map(idx).value_counts().reindex(range(N), fill_value=0).to_numpy()
    wins = wins.round().astype(int)

    # ---- head-to-head wins h2h[s, i, j] (i beat j), for the tiebreaker
    pair = (h * (N + 1) + a)
    pair_r = (a * (N + 1) + h)
    h2h = np.zeros((n, (N + 1) ** 2), np.float32)
    real = a < N
    np.add.at(h2h, (slice(None), pair[real]), home_won[:, real])
    np.add.at(h2h, (slice(None), pair_r[real]), ~home_won[:, real])
    h2h = h2h.reshape(n, N + 1, N + 1)[:, :N, :N]
    pi, pj = played["HOME_ID"].map(idx).to_numpy(), played["AWAY_ID"].map(idx).to_numpy()
    hw = played["HOME_WIN"].to_numpy() == 1
    base = np.zeros((N, N))
    np.add.at(base, (pi[hw], pj[hw]), 1)
    np.add.at(base, (pj[~hw], pi[~hw]), 1)
    h2h += base

    # ---- seeding per conference: wins, then pooled H2H among tied teams, then random
    conf = tm["CONF"].to_numpy()
    tied = (wins[:, :, None] == wins[:, None, :]) & (conf[:, None] == conf[None, :]) & ~np.eye(N, dtype=bool)
    tw, tl = (h2h * tied).sum(2), (h2h.transpose(0, 2, 1) * tied).sum(2)
    h2h_pct = np.where(tw + tl > 0, tw / np.maximum(tw + tl, 1), 0.5)
    key = wins + 0.1 * h2h_pct + 0.01 * rng.random((n, N))
    seed = np.zeros((n, N), int)
    bracket = {}
    for c in np.unique(conf):
        cols = np.where(conf == c)[0]
        order = cols[np.argsort(-key[:, cols], axis=1)]                                  # (n, 15) best first
        seed[np.arange(n)[:, None], order] = np.arange(1, len(cols) + 1)
        # play-in (home = higher seed)
        s7, s8, s9, s10 = order[:, 6], order[:, 7], order[:, 8], order[:, 9]
        w78 = _game(r, s7, s8, cfg["home_elo"], rng)
        l78 = np.where(w78 == s7, s8, s7)
        w910 = _game(r, s9, s10, cfg["home_elo"], rng)
        eighth = _game(r, l78, w910, cfg["home_elo"], rng)
        bracket[c] = np.column_stack([order[:, :6], w78, eighth])                        # seeds 1..8

    # ---- playoffs
    reached = np.zeros((n, N), int)  # 1 = made playoffs, 2 = won R1, 3 = won R2, 4 = won conf, 5 = champion
    rows = np.arange(n)[:, None]
    champs = {}
    for c, b in bracket.items():
        reached[rows, b] = 1
        r1 = [_series(r, b[:, i], b[:, j], cfg["home_elo"], rng) for i, j in ((0, 7), (3, 4), (1, 6), (2, 5))]
        for w in r1:
            reached[np.arange(n), w] = 2

        def better(x, y):  # lower seed number hosts
            sx, sy = seed[np.arange(n), x], seed[np.arange(n), y]
            return np.where(sx < sy, x, y), np.where(sx < sy, y, x)
        r2 = [_series(r, *better(r1[0], r1[1]), cfg["home_elo"], rng),
              _series(r, *better(r1[2], r1[3]), cfg["home_elo"], rng)]
        for w in r2:
            reached[np.arange(n), w] = 3
        champs[c] = _series(r, *better(r2[0], r2[1]), cfg["home_elo"], rng)
        reached[np.arange(n), champs[c]] = 4
    e, w = champs["East"], champs["West"]
    s = np.arange(n)
    e_better = wins[s, e] + 0.01 * rng.random(n) > wins[s, w] + 0.01 * rng.random(n)
    champ = _series(r, np.where(e_better, e, w), np.where(e_better, w, e), cfg["home_elo"], rng)
    reached[s, champ] = 5
    return {"ids": ids, "wins": wins, "seed": seed, "reached": reached, "conf": conf}


def summarize(out: dict, tm: pd.DataFrame, records: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    wins, seed, reached = out["wins"], out["seed"], out["reached"]
    df = pd.DataFrame({"TEAM_ID": out["ids"], "CONF": out["conf"]})
    df = df.merge(tm[["TEAM_ID", "NAME", "DIV"]], on="TEAM_ID").merge(records[["TEAM_ID", "W", "L"]], on="TEAM_ID")
    df["RATING"] = ratings.loc[df["TEAM_ID"], "ELO"].to_numpy().round(1)
    df["WINS_MEAN"] = wins.mean(0)
    for q in (10, 50, 90):
        df[f"WINS_P{q}"] = np.percentile(wins, q, axis=0)
    df["P_PLAYIN"] = ((seed >= 7) & (seed <= 10)).mean(0)
    df["P_PLAYOFFS"] = (reached >= 1).mean(0)
    df["P_LOTTERY"] = 1 - df["P_PLAYOFFS"]
    # reached: 1 made playoffs, 2 won R1, 3 won R2 (= conf finals), 4 won conf (= Finals), 5 champion
    df["P_WIN_R1"] = (reached >= 2).mean(0)
    df["P_CONF_FINALS"] = (reached >= 3).mean(0)
    df["P_FINALS"] = (reached >= 4).mean(0)
    df["P_TITLE"] = (reached >= 5).mean(0)
    for k in range(1, 16):
        df[f"P_SEED_{k}"] = (seed == k).mean(0)
    return df.sort_values("P_TITLE", ascending=False).reset_index(drop=True)


def check(df: pd.DataFrame, n: int) -> None:
    """Sanity constraints from the spec. Raises if anything is off."""
    tol = 1e-9
    assert abs(df["P_TITLE"].sum() - 1) < tol, df["P_TITLE"].sum()
    assert abs(df["P_FINALS"].sum() - 2) < tol and abs(df["P_CONF_FINALS"].sum() - 4) < tol
    assert abs(df["P_PLAYOFFS"].sum() - 16) < tol
    for _, g in df.groupby("CONF"):
        for k in range(1, len(g) + 1):
            assert abs(g[f"P_SEED_{k}"].sum() - 1) < tol, k
    assert abs(df["WINS_MEAN"].mean() - 41) < 0.1, df["WINS_MEAN"].mean()   # placeholder games add a hair of slack


if __name__ == "__main__":
    import time

    from pipeline.fetch_data import CURRENT_SEASON
    from season import inputs
    from season.ratings import load_config, team_ratings
    cfg = load_config()
    today = pd.Timestamp.today().normalize()
    tm, played, rem = inputs.as_of(CURRENT_SEASON, today)
    ratings = team_ratings(CURRENT_SEASON, today, tm["TEAM_ID"], cfg)
    t0 = time.perf_counter()
    out = run(tm, played, rem, ratings, cfg)
    t1 = time.perf_counter()
    df = summarize(out, tm, inputs.records(tm, played), ratings)
    check(df, cfg["n_sims"])
    print(f"{cfg['n_sims']:,} seasons x {len(rem)} remaining games in {t1 - t0:.1f}s; sanity checks passed")
    print(df[["NAME", "RATING", "WINS_P10", "WINS_P50", "WINS_P90", "P_PLAYOFFS", "P_FINALS", "P_TITLE"]]
          .head(10).round(3).to_string(index=False))
