# NBA Game Predictor

Predicts win probability, projected score and margin for each NBA game of the day. Output goes to `data/predictions.json`; a Streamlit app (`app.py`) only reads that file and never calls nba_api.

## Environment
- Always `source venv/bin/activate` first (conda base is on by default; never pip install into it).
- Run modules from the project root: `python -m pipeline.fetch_data`.
- Folder name has a space: `cd ~/Desktop/"NBA Model"`.

## Hard rules
- **No leakage**: a game's features use only data from before its tip-off. Rolling stats are shifted; nba_api filters use `date_to_nullable` = the day before the game.
- Split train/test by date, never randomly. Current plan: train 2015-16..2022-23, validate 2023-24, test 2024-25 + 2025-26.
- Cache every raw pull in `data/raw/` and never re-fetch a completed season. `time.sleep` between API calls.
- Print `df.columns` the first time you use any endpoint; don't assume column names.

## Data facts (verified)
- `LeagueGameLog(player_or_team_abbreviation="T")`: one row per team per game, 29 columns, including box stats (FGA, FTA, OREB, TOV, PTS, PLUS_MINUS, MIN=240 for regulation team minutes). One call per season.
- `GAME_ID` is a zero-padded string. Home team has "vs." in `MATCHUP`, away team has "@".
- 10 neutral-site games since 2024-25 have "@" for both teams → treat as home advantage 0.
- 2019-20 (1,059 games, bubble) and 2020-21 (72-game season, few fans) are short and have weak home-court advantage.

## Decisions
- Possessions, off/def rating and Four Factors are computed from box scores ourselves (no per-game advanced-stat calls).
- Historical availability = who actually appeared in the box score. Live availability = an isolated injury module (source swappable).
- Glue guy index (`features/glue_index.py`): user's own unvalidated feature. Use weekly on/off snapshots instead of per-game calls. It must pass an ablation test to stay in the model.
- Prior feature weights in the handoff are the user's guesses, not truth; compare them with permutation importance after training.
- Points are not Poisson: the Monte Carlo layer (`simulate.py`) uses Normal draws from model-predicted mean/sigma.
- Betting lines (optional, later): always evaluate with and without them.

## Pipeline so far
1. `python -m pipeline.fetch_data`: raw logs → `data/raw/`
2. `python -m features.build`: per-game features → `data/processed/games.parquet` (runs a leakage self-check)
3. `python -m models.baselines`: scores baselines

## Baselines to beat (test = 2024-25 + 2025-26, 2,460 games)
| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Home always wins | 0.689 | 0.248 | 54.9% |
| Elo (HCA 60) | 0.604 | 0.208 | 67.5% |
| logreg_full (step 4, chosen on val) | 0.602 | 0.208 | 67.8% |

## Step 4 findings (val = 2023-24)
- Model choice uses val only; test is reported, never tuned on. `python -m models.train` writes `models/model.joblib` + `models/calibration.png`.
- ML models barely beat Elo (test log loss 0.604 → 0.601-0.602). All are well calibrated.
- Elo carries almost everything (corr with rolling net rating 0.81). Drop-one-group ablation on val: Elo +0.021, rest +0.004, four factors +0.0003, efficiency -0.0004 (redundant), form/pace ~0.
- LightGBM doesn't beat logistic regression on this feature set.
- Home and away scores are positively correlated (r ≈ 0.25): the Monte Carlo layer must draw them jointly, not independently. Per-team sigma ≈ 11.6, margin sigma ≈ 14.1.

## Availability (features/availability.py)
- Missing value = sum / max of pre-game avg Game Score (last 20 games) for recent rotation players (>=10 MPG, played for the team this season within its last 10 games) with no box-score row. Cached in `data/processed/availability.parquet`, rebuilt when raw player logs change.
- Ablation, test log loss: logreg 0.602 → 0.595, LightGBM 0.605 → 0.596; accuracy up to 68.3-68.5%. Availability is now the #2 feature group after Elo.
- Caveat: historical availability is perfect hindsight (who actually played). Live uses the injury report (questionable players, late scratches), so expect live gains to be smaller.
- Tests: `pytest` (synthetic roster cases). Spot check: `python -m features.availability DEN 2025-03-10`.

## Glue guy index (features/glue_index.py), validated
- Hustle from weekly `leaguehustlestatsplayer` snapshots (266 cached in data/raw/hustle/, fetch retries on timeout). On/off from box-score plus/minus.
- Ablation (train 2016-17..2022-23): val 0.6014 → 0.5959, test 0.5945 → 0.5912. Test bootstrap 95% CI [-0.0058, -0.0008], better in 99.4% of resamples.
- Split: hustle-only -0.0015 (significant), on/off-only -0.0020 (not significant), full formula best. Glue is in the main model.
- Skews toward bigs (screen assists, contests, box outs). Untuned knobs: PRIOR_GAMES=15, ONOFF_SHRINK_MIN=1000, component weights.

## Current model (step 5): logreg_full, train 2016-17..2022-23
Test: log loss 0.591, Brier 0.203, accuracy 68.1% (LightGBM 69.2% acc but worse val log loss). Importance: Elo 78%, availability 10.5%, glue 4.9%, rest 2.6%.

## Monte Carlo (simulate.py), step 6
- `simulate(mu_home, sigma_home, mu_away, sigma_away, n, rho)`: joint Normal draws; returns win prob, margin/total/score percentiles.
- rho = correlation of home/away points-model residuals on val (0.277), saved in model.joblib. sigma is constant per side (≈11.6/11.4); ponytail: per-matchup sigma if residuals prove heteroscedastic.
- Test check (`python simulate.py`): log loss 0.5912 = logistic model; 90% margin band covers 89.5%, total band 89.6%. Independent draws: margin band 93.6% (too wide), total 84.5% (too narrow), log loss 0.5962.
