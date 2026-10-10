# NBA Game Predictor

Predicts win probability, projected score and margin for each NBA game of the day. Output goes to `data/predictions.json`; a Streamlit app (`app.py`) only reads that file and never calls nba_api.

## Environment
- Always `source venv/bin/activate` first (conda base is on by default; never pip install into it).
- Run modules from the project root: `python -m pipeline.fetch_data`.
- Project lives at `~/Projects/NBA Model`; `~/Desktop/NBA Model` is a shortcut (symlink) to it. Moved 2026-10-09 because
  macOS blocks launchd jobs from reading ~/Desktop. Folder name has a space: quote it.

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

## App (step 7)
- `python -m pipeline.team_state` → `data/team_state.json` (committed: the app needs it). Elo (carried into new season), last-10 team stats, current rosters (CommonTeamRoster, cached per day) with player value / minutes / glue.
- `predict.py`: `predict_matchup(state, art, home, away, out_home, out_away, rest_home, rest_away, neutral)` builds training-identical features, reuse it for daily predictions (step 8).
- Headline win prob = simulation (points model), so prob, score and margin agree. Logistic gives the "why" breakdown. The two differ by 2.6 pts median on test games.
- `streamlit run app.py`: tabs Try a matchup / Power ratings / Track record / How it works. Never calls nba_api.
- Known limit: team ratings are last season's until 2026-27 games are ingested (step 8).

## Daily pipeline (step 8)
- `python -m pipeline.predict_today [--date YYYY-MM-DD]`: refresh current-season logs (if >6h old), weekly hustle snapshot, rosters, schedule (`ScheduleLeagueV2`, game IDs `002…` only), ESPN injuries → `data/team_state.json`, `data/predictions.json`, `data/history/predictions_<date>.json`. Default date = next game day on/after today (US Eastern).
- Injuries (`features/injuries.py`, swappable): "Out"/"Doubtful" = missing; "Day-To-Day" shown but assumed to play. Teams matched by nickname (ESPN "LA Clippers").
- Missing-player rule matches training (played for the team in its last 10 games this season), except before a team's 10th game, when every rotation player counts.
- Rest / 3-in-4 come from the schedule. Empty current season is handled (`_concat` skips empty frames).
- GitHub Actions `.github/workflows/daily.yml`: 15:00 + 22:00 UTC, commits results. Historical raw data is committed so the job never re-downloads it.
- Season rollover: bump `CURRENT_SEASON` + `SEASONS` in fetch_data.py and the season in .gitignore. Retrain (`features.build` → `models.train`) monthly, not in the daily job.

## Live track record
- `pipeline/track_record.py`: `save_history` freezes each game's prediction once it tips off; `update` grades history vs current-season results → `data/track_record.json` (shown on the Games page). Runs inside predict_today.

## Season outlook (season/, pipeline/simulate_season.py, views/2_Season_Outlook.py)
- `season/inputs.py`: teams/conf/div (`LeagueStandingsV3`), played + remaining games as of any date. Current season: schedule + placeholders vs league-average opponent for the ~30 unscheduled/TBD Cup games (60 team slots). Verified: rebuilt records = official standings, 2023-26.
- `season/ratings.py`: Elo as of date (configurable offseason regression via `add_elo(carry=)`), roster-change adjustment (top-8 last-season Game Score delta × roster_k, fading over 30 games), long-term injury adjustment (live only), fast `win_prob`. Fast path vs full game model on test: mean |diff| 0.055, corr 0.94.
- `season/simulate.py`: vectorized 10k seasons in ~5s. Play-in + 2-2-1-1-1 best-of-7. Tiebreak SIMPLIFIED: pooled H2H among tied teams, then random. `check()` enforces sums (title 100%, seeds, 41 avg wins). Rule tests in tests/test_season_sim.py.
- `season/backtest.py` (~6 min): tuned config.toml → regression 0.33, roster_k 3, noise 75 preseason / half-life 40 games. Preseason win MAE 7.1 (everyone-41: 9.8), 80% ranges cover 78%; ~20 games in 5.1; ~50 games in 2.9. Champion's avg preseason title odds 15% (uniform 3%). No playoff-strength bonus (optional, not built).
- `python -m pipeline.simulate_season [--force]` → data/season_outlook.json + season_outlook_history.csv; skips (but stamps last_checked) when no new games. In the daily workflow.
- App: `app.py` is a router (st.navigation, nav on top) → views/1_Games.py, views/2_Season_Outlook.py (url /season).
  Folder is `views/`, not `pages/`: that name triggers Streamlit's old auto-navigation on direct links.
- Site name: The Hardwood Model. Look: `.streamlit/config.toml` (dark theme, amber accent, Barlow Condensed + IBM Plex
  Mono) + `ui.py` (CSS, game cards, odds list). Team logos live in `static/logos/` (from cdn.nba.com) and are embedded
  as data URIs, because the NBA CDN refuses some browsers.

## Automation (actual setup)
- GitHub Actions can't reach stats.nba.com (tested 2026-10-09: 60s timeout), so `.github/workflows/daily.yml` is manual-only.
- The Mac runs `scripts/daily_local.sh` (real path, not the Desktop shortcut) via launchd (`~/Library/LaunchAgents/com.nbapredictor.daily.plist`) at 9:00 and 16:00 local (11am / 6pm ET): pull, predict_today, simulate_season, commit data, push → Streamlit Cloud redeploys. Missed runs (Mac asleep) fire on wake. Log: `logs/daily.log`.
- Run now: `launchctl start com.nbapredictor.daily`. Disable: `launchctl unload ~/Library/LaunchAgents/com.nbapredictor.daily.plist`.
