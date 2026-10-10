"""Games page: scheduled games, matchup tool, power ratings, track record, how it works.
Reads data/*.json + models/model.joblib only; never calls nba_api."""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui
from predict import load, predict_matchup, team_levels

@st.cache_resource
def cached_load():
    return load()


state, art = cached_load()
teams = state["teams"]
names = {abbr: t["name"] for abbr, t in teams.items()}
GROUP_LABELS = {"elo": "Team rating (Elo)", "availability": "Players out", "glue": "Glue guys (hustle)",
                "rest": "Rest", "four_factors": "Four Factors", "efficiency": "Off/def rating",
                "recent_form": "Recent margin", "pace": "Pace", "neutral_site": "Neutral site"}
REST = {"Back-to-back": 1, "1 day off": 2, "2+ days off": 3}

ui.page_head(f"{state['season']} season · game predictions", "Every game, simulated 10,000 times",
             f"Win chances and projected scores · {state['season']} rosters · team ratings through {state['as_of']}")

tab_today, tab_match, tab_ratings, tab_record, tab_how = st.tabs(
    ["Games", "Try a matchup", "Power ratings", "Track record", "How it works"])

# ---------------------------------------------------------------- scheduled games
with tab_today:
    pred_path = Path("data/predictions.json")
    preds = json.loads(pred_path.read_text()) if pred_path.exists() else None
    if not preds or not preds["games"]:
        st.info("No upcoming games found yet. Predictions appear here once the schedule is out.")
    else:
        day = pd.Timestamp(preds["date"])
        st.subheader(f"{day:%A, %B} {day.day}")
        st.html(f'<div class="hw-meta"><span class="hw-chip">{len(preds["games"])} games</span>'
                f"Updated {pd.Timestamp(preds['generated_at']).tz_convert('America/New_York'):%b %d, %I:%M %p} ET · "
                "injuries from ESPN · players listed Out are removed, Day-To-Day assumed to play</div>")
        ui.games_grid(preds["games"])

# ---------------------------------------------------------------- matchup
with tab_match:
    c1, c2 = st.columns(2)
    order = sorted(teams, key=lambda a: names[a])
    away = c1.selectbox("Away team", order, index=order.index("BOS"), format_func=lambda a: names[a])
    home = c2.selectbox("Home team", order, index=order.index("NYK"), format_func=lambda a: names[a])
    if away == home:
        st.warning("Pick two different teams.")
        st.stop()

    def player_label(team):
        lookup = {p["id"]: p for p in teams[team]["players"]}
        return lambda pid: (f"{lookup[pid]['name']}  ·  value {lookup[pid]['value']:.1f}"
                            if lookup[pid]["value"] is not None else f"{lookup[pid]['name']}  ·  no NBA history")

    with st.expander("Game settings: injuries, rest, venue"):
        s1, s2 = st.columns(2)
        out_away = s1.multiselect(f"{teams[away]['name']} players out", [p["id"] for p in teams[away]["players"]],
                                  format_func=player_label(away))
        out_home = s2.multiselect(f"{teams[home]['name']} players out", [p["id"] for p in teams[home]["players"]],
                                  format_func=player_label(home))
        rest_away = s1.radio(f"{away} rest", list(REST), index=1, horizontal=True)
        rest_home = s2.radio(f"{home} rest", list(REST), index=1, horizontal=True)
        neutral = st.checkbox("Neutral site (no home-court edge)")
        st.caption("Value = average Game Score over the last 20 games (a one-number box score summary; "
                   "~10 is a solid starter, 20+ is a star). Only players averaging 10+ minutes move the needle.")

    r = predict_matchup(state, art, home, away, out_home, out_away, REST[rest_home], REST[rest_away],
                        neutral, keep_draws=True)
    p_home = r["home_win_prob"]
    fav, p_fav = (home, p_home) if p_home >= 0.5 else (away, 1 - p_home)

    ui.matchup_panel(away, home, names, p_home, r["mu_away"], r["mu_home"], neutral)
    st.markdown(f"**{p_fav:.0%}** means the model expects the **{names[fav]}** to win about "
                f"**{round(p_fav * 10)} times in 10** games like this one. The favorite is not a sure thing.")

    k1, k2, k3 = st.columns(3)
    margin = r["mu_home"] - r["mu_away"]
    k1.metric("Projected score", f"{away} {r['mu_away']:.0f} – {home} {r['mu_home']:.0f}")
    k2.metric("Expected margin", f"{fav} by {abs(margin):.1f}",
              f"90% range: {home} {r['margin_pct'][5]:+.0f} to {r['margin_pct'][95]:+.0f}", delta_color="off")
    k3.metric("Total points", f"{r['total_mean']:.0f}",
              f"90% range: {r['total_pct'][5]:.0f}–{r['total_pct'][95]:.0f}", delta_color="off")

    left, right = st.columns(2)
    with left:
        st.subheader(f"{home} margin in 10,000 simulated games")
        bins = np.arange(-50, 52, 2)
        counts, _ = np.histogram(np.clip(r["margin_draws"], -49.9, 49.9), bins)
        st.bar_chart(pd.DataFrame({"simulated games": counts}, index=bins[:-1] + 1), x_label=f"{home} margin",
                     y_label="games", color="#f5a524")
        st.caption(f"Bars right of 0 are {home} wins.")
    with right:
        st.subheader("What drives this prediction")
        why = r["why"].rename(GROUP_LABELS).sort_values(key=abs, ascending=False)
        why = why[why.abs() > 0.005]
        st.bar_chart(pd.DataFrame({f"→ favors {home}  /  ← favors {away}": why}), horizontal=True, color="#f5a524", sort=False)
        st.caption("Each factor's push on the odds, from the win-probability model. Bars to the right favor "
                   f"{home}, to the left favor {away}.")

# ---------------------------------------------------------------- ratings
with tab_ratings:
    rows = []
    for abbr, t in teams.items():
        rows.append({"": ui.logo(abbr), "Team": t["name"], "Elo": round(t["elo"]), "Net rating (last 10)": round(t["recent"]["NET"], 1),
                     "Glue (full roster)": round(team_levels(t)["GLUE"], 2),
                     "Top players": ", ".join(p["name"] for p in t["players"][:3])})
    df = pd.DataFrame(rows).sort_values("Elo", ascending=False)
    df.insert(0, "#", range(1, len(df) + 1))
    st.dataframe(df, hide_index=True, width="stretch", height=35 * (len(df) + 1) + 3, column_config={"": ui.LOGO_COL})
    st.caption("Elo is the model's main team rating (1500 = average). Before games are played this season it's "
               "last season's final rating pulled a quarter of the way back to average.")

# ---------------------------------------------------------------- track record
with tab_record:
    st.subheader(f"This season ({state['season']})")
    rec_path = Path("data/track_record.json")
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"n": 0}
    if not rec["n"]:
        st.info("Every prediction is saved before tip-off and graded after the final buzzer. "
                "The live record starts with the first games of the season.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Games graded", rec["n"])
        b.metric("Picked the winner", f"{rec['accuracy']:.1%}")
        c.metric("Avg. margin miss", f"{rec['margin_mae']:.1f} pts")
        d.metric("Final margin inside 90% range", f"{rec['band_coverage']:.0%}")
        st.caption(f"Log loss {rec['log_loss']:.3f}, Brier {rec['brier']:.3f} (lower is better; "
                   "a coin flip scores 0.693 and 0.250).")
        bk = pd.DataFrame(rec["buckets"]).rename(columns={"bucket": "Model confidence", "games": "Games",
                                                          "predicted": "Expected win rate", "actual": "Actual win rate"})
        st.dataframe(bk.style.format({"Expected win rate": "{:.0%}", "Actual win rate": "{:.0%}"}),
                     hide_index=True, width="stretch")
        st.caption("Honest percentages mean the favorite's actual win rate tracks the expected rate in each row.")
        games = pd.DataFrame(rec["games"])
        games["Result"] = np.where(games["correct"] == 1, "✅", "❌")
        games["Game"] = games["away"] + " @ " + games["home"]
        games["Away"], games["Home"] = games["away"].map(ui.logo), games["home"].map(ui.logo)
        games["Pick"] = games["pick"] + " " + (games["confidence"] * 100).round().astype(int).astype(str) + "%"
        st.dataframe(games.rename(columns={"date": "Date", "proj": "Projected", "final": "Final"})
                     [["Date", "Away", "Home", "Game", "Pick", "Projected", "Final", "Result"]],
                     hide_index=True, width="stretch", height=400,
                     column_config={"Away": ui.LOGO_COL, "Home": ui.LOGO_COL})

    st.subheader("Past seasons (held-out test)")
    res = art["results"]
    test = res[res["split"] == "test"].set_index("model")
    nice = {"elo": "Elo only", "logreg_minimal": "Simple model", "logreg_full": "Full model",
            "lgbm_full": "Gradient boosting", "points_model": "Score model (powers the app)"}
    acc = test.loc["points_model", "accuracy"]
    st.markdown(f"Tested on **{int(test['n'].iloc[0]):,} games** from 2024-25 and 2025-26 that the model never saw "
                f"during training. The model picked the winner in **{acc:.1%}** of them; always picking the home "
                f"team gets about 55%.")
    tbl = test.rename(index=nice)[["accuracy", "log_loss", "brier"]]
    tbl.columns = ["Accuracy", "Log loss (lower is better)", "Brier score (lower is better)"]
    st.dataframe(tbl.style.format({"Accuracy": "{:.1%}", "Log loss (lower is better)": "{:.3f}",
                                   "Brier score (lower is better)": "{:.3f}"}), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Are the percentages honest?")
        cal = art["calibration"].assign(model=lambda d: d["model"].map(nice))
        order = [nice[m] for m in ("points_model", "logreg_full", "elo")]
        color = alt.Color("model:N", title=None, sort=order, legend=alt.Legend(orient="bottom", labelLimit=0),
                          scale=alt.Scale(domain=order, range=["#f5a524", "#8fb7d9", "#6f695c"]))
        pct_axis = dict(format="%", values=[0, .2, .4, .6, .8, 1])
        x = alt.X("predicted:Q", title="Model's home-win chance", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(**pct_axis))
        y = alt.Y("actual:Q", title="How often the home team won", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(**pct_axis))
        tip = [alt.Tooltip("model:N", title="Model"), alt.Tooltip("predicted:Q", title="Predicted", format=".0%"),
               alt.Tooltip("actual:Q", title="Actual", format=".0%"), alt.Tooltip("games:Q", title="Games")]
        diag = alt.Chart(pd.DataFrame({"predicted": [0, 1], "actual": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="#6f695c").encode(x=x, y=y)
        base = alt.Chart(cal).encode(x=x, y=y, color=color, order=alt.Order("bin:Q"), tooltip=tip)
        st.altair_chart(diag + base.mark_line(strokeWidth=2) + base.mark_circle(opacity=1).encode(
            size=alt.Size("games:Q", legend=None, scale=alt.Scale(range=[15, 160]))), height=380)
        st.caption("When the model says 70%, the home team should win about 70% of the time. Points near the dotted "
                   "line mean the percentages can be taken at face value. Bigger dots = more games.")
    with c2:
        st.subheader("What matters most")
        imp = art["importance"].clip(lower=0)
        st.bar_chart((imp / imp.sum() * 100).rename(GROUP_LABELS).sort_values(ascending=False), horizontal=True, sort=False,
                     x_label="share of importance (%)", color="#f5a524")
        st.caption("How much worse predictions get when each factor is scrambled (2023-24 season).")

# ---------------------------------------------------------------- how it works
with tab_how:
    st.markdown(f"""
### The short version
For each game the model estimates how many points each team will score, then plays the game
**10,000 times** with realistic randomness. The win chance is the share of those games each team wins.

### What goes in
- **Team rating (Elo).** Goes up when a team wins, more for big wins and road wins. By far the strongest signal.
- **Players out.** How much production is missing, from each absent player's recent Game Score.
  Mark injuries in *Game settings* to see the effect.
- **Glue guys.** A custom hustle index: deflections, screen assists, loose balls, contested shots, charges,
  box outs and on/off impact. It improved accuracy in testing (it's kept only because it did).
- **Rest.** Back-to-backs and days off.
- **Recent form.** Offensive/defensive rating, pace and the Four Factors over the last 10 games.
- **Home court.** Worth roughly 2 points now, less than it was before 2020.

### How it was built
- NBA stats from 2015-16 to 2025-26 via `nba_api`. Trained on {art['train_seasons'][0]}-{str(art['train_seasons'][1] + 1)[-2:]},
  tuned on 2023-24, tested on 2024-25 and 2025-26.
- Every feature uses only information available **before** tip-off.
- Home and away scores are simulated together: fast games and overtime raise both scores at once.

### Limits
- Early in a season, team ratings are mostly last season's. Offseason trades change the rosters
  (and the players-out and glue numbers) but not yet the team rating.
- Injuries are entered by hand here. Last-minute scratches and "questionable" players are the hardest part to predict.
- The NBA is noisy: even a 70% favorite loses 3 times in 10.
""")
