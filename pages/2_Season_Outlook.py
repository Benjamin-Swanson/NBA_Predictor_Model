"""Season outlook page: championship odds, projected standings, team detail.
Reads data/season_outlook.json, data/season_outlook_history.csv, data/season_backtest.json only."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path("data")
out_path = DATA / "season_outlook.json"
if not out_path.exists():
    st.info("Season odds haven't been generated yet. Run `python -m pipeline.simulate_season`.")
    st.stop()
o = json.loads(out_path.read_text())
teams = pd.DataFrame(o["teams"])
bt = json.loads((DATA / "season_backtest.json").read_text()) if (DATA / "season_backtest.json").exists() else None


def pct(p: float) -> str:
    """Never show exactly 0% or 100%: a simulated 0 means 'rare', not impossible."""
    return "<1%" if p < 0.01 else ">99%" if p > 0.99 else f"{p:.0%}"


# ---------------------------------------------------------------- header + freshness
et = "America/New_York"
updated = pd.Timestamp(o["last_updated"]).tz_convert(et)
checked = pd.Timestamp(o.get("last_checked", o["last_updated"])).tz_convert(et)
st.title(f"🏆 {o['season']} season outlook")
st.markdown(f"**Odds last updated {updated:%b %d, %I:%M %p} ET** · {o['games_played']} games played · "
            f"{o['config']['n_sims']:,} simulated seasons")
if pd.Timestamp.now(tz=et) - checked > pd.Timedelta(days=2):
    st.warning(f"These odds haven't been refreshed since {checked:%b %d}. The daily update may have failed.")
elif o["games_played"] == 0:
    st.caption("Preseason odds: based on last season's ratings and offseason roster moves. "
               "They start updating after opening night.")

# ---------------------------------------------------------------- hero: favorites
st.subheader("Championship favorites")
top = teams.nlargest(10, "p_title")
left, right = st.columns([3, 2])
with left:
    chart = top.set_index("name")["p_title"].mul(100).rename("Title chance (%)")
    st.bar_chart(chart, horizontal=True, color="#c8102e", sort="-Title chance (%)", height=380)
with right:
    st.dataframe(pd.DataFrame({
        "Team": top["abbr"], "Title": top["p_title"].map(pct), "Finals": top["p_finals"].map(pct),
        "Proj. wins": top["wins_p50"]}), hide_index=True, width="stretch", height=390)
st.caption("A 20% favorite still fails to win the title 4 times in 5. These are chances, not predictions.")

# ---------------------------------------------------------------- conference views
st.subheader("Projected standings")
for tab, conf in zip(st.tabs(["East", "West"]), ["East", "West"]):
    with tab:
        c = teams[teams["conf"] == conf].sort_values(["wins_mean", "p_title"], ascending=False)
        st.dataframe(pd.DataFrame({
            "Team": c["name"], "Record": c["record"], "Proj. wins": c["wins_p50"],
            "Range (10-90%)": c["wins_p10"].astype(str) + "–" + c["wins_p90"].astype(str),
            "Playoffs": c["p_playoffs"].map(pct), "Play-in": c["p_playin"].map(pct), "Lottery": c["p_lottery"].map(pct),
            "Conf. finals": c["p_conf_finals"].map(pct), "Finals": c["p_finals"].map(pct), "Title": c["p_title"].map(pct)}),
            hide_index=True, width="stretch", height=560)
        with st.expander("Seed odds"):
            seeds = pd.DataFrame(c["p_seed"].tolist(), index=c["name"], columns=[str(k) for k in range(1, 16)])
            st.dataframe(seeds.map(lambda p: "" if p == 0 else pct(p)), width="stretch", height=560)
            st.caption("Seeds 1-6 go straight to the playoffs, 7-10 go to the play-in, 11-15 go to the lottery.")

# ---------------------------------------------------------------- team detail
st.subheader("Team detail")
pick = st.selectbox("Team", teams.sort_values("name")["abbr"], format_func=lambda a: teams.set_index("abbr").loc[a, "name"])
t = teams.set_index("abbr").loc[pick]
m = st.columns(5)
m[0].metric("Record", t["record"])
m[1].metric("Projected wins", int(t["wins_p50"]), f"80% range {t['wins_p10']}–{t['wins_p90']}", delta_color="off")
m[2].metric("Playoffs", pct(t["p_playoffs"]))
m[3].metric("Finals", pct(t["p_finals"]))
m[4].metric("Title", pct(t["p_title"]))
a, b = st.columns(2)
with a:
    st.markdown("**Final win total, across simulated seasons**")
    dist = pd.Series(t["wins_dist"], name="share of seasons")
    lo, hi = max(int(t["wins_p10"]) - 10, 0), min(int(t["wins_p90"]) + 10, 82)
    st.bar_chart(dist.loc[lo:hi], x_label="wins", y_label="share of seasons", color="#1d428a")
with b:
    st.markdown("**Seed**")
    st.bar_chart(pd.Series(t["p_seed"], index=range(1, 16), name="chance"), x_label="seed", y_label="chance",
                 color="#1d428a")
hist_path = DATA / "season_outlook_history.csv"
if hist_path.exists():
    h = pd.read_csv(hist_path, parse_dates=["date"])
    h = h[h["abbr"] == pick].set_index("date")["p_title"].mul(100).rename("Title chance (%)")
    st.markdown("**Title chance over the season**")
    if len(h) > 1:
        st.line_chart(h, color="#c8102e")
    else:
        st.caption("The trend line starts once the odds have been updated on more than one day.")

# ---------------------------------------------------------------- methodology
with st.expander("How these odds are made"):
    cfg = o["config"]
    st.markdown(f"""
**The short version:** we play out the rest of the season {cfg['n_sims']:,} times, including the play-in and
every playoff series, and count how often each team finishes where.

- **Team strength** is each team's Elo rating, the same rating behind the game predictions. Over the
  offseason every team gives back **{cfg['regression']:.0%}** of its distance from average, because teams change.
- **Uncertainty about strength.** In each simulated season every team's true strength is drawn at random
  around its rating (spread **±{cfg['noise_preseason']:.0f} Elo** before the season, shrinking as games are
  played). Without this, the favorites look far too safe.
- **Games** are decided by coin flips weighted by the two ratings plus home court.
- **Playoffs** follow the current format: seeds 7-10 play in, then best-of-7 series with home court
  (2-2-1-1-1) going to the better seed.
- **Tiebreakers are simplified:** tied teams are ordered by their head-to-head record against each other,
  then at random. The NBA's full rules (division leaders, conference record, ...) are not modeled.
- **Unscheduled games.** About 30 games (NBA Cup knockout games and their make-up dates) aren't scheduled
  until December. Until then they're played against a league-average opponent.
- **Injuries:** only long-term absences (out more than two weeks) lower a team's season rating.
""")
    if bt:
        pre, g20 = bt["by_checkpoint"]["preseason"], bt["by_checkpoint"]["g20"]
        st.markdown(f"""
**How well this has worked.** Tested on {len(bt['seasons'])} past seasons ({bt['seasons'][0]} to {bt['seasons'][-1]},
skipping the 2019-20 bubble), using only what was known at the time:

- Preseason win totals missed by **{pre['win_mae']:.1f} wins** per team on average (guessing 41 for everyone
  misses by {bt['naive_win_mae']:.1f}). By ~20 games in: **{g20['win_mae']:.1f}**.
- The 80% win range contained the real total **{pre['win_cover_80']:.0%}** of the time (it should be about 80%).
- The eventual champion was given a **{pre['champ_p']:.0%}** preseason title chance on average.
""")
        st.dataframe(pd.DataFrame(bt["per_season"]).rename(columns={
            "season": "Season", "favorite": "Preseason favorite", "champion": "Champion",
            "champion_odds": "Champion's preseason odds", "win_mae": "Win total miss"}).assign(
            **{"Champion's preseason odds": lambda d: d["Champion's preseason odds"].map(pct)}),
            hide_index=True, width="stretch")
    st.caption("These are probabilities, not predictions. Upsets are part of the math.")
