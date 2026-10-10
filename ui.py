"""Shared look for the site: team logos, CSS, and the HTML pieces (game cards, matchup panel, odds list).
Colors and fonts match .streamlit/config.toml."""
import base64
from functools import cache
from html import escape
from pathlib import Path

import streamlit as st

LOGOS = Path(__file__).resolve().parent / "static" / "logos"  # from cdn.nba.com/logos/nba/<team id>/primary/L/logo.svg
LOGO_COL = st.column_config.ImageColumn("", width=40)


@cache
def logo(abbr: str) -> str:
    """Embedded as a data URI: the NBA CDN refuses some browsers, and nothing extra has to be served."""
    return "data:image/svg+xml;base64," + base64.b64encode((LOGOS / f"{abbr}.svg").read_bytes()).decode()


def img(abbr: str, size: int = 32) -> str:
    return f'<img class="hw-logo" src="{logo(abbr)}" width="{size}" height="{size}" alt="{abbr}">'


CSS = """
:root{--paper:#12100d;--panel:#1c1915;--panel2:#24201a;--rule:#2e2a23;--rule-strong:#463f33;--ink:#f2ede0;
--muted:#a39d8d;--faint:#6f695c;--accent:#f5a524;--dim:#4a4337;--win:#4cc06d;--loss:#e05a3a;--hot:#e6c35c;
--head:'Barlow Condensed','Arial Narrow',sans-serif;--mono:'IBM Plex Mono',ui-monospace,monospace}
.block-container{max-width:1180px;padding-top:3.5rem}
h1,h2,h3{font-family:var(--head)!important;text-transform:uppercase;letter-spacing:.02em}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px 16px}
[data-testid="stMetricValue"]{font-family:var(--mono);font-size:1.6rem}
[data-testid="stMetricLabel"] p{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.stTabs [data-baseweb="tab"] p{font-family:var(--head);font-size:1.15rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
[data-testid="stImage"] img{border-radius:8px}
img.hw-logo{filter:drop-shadow(0 0 1px rgba(242,237,224,.7))}
.tnum{font-family:var(--mono);font-variant-numeric:tabular-nums}
.hw-brand{display:flex;align-items:center;gap:12px;padding-bottom:14px;border-bottom:1px solid var(--rule);margin-bottom:6px}
.hw-mark{width:34px;height:34px;border-radius:50%;background:var(--accent);position:relative;flex:none;
  box-shadow:inset 0 0 0 2px #12100d}
.hw-mark:before,.hw-mark:after{content:"";position:absolute;background:#12100d}
.hw-mark:before{left:16px;top:0;width:2px;height:34px}.hw-mark:after{top:16px;left:0;height:2px;width:34px}
.hw-word{font-family:var(--head);font-weight:700;font-size:1.6rem;letter-spacing:.06em;text-transform:uppercase;line-height:1}
.hw-word span{color:var(--accent)}
.hw-tag{margin-left:auto;font-family:var(--mono);font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
.hw-kicker{font-family:var(--mono);font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
.hw-title{font-family:var(--head);font-weight:700;font-size:2.7rem;line-height:1;text-transform:uppercase;margin:.25rem 0 .4rem}
.hw-meta{color:var(--muted);font-size:.86rem;margin-bottom:.6rem}
.hw-chip{display:inline-block;font-family:var(--mono);font-size:.72rem;border:1px solid var(--rule-strong);border-radius:99px;
  padding:2px 10px;color:var(--ink);margin-right:6px}
.hw-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.hw-card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px 16px}
.hw-card-top span:first-child{white-space:nowrap}
.hw-card-top{display:flex;justify-content:space-between;gap:8px;font-family:var(--mono);font-size:.7rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
.hw-team .hw-name{font-size:1.15rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hw-team{display:grid;grid-template-columns:34px minmax(0,1fr) auto 3.4em;align-items:center;gap:10px;padding:4px 0}
.hw-name{font-family:var(--head);font-size:1.3rem;font-weight:600;text-transform:uppercase;letter-spacing:.02em}
.hw-pts{font-family:var(--mono);color:var(--muted);font-size:.82rem}
.hw-prob{font-family:var(--mono);font-weight:600;font-size:1.15rem;text-align:right;color:var(--muted)}
.hw-fav .hw-prob{color:var(--accent)}
.hw-bar{display:flex;height:6px;border-radius:3px;overflow:hidden;background:var(--dim);margin:10px 0 10px}
.hw-bar i{display:block;background:var(--dim)}.hw-bar i.on{background:var(--accent)}
.hw-foot{font-size:.8rem;color:var(--muted);display:flex;flex-wrap:wrap;gap:2px 14px}
.hw-foot b{color:var(--ink);font-weight:600}
details.hw-inj{margin-top:8px;font-size:.8rem;color:var(--muted);border-top:1px solid var(--rule);padding-top:6px}
details.hw-inj summary{cursor:pointer;font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.06em}
.hw-out{color:var(--loss)}.hw-dtd{color:var(--hot)}
.hw-match{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;background:var(--panel);border:1px solid var(--rule);
  border-radius:10px;padding:20px 24px 14px;text-align:center}
.hw-side .hw-name{font-size:1.5rem;margin-top:6px}
.hw-big{font-family:var(--mono);font-size:2.6rem;font-weight:600;line-height:1.1;color:var(--muted)}
.hw-side.hw-fav .hw-big{color:var(--accent)}
.hw-vs{font-family:var(--head);font-size:1.4rem;color:var(--faint);padding:0 12px}
.hw-match .hw-bar{grid-column:1/4;height:8px;margin:16px 0 4px}
.hw-list{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:6px 0}
.hw-row{display:grid;grid-template-columns:2.2em 30px minmax(7em,1fr) 2fr 4em 4.4em 4em;align-items:center;gap:10px;padding:7px 16px;
  border-top:1px solid var(--rule)}
.hw-row:first-child{border-top:none}
.hw-row.hw-headrow{font-family:var(--mono);font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.hw-row .r{text-align:right}
.hw-row .hw-name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.hw-rank{font-family:var(--mono);color:var(--faint)}
.hw-track{height:8px;background:var(--panel2);border-radius:4px;overflow:hidden}
.hw-track i{display:block;height:100%;background:var(--accent);border-radius:4px}
.hw-teamhead{display:flex;align-items:center;gap:14px;margin:.4rem 0 .8rem}
@media (max-width:640px){.hw-title{font-size:2rem}.hw-tag{display:none}
  .hw-row{grid-template-columns:1.6em 26px 1fr 4em 4em;padding:7px 10px}.hw-row .hw-track,.hw-row .hide-sm{display:none}
  .hw-big{font-size:2rem}.hw-match{padding:16px 10px 12px}}
"""


def style() -> None:
    st.html(f"<style>{CSS}</style>")


def brand() -> None:
    st.html('<div class="hw-brand"><div class="hw-mark"></div>'
            '<div class="hw-word">The Hardwood <span>Model</span></div>'
            '<div class="hw-tag">NBA odds from 10,000 simulations</div></div>')


def page_head(kicker: str, title: str, meta: str = "") -> None:
    st.html(f'<div class="hw-kicker">{kicker}</div><div class="hw-title">{title}</div>'
            + (f'<div class="hw-meta">{meta}</div>' if meta else ""))


def prob_bar(p_away: float) -> str:
    """Away share left, home share right; the favorite's side is highlighted."""
    a, h = ("on", "") if p_away >= 0.5 else ("", "on")
    return f'<div class="hw-bar"><i class="{a}" style="width:{p_away:.1%}"></i><i class="{h}" style="width:{1 - p_away:.1%}"></i></div>'


def game_card(g: dict) -> str:
    p_home = g["home_win_prob"]
    rows = ""
    for side, p in (("away", 1 - p_home), ("home", p_home)):
        abbr = g[side]
        rows += (f'<div class="hw-team{" hw-fav" if p >= 0.5 else ""}">{img(abbr)}'
                 f'<div class="hw-name">{escape(g[side + "_name"])}</div>'
                 f'<div class="hw-pts">{g[side + "_pts"]:.0f} pts</div><div class="hw-prob">{p:.0%}</div></div>')
    fav = g["home"] if p_home >= 0.5 else g["away"]
    where = escape(g["arena"]) + (" · neutral" if g["neutral"] else "")
    foot = (f'<span><b>{fav} by {abs(g["home_pts"] - g["away_pts"]):.1f}</b></span>'
            f'<span>Total <b>{g["total_mean"]:.0f}</b> ({g["total_pct"]["5"]:.0f}–{g["total_pct"]["95"]:.0f})</span>'
            f'<span>Rest {g["away"]} {g["rest_away"]}d · {g["home"]} {g["rest_home"]}d</span>')
    inj = [f'<div><span class="{"hw-out" if i["out"] else "hw-dtd"}">●</span> {escape(i["name"])} '
           f'({team}, {escape(i["status"])})</div>'
           for team, key in ((g["away"], "injuries_away"), (g["home"], "injuries_home")) for i in g[key]]
    inj_html = (f'<details class="hw-inj"><summary>Injury report ({len(inj)})</summary>{"".join(inj)}</details>'
                if inj else "")
    return (f'<div class="hw-card"><div class="hw-card-top"><span>{g["tip_et"]}</span><span>{where}</span></div>'
            f'{rows}{prob_bar(1 - p_home)}<div class="hw-foot">{foot}</div>{inj_html}</div>')


def games_grid(games: list[dict]) -> None:
    st.html(f'<div class="hw-grid">{"".join(game_card(g) for g in games)}</div>')


def matchup_panel(away: str, home: str, names: dict, p_home: float, pts_away: float, pts_home: float,
                  neutral: bool) -> None:
    sides = ""
    for abbr, p, pts in ((away, 1 - p_home, pts_away), (home, p_home, pts_home)):
        sides += (f'<div class="hw-side{" hw-fav" if p >= 0.5 else ""}">{img(abbr, 84)}'
                  f'<div class="hw-name">{escape(names[abbr])}</div><div class="hw-big">{p:.0%}</div>'
                  f'<div class="hw-pts">projected {pts:.0f}</div></div>')
        if abbr == away:
            sides += f'<div class="hw-vs">{"vs" if neutral else "@"}</div>'
    st.html(f'<div class="hw-match">{sides}{prob_bar(1 - p_home)}</div>')


def odds_list(rows: list[dict], label: str) -> None:
    """Ranked list with logos and a bar per team. rows: abbr, name, p (bar + %), p2 (finals), wins."""
    top = max(r["p"] for r in rows) or 1
    html = (f'<div class="hw-row hw-headrow"><span>#</span><span></span><span>Team</span><span class="hide-sm"></span>'
            f'<span class="r">{label}</span><span class="r hide-sm">Finals</span><span class="r">Wins</span></div>')
    for k, r in enumerate(rows, 1):
        html += (f'<div class="hw-row"><span class="hw-rank">{k}</span>{img(r["abbr"], 28)}'
                 f'<span class="hw-name" style="font-size:1.15rem">{escape(r["name"])}</span>'
                 f'<div class="hw-track"><i style="width:{r["p"] / top:.0%}"></i></div>'
                 f'<span class="r tnum" style="color:var(--accent);font-weight:600">{r["pct"]}</span>'
                 f'<span class="r tnum hide-sm">{r["pct2"]}</span><span class="r tnum">{r["wins"]}</span></div>')
    st.html(f'<div class="hw-list">{html}</div>')


def team_head(abbr: str, name: str, sub: str = "") -> None:
    st.html(f'<div class="hw-teamhead">{img(abbr, 56)}<div><div class="hw-title" style="font-size:2rem;margin:0">'
            f'{escape(name)}</div><div class="hw-meta" style="margin:0">{sub}</div></div></div>')
