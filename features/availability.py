"""Player availability: how much value is missing from each team in each game.

Historical: a rotation player "missed" a game if they're on the team's recent roster but have no
row in that game's box score. Live (later): the same roster state, intersected with the injury report.

Value = player's average Game Score over their last 20 games, known BEFORE the game.
Roster = played for this team, this season, within the team's last 10 games, and hasn't since
played for another team. Players out 10+ team games drop off (Elo/rolling stats have adapted by then).

Inspect:  python -m features.availability DEN 2025-01-15    (team's next game on/after that date)
"""
import sys
from collections import deque
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "availability.parquet"
VALUE_GAMES, ROSTER_GAMES, MIN_MPG = 20, 10, 10.0


def game_score(p: pd.DataFrame) -> pd.Series:
    """Hollinger Game Score: one-number box-score value for a player-game."""
    return (p["PTS"] + 0.4 * p["FGM"] - 0.7 * p["FGA"] - 0.4 * (p["FTA"] - p["FTM"]) + 0.7 * p["OREB"]
            + 0.3 * p["DREB"] + p["STL"] + 0.7 * p["AST"] + 0.7 * p["BLK"] - 0.4 * p["PF"] - p["TOV"])


def compute(pl: pd.DataFrame) -> pd.DataFrame:
    """One row per (GAME_ID, TEAM_ID) with missing-value features. Expects player game log columns."""
    pl = pl.assign(GAME_DATE=pd.to_datetime(pl["GAME_DATE"]), GMSC=game_score(pl),
                   SEASON=pl["SEASON_ID"].astype(str).str[1:].astype(int))
    state: dict[int, dict] = {}  # player -> team, season, last team-game number, recent gmsc/min
    roster: dict[int, set] = {}  # team -> player ids whose latest team is this team
    team_n: dict[tuple, int] = {}  # (team, season) -> games played so far
    rows = []
    for _, day in pl.groupby("GAME_DATE", sort=True):
        # 1) score every game today using only state from earlier days
        for (gid, team, season), box in day.groupby(["GAME_ID", "TEAM_ID", "SEASON"]):
            n = team_n.get((team, season), 0)
            played = set(box["PLAYER_ID"])
            missing = [(pid, s) for pid in roster.get(team, ()) if pid not in played
                       for s in [state[pid]] if s["season"] == season and n - s["tgn"] <= ROSTER_GAMES
                       and sum(s["min"]) / len(s["min"]) >= MIN_MPG]
            values = sorted(((sum(s["gmsc"]) / len(s["gmsc"]), s["name"]) for _, s in missing), reverse=True)
            rows.append({"GAME_ID": gid, "TEAM_ID": team,
                         "MISSING_GMSC": sum(v for v, _ in values),
                         "MISSING_TOP": values[0][0] if values else 0.0,
                         "MISSING_N": len(values),
                         "MISSING_NAMES": ", ".join(f"{nm} ({v:.1f})" for v, nm in values)})
        # 2) then update state with today's box scores
        for (team, season), box in day.groupby(["TEAM_ID", "SEASON"]):
            n = team_n[(team, season)] = team_n.get((team, season), 0) + 1
            for r in box.itertuples():
                s = state.setdefault(r.PLAYER_ID, {"gmsc": deque(maxlen=VALUE_GAMES),
                                                   "min": deque(maxlen=ROSTER_GAMES), "team": None})
                if s["team"] != team:
                    roster.get(s["team"], set()).discard(r.PLAYER_ID)
                    roster.setdefault(team, set()).add(r.PLAYER_ID)
                s.update(team=team, season=season, tgn=n, name=r.PLAYER_NAME)
                s["gmsc"].append(r.GMSC)
                s["min"].append(r.MIN)
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    """Cached; rebuilt (~40s) whenever raw player logs are newer than the cache."""
    from pipeline.fetch_data import RAW, all_player_game_logs
    newest_raw = max(f.stat().st_mtime for f in RAW.glob("player_gamelog_*.parquet"))
    if not OUT.exists() or OUT.stat().st_mtime < newest_raw:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        compute(all_player_game_logs()).to_parquet(OUT, index=False)
    return pd.read_parquet(OUT)


if __name__ == "__main__":
    from pipeline.fetch_data import all_team_game_logs
    av = load()
    tg = all_team_game_logs()[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE", "MATCHUP", "WL"]]
    av = av.merge(tg, on=["GAME_ID", "TEAM_ID"])
    if len(sys.argv) == 3:
        team, date = sys.argv[1].upper(), sys.argv[2]
        hit = av[(av["TEAM_ABBREVIATION"] == team) & (av["GAME_DATE"] >= date)].sort_values("GAME_DATE").head(1)
        print(hit.drop(columns=["GAME_ID", "TEAM_ID"]).T.to_string() if len(hit) else "no game found")
    else:
        print(f"{len(av)} team-games. Biggest missing value:")
        print(av.nlargest(5, "MISSING_TOP")[["GAME_DATE", "MATCHUP", "WL", "MISSING_GMSC", "MISSING_NAMES"]]
              .to_string(index=False, max_colwidth=90))
