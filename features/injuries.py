"""Live injury report. Isolated so the source can be swapped: anything returning the same columns works.

Source: ESPN's public injuries JSON (statuses seen: "Out", "Day-To-Day").
"Out" → player treated as missing. "Day-To-Day" → listed in the app as questionable but assumed to play.
ponytail: questionable players aren't simulated as a coin flip; add a random availability branch if it matters.

Try:  python -m features.injuries
"""
import re
import unicodedata

import pandas as pd
import requests

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
OUT_STATUSES = {"Out", "Doubtful"}


def fetch() -> pd.DataFrame:
    """Columns: TEAM_NAME, PLAYER, STATUS, NOTE, RETURN_DATE."""
    data = requests.get(URL, timeout=30).json()
    rows = [{"TEAM_NAME": t["displayName"], "PLAYER": i["athlete"]["displayName"], "STATUS": i["status"],
             "NOTE": i.get("shortComment", ""), "RETURN_DATE": i.get("details", {}).get("returnDate")}
            for t in data.get("injuries", []) for i in t["injuries"]]
    return pd.DataFrame(rows, columns=["TEAM_NAME", "PLAYER", "STATUS", "NOTE", "RETURN_DATE"])


def norm(name: str) -> str:
    """'Nikola Jokić' / 'Jimmy Butler III' / 'P.J. Washington Jr.' → 'nikola jokic' / 'jimmy butler' / 'pj washington'."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return " ".join(s.split())


def attach(state: dict, report: pd.DataFrame) -> tuple[dict, list[str]]:
    """Map report rows onto team_state players. Returns {abbr: [{id, name, status, note}]} and unmatched names."""
    # Match teams by nickname: ESPN says "LA Clippers", nba_api says "Los Angeles Clippers"
    by_team_name = {t["name"].split()[-1]: abbr for abbr, t in state["teams"].items()}
    found, unmatched = {abbr: [] for abbr in state["teams"]}, []
    for r in report.itertuples():
        abbr = by_team_name.get(r.TEAM_NAME.split()[-1])
        match = abbr and next((p for p in state["teams"][abbr]["players"] if norm(p["name"]) == norm(r.PLAYER)), None)
        if not match:
            unmatched.append(f"{r.PLAYER} ({r.TEAM_NAME})")
            continue
        found[abbr].append({"id": match["id"], "name": match["name"], "status": r.STATUS, "note": r.NOTE,
                            "out": r.STATUS in OUT_STATUSES})
    return found, unmatched


if __name__ == "__main__":
    import json
    from pathlib import Path
    state = json.loads((Path(__file__).resolve().parent.parent / "data" / "team_state.json").read_text())
    rep = fetch()
    found, unmatched = attach(state, rep)
    print(rep["STATUS"].value_counts().to_string())
    print(f"matched {sum(map(len, found.values()))}/{len(rep)}; unmatched: {unmatched}")
    for abbr in ("NYK", "SAS", "BOS"):
        print(abbr, [(p["name"], p["status"]) for p in found[abbr]])
