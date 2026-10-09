#!/bin/zsh
# Daily update run by launchd on this Mac (stats.nba.com blocks GitHub's servers).
# Refreshes predictions + season odds, then pushes them so the Streamlit site updates.
set -e
cd "$HOME/Desktop/NBA Model"
echo "=== $(date) ==="
/usr/bin/git pull --rebase --autostash -q origin main
venv/bin/python -m pipeline.predict_today
venv/bin/python -m pipeline.simulate_season
/usr/bin/git add data/predictions.json data/team_state.json data/track_record.json data/history data/raw \
  data/season_outlook.json data/season_outlook_history.csv
/usr/bin/git diff --cached --quiet || /usr/bin/git commit -q -m "Daily predictions $(date +%F)"
GIT_TERMINAL_PROMPT=0 /usr/bin/git push -q origin main
echo "done"
