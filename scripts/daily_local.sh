#!/bin/zsh
# Daily update run by launchd on this Mac (stats.nba.com blocks GitHub's servers), via caffeinate so a
# scheduled wake doesn't fall back asleep mid-run. Refreshes predictions + season odds and pushes them so the
# Streamlit site updates. stats.nba.com and the post-wake network are flaky, so the whole update retries.
cd "$(dirname "$0")/.."  # real path (~/Projects/NBA Model): launchd may not read ~/Desktop
echo "=== $(date) ==="

# Right after a wake, Wi-Fi/DNS can take a minute ("Could not resolve host").
for i in {1..30}; do /usr/bin/curl -sI --max-time 5 https://github.com >/dev/null && break; sleep 10; done

update() {
  local failed=0
  /usr/bin/git pull --rebase --autostash -q origin main || failed=1
  venv/bin/python -m pipeline.predict_today || failed=1
  venv/bin/python -m pipeline.simulate_season || failed=1
  # Push whatever succeeded (e.g. today's games even if the season sim failed)
  /usr/bin/git add data/predictions.json data/team_state.json data/track_record.json data/history data/raw \
    data/season_outlook.json data/season_outlook_history.csv
  /usr/bin/git diff --cached --quiet || /usr/bin/git commit -q -m "Daily predictions $(date +%F)"
  GIT_TERMINAL_PROMPT=0 /usr/bin/git push -q origin main || failed=1
  return $failed
}

for attempt in 1 2 3 4; do
  if update; then echo "done"; exit 0; fi
  [[ $attempt -lt 4 ]] && { echo "attempt $attempt failed, retrying in 10 min ($(date +%H:%M))"; sleep 600; }
done
echo "FAILED after 4 attempts ($(date +%H:%M))"
exit 1
