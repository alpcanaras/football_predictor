#!/bin/bash
# Unattended daily refresh: fetch, rebuild features, then VERIFY.
#
# The verification half matters more than the fetch. This job runs with nobody
# watching, and the failures that hurt are the silent ones — a provider serving
# the wrong league's file, or a season rollover overwriting the previous
# season. Both have happened. A fetch that "succeeds" while quietly destroying
# data looks identical to a good one unless something checks.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=./venv/bin/python
LOG=logs/daily.log
mkdir -p logs
exec >>"$LOG" 2>&1

echo "=================================================================="
echo "  daily refresh  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

$PY scripts/fetch_latest.py --apply --refresh-processed
fetch_rc=$?
echo "  fetch exit=$fetch_rc  (3 = some leagues not published, which is normal)"

# Keep the club-Elo snapshots current for the European/UCL model.
$PY scripts/clubs_europe.py update >/dev/null 2>&1 \
    && echo "  clubelo snapshot ok" || echo "  clubelo refresh FAILED"

# Internationals refresh quietly; they matter far less week to week.
$PY scripts/international.py update >/dev/null 2>&1 \
    && echo "  internationals ok" || echo "  internationals refresh FAILED"

echo "  --- health check ---"
$PY scripts/selftest.py --quick
test_rc=$?
if [ $test_rc -ne 0 ]; then
    echo "  *** SELF TEST FAILED — something is broken, look above ***"
fi
echo "  done $(date '+%H:%M:%S'), selftest exit=$test_rc"
exit $test_rc
