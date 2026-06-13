#!/usr/bin/env python3
"""
Daily football-data.co.uk fetcher
=================================
Downloads the current-season (or cumulative) CSV for every league in
``config.FETCH_SOURCES`` and reports on what changed.

Default (no flags) is **staging-only**: files land in
``data/_incoming/<YYYY-MM-DD>/<league>/<target>`` and ``data/<league>/...``
is left untouched. Inspect the diagnostic table, then re-run with
``--apply`` to promote the latest staged batch into the league folders.

Workflow::

    # 1. Download to staging, see diffs
    python scripts/fetch_latest.py

    # 2. Happy with the staged files? Promote them:
    python scripts/fetch_latest.py --apply --refresh-processed

    # 3. Once confident, activate the daily LaunchAgent:
    #    launchctl load -w ~/Library/LaunchAgents/com.footballpredictor.fetch.plist

Flags:
    --apply               Promote the latest staged batch into data/<league>/
    --refresh-processed   After --apply, rebuild full_processed_data.csv
    --only LEAGUE[,LEAGUE...]   Subset (matches LEAGUE_REGISTRY keys)
    --dry-run             Plan only; print URLs and targets, don't download
    --timeout SECONDS     Per-file HTTP timeout (default 30)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import shutil
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # type: ignore

from scripts import config


STAGING_ROOT = os.path.join(config.DATA_DIR, '_incoming')
LAST_FETCH_FILE = os.path.join(config.DATA_DIR, '_incoming', 'last_fetch.json')


# -----------------------------------------------------------------------------
# URL season handling
# -----------------------------------------------------------------------------
def current_season_code(today: Optional[dt.date] = None) -> str:
    """Return YYyy season code used by football-data.co.uk.

    Their season convention: 2025/26 -> '2526'. Season flips in August.
    Months Jan-Jul belong to the *previous* year's season, Aug-Dec to the
    current year's.
    """
    today = today or dt.date.today()
    if today.month >= 8:
        start = today.year % 100
    else:
        start = (today.year - 1) % 100
    end = (start + 1) % 100
    return f'{start:02d}{end:02d}'


def build_url(template: str, season: str) -> str:
    return template.replace('{season}', season)


# -----------------------------------------------------------------------------
# Fetch + stage
# -----------------------------------------------------------------------------
def _stage_dir(today: Optional[dt.date] = None) -> str:
    today = today or dt.date.today()
    return os.path.join(STAGING_ROOT, today.isoformat())


def _csv_stats(path: str) -> dict:
    """Return simple stats: row count and last-match date (if parseable)."""
    try:
        with open(path, encoding='latin1', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
    except Exception:
        return {'rows': 0, 'last_date': None}
    if not header:
        return {'rows': 0, 'last_date': None}
    # Prefer a 'Date' column; fall back to index 3 (sparse-format position).
    try:
        di = header.index('Date')
    except ValueError:
        di = 3 if len(header) > 3 else None
    last = None
    if di is not None:
        for r in reversed(rows):
            if len(r) > di and r[di].strip():
                last = r[di].strip()
                break
    return {'rows': len(rows), 'last_date': last}


def _download(url: str, dest: str, timeout: int) -> tuple[bool, str, int]:
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={'User-Agent': 'football-predictor/1.0'})
    except Exception as e:
        return False, f'{type(e).__name__}: {e}', 0
    if resp.status_code != 200:
        return False, f'HTTP {resp.status_code}', resp.status_code
    body = resp.content
    if len(body) < 200 or b'HomeTeam' not in body and b'Home,Away' not in body and b'Home' not in body.split(b'\n', 1)[0]:
        return False, 'payload-not-csv', resp.status_code
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as f:
        f.write(body)
    return True, 'OK', resp.status_code


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def _fmt_size(n: int) -> str:
    if n < 1024:
        return f'{n}B'
    if n < 1024 * 1024:
        return f'{n / 1024:.1f}K'
    return f'{n / 1024 / 1024:.1f}M'


def _report(results: list[dict]) -> None:
    hdr = f"  {'League':<22} {'Status':<9} {'Size':>8} {'Rows':>7} {'Last date':>12}  {'Prev size':>10}"
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for r in results:
        prev = _fmt_size(r['prev_size']) if r['prev_size'] else '-'
        print(
            f"  {r['league']:<22} {r['status']:<9} "
            f"{_fmt_size(r['size']):>8} {r['rows']:>7} "
            f"{(r['last_date'] or '-'):>12}  {prev:>10}"
        )


# -----------------------------------------------------------------------------
# Apply staged files
# -----------------------------------------------------------------------------
def _promote(league: str, staged_path: str) -> str:
    """Copy staged file into data/<league>/<target>, overwriting."""
    src_cfg = config.FETCH_SOURCES[league]
    target_dir = os.path.join(config.DATA_DIR, league)
    target_path = os.path.join(target_dir, src_cfg['target'])
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(staged_path, target_path)
    return target_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description='Fetch latest football-data.co.uk CSVs')
    parser.add_argument('--apply', action='store_true',
                        help='Promote staged files into data/<league>/')
    parser.add_argument('--refresh-processed', action='store_true',
                        help='After --apply, rebuild full_processed_data.csv')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated league keys to fetch (subset)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Plan only; do not download or write')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Per-file HTTP timeout in seconds (default 30)')
    parser.add_argument('--season', type=str, default=None,
                        help='Override season code (default: current)')
    args = parser.parse_args()

    season = args.season or current_season_code()
    today = dt.date.today()
    stage = _stage_dir(today)

    keys = list(config.FETCH_SOURCES.keys())
    if args.only:
        wanted = {k.strip() for k in args.only.split(',')}
        missing = wanted - set(keys)
        if missing:
            print(f"ERROR: unknown league keys: {sorted(missing)}")
            return 1
        keys = [k for k in keys if k in wanted]

    print('=' * 60)
    print(f"  FETCH  {today.isoformat()}  season={season}  "
          f"{len(keys)} leagues  dry_run={args.dry_run}  apply={args.apply}")
    print('=' * 60)

    results = []
    for lg in keys:
        src = config.FETCH_SOURCES[lg]
        url = build_url(src['url'], season)
        staged_path = os.path.join(stage, lg, src['target'])

        # Existing file size in data/<league>/<target> for the diff column.
        existing = os.path.join(config.DATA_DIR, lg, src['target'])
        prev_size = os.path.getsize(existing) if os.path.isfile(existing) else 0

        if args.dry_run:
            results.append({
                'league': lg, 'status': 'DRY',
                'size': 0, 'rows': 0, 'last_date': None,
                'prev_size': prev_size, 'url': url, 'staged': staged_path,
                'http': 0,
            })
            continue

        ok, status, http = _download(url, staged_path, args.timeout)
        size = os.path.getsize(staged_path) if os.path.isfile(staged_path) else 0
        stats = _csv_stats(staged_path) if ok else {'rows': 0, 'last_date': None}
        results.append({
            'league': lg, 'status': 'OK' if ok else status,
            'size': size, 'rows': stats['rows'], 'last_date': stats['last_date'],
            'prev_size': prev_size, 'url': url, 'staged': staged_path,
            'http': http,
        })

    _report(results)

    # Write last_fetch sidecar (success/failure summary).
    if not args.dry_run:
        os.makedirs(STAGING_ROOT, exist_ok=True)
        summary = {
            'fetched_at': dt.datetime.now().isoformat(timespec='seconds'),
            'season': season,
            'stage_dir': stage,
            'results': [
                {k: v for k, v in r.items() if k != 'staged'}
                for r in results
            ],
        }
        with open(LAST_FETCH_FILE, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    if args.apply and not args.dry_run:
        promoted = []
        for r in results:
            if r['status'] != 'OK':
                continue
            tgt = _promote(r['league'], r['staged'])
            promoted.append((r['league'], tgt))
        print()
        print(f"  Promoted {len(promoted)} files into data/<league>/")
        for lg, tgt in promoted:
            print(f"    {lg} -> {tgt}")

        if args.refresh_processed:
            print()
            print('  Rebuilding full_processed_data.csv ...')
            from scripts import data_loader
            all_data = data_loader.load_and_process_all_leagues(verbose=False)
            if all_data.empty:
                print('  ERROR: refresh produced empty dataframe; NOT writing.')
                return 2
            all_data.to_csv(config.PROCESSED_DATA_FILE, index=False)
            print(f"  Saved -> {config.PROCESSED_DATA_FILE} "
                  f"({len(all_data)} matches, {all_data['league'].nunique()} leagues, "
                  f"through {all_data['Date'].max().date()})")

    failed = [r for r in results if r['status'] not in ('OK', 'DRY')]
    if failed:
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
