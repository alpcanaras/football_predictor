"""
Batch-rename data CSVs to year-bearing names
=============================================
Renames every file under data/<league>/ to a consistent, human-readable
scheme based on the dates INSIDE each file:

    <league>_<startYear>-<endYear>.csv     (multi-season / cumulative)
    <league>_<year>.csv                     (single calendar year)

e.g.  data/british_pl/E0-4.csv      -> data/british_pl/british_pl_2025-2026.csv
      data/BRAZIL/BRA.csv           -> data/BRAZIL/BRAZIL_2015-2026.csv
      data/british_pl/pl2020-2021.csv -> data/british_pl/british_pl_2020-2021.csv

Safe by design:
  * dry-run is the DEFAULT — nothing is touched until you pass --apply
  * the loader globs *.csv, so renaming never changes what gets loaded
  * a manifest is written so --undo restores the original names
  * idempotent: re-run any time (e.g. after a fetch) to re-tidy new files
  * collisions (two files -> same name) are reported and skipped, not merged

    python scripts/rename_data.py            # preview
    python scripts/rename_data.py --apply     # do it
    python scripts/rename_data.py --undo      # revert last apply
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts import config

SKIP_DIRS = {'_incoming', '_fixtures', 'global'}
MANIFEST = os.path.join(config.DATA_DIR, '.rename_manifest.json')


def _year_range(path: str) -> tuple[int, int] | None:
    """Min/max calendar year of the Date column inside a CSV."""
    try:
        d = pd.read_csv(path, encoding='latin1', on_bad_lines='skip',
                        usecols=lambda c: c in ('Date', 'date'))
    except Exception:
        try:
            d = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
        except Exception:
            return None
    col = 'Date' if 'Date' in d.columns else ('date' if 'date' in d.columns else None)
    if col is None:
        return None
    dt = pd.to_datetime(d[col], dayfirst=True, errors='coerce').dropna()
    if dt.empty:
        return None
    return int(dt.dt.year.min()), int(dt.dt.year.max())


def _target_name(league: str, yr: tuple[int, int]) -> str:
    lo, hi = yr
    return f"{league}_{lo}.csv" if lo == hi else f"{league}_{lo}-{hi}.csv"


def plan() -> list[dict]:
    """Build the list of intended renames across all league folders."""
    actions = []
    for league in sorted(os.listdir(config.DATA_DIR)):
        folder = os.path.join(config.DATA_DIR, league)
        if not os.path.isdir(folder) or league in SKIP_DIRS or league.startswith('.'):
            continue
        taken = set()
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.csv'):
                continue
            src = os.path.join(folder, fname)
            yr = _year_range(src)
            if yr is None:
                actions.append({'src': src, 'dst': None, 'note': 'no parseable dates'})
                continue
            new = _target_name(league, yr)
            dst = os.path.join(folder, new)
            if new == fname:
                continue                      # already tidy
            if new in taken or (os.path.exists(dst) and os.path.abspath(dst) != os.path.abspath(src)):
                actions.append({'src': src, 'dst': dst, 'note': 'COLLISION - skipped'})
                continue
            taken.add(new)
            actions.append({'src': src, 'dst': dst, 'note': 'ok'})
    return actions


def main():
    ap = argparse.ArgumentParser(description='Rename data CSVs to year names')
    ap.add_argument('--apply', action='store_true', help='Actually rename')
    ap.add_argument('--undo', action='store_true', help='Revert last --apply')
    args = ap.parse_args()

    if args.undo:
        if not os.path.exists(MANIFEST):
            print("No manifest to undo.")
            return
        with open(MANIFEST, encoding='utf-8') as f:
            done = json.load(f)
        n = 0
        for rec in reversed(done):
            if os.path.exists(rec['dst']) and not os.path.exists(rec['src']):
                os.rename(rec['dst'], rec['src'])
                n += 1
        os.remove(MANIFEST)
        print(f"Reverted {n} files.")
        return

    actions = plan()
    ok = [a for a in actions if a['note'] == 'ok']
    bad = [a for a in actions if a['note'] != 'ok']

    print(f"\n  {'PLAN' if not args.apply else 'APPLYING'}  "
          f"({len(ok)} renames, {len(bad)} skipped)\n")
    for a in actions:
        rel_s = os.path.relpath(a['src'], config.DATA_DIR)
        rel_d = os.path.relpath(a['dst'], config.DATA_DIR) if a['dst'] else '-'
        flag = '' if a['note'] == 'ok' else f"   [{a['note']}]"
        print(f"  {rel_s:<34} ->  {os.path.basename(rel_d):<28}{flag}")

    if not args.apply:
        print("\n  Dry run. Re-run with --apply to rename, --undo to revert later.")
        return

    done = []
    for a in ok:
        os.rename(a['src'], a['dst'])
        done.append({'src': a['src'], 'dst': a['dst']})
    with open(MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(done, f, indent=2)
    print(f"\n  Renamed {len(done)} files. Manifest -> {MANIFEST}")
    print("  (loader is unaffected; run with --undo to revert.)")


if __name__ == '__main__':
    main()
