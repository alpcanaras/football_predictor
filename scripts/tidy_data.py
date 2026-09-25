#!/usr/bin/env python3
"""
Tidy the data folder into one naming convention
===============================================
Files arrived by several routes over time — hand-named seasons
(``d20-21.csv``), browser duplicate downloads (``G1 (2).csv``, ``E2-7.csv``)
and the fetcher's own output — so the same season often sits on disk two or
three times under different names. The loader de-duplicates, so this costs
correctness nothing, but it makes the folder impossible to reason about and
hides which seasons you actually have.

One convention, matching what fetch_latest now writes:

    rich leagues    <DIV>_<season>.csv     D1_2526.csv, E0_2627.csv
    sparse leagues  <CODE>.csv             SWE.csv, ARG.csv

Anything else is retired to ``data/_retired/<league>/`` — moved, never
deleted, and only after proving every match it contains also exists in a
canonical file. A manifest makes it reversible.

    python scripts/tidy_data.py              # dry run (default)
    python scripts/tidy_data.py --apply
    python scripts/tidy_data.py --undo
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts import config

RETIRED = os.path.join(config.DATA_DIR, '_retired')
MANIFEST = os.path.join(RETIRED, 'manifest.json')
CANON_RICH = re.compile(r'^[A-Z0-9]+_\d{4}\.csv$')


def match_keys(path: str):
    """(date, home, away) for every row, or None if unreadable."""
    d = None
    # Mirror the loader: these files are a mix of utf-8-sig and latin-1, and a
    # stricter reader here would silently mark a file "unverifiable" and keep
    # a duplicate forever.
    for enc in ('utf-8-sig', 'latin1'):
        try:
            d = pd.read_csv(path, encoding=enc, low_memory=False,
                            on_bad_lines='skip')
            break
        except Exception:
            continue
    if d is None:
        return None
    cols = {c.lower(): c for c in d.columns}
    dt = cols.get('date')
    h = cols.get('hometeam') or cols.get('home')
    a = cols.get('awayteam') or cols.get('away')
    if not (dt and h and a):
        return None
    dd = pd.to_datetime(d[dt], dayfirst=True, errors='coerce')
    return {(str(x.date()), str(y).strip(), str(z).strip())
            for x, y, z in zip(dd, d[h], d[a]) if pd.notna(x)}


def plan():
    """Return (renames, retires, blocked) without touching anything."""
    renames, retires, blocked = [], [], []
    for lg, info in config.LEAGUE_REGISTRY.items():
        src = config.FETCH_SOURCES.get(lg)
        if not src:
            continue
        code = os.path.basename(src['url']).replace('.csv', '')
        folder = os.path.join(config.DATA_DIR, lg)
        files = sorted(glob.glob(os.path.join(folder, '*.csv')))
        if not files:
            continue

        if info['type'] != 'rich':
            # one cumulative file per league; normalise its name
            want = os.path.join(folder, f'{code}.csv')
            for f in files:
                if f != want:
                    if os.path.exists(want):
                        blocked.append((f, want, 'target exists'))
                    else:
                        renames.append((f, want))
            continue

        canon = [f for f in files if CANON_RICH.match(os.path.basename(f))]
        others = [f for f in files if not CANON_RICH.match(os.path.basename(f))]
        if not others:
            continue
        covered = set()
        for f in canon:
            k = match_keys(f)
            if k:
                covered |= k
        for f in others:
            k = match_keys(f)
            if k is None:
                blocked.append((f, '', 'unreadable'))
                continue
            missing = k - covered
            if missing:
                # holds matches nothing else has — keep it, flag it loudly
                blocked.append((f, '', f'{len(missing)} unique matches'))
            else:
                retires.append(f)
    return renames, retires, blocked


def do_apply(renames, retires):
    manifest = {'renames': [], 'retires': []}
    for src, dst in renames:
        os.rename(src, dst)
        manifest['renames'].append([src, dst])
    for f in retires:
        lg = os.path.basename(os.path.dirname(f))
        dest_dir = os.path.join(RETIRED, lg)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(f))
        shutil.move(f, dest)
        manifest['retires'].append([f, dest])
    os.makedirs(RETIRED, exist_ok=True)
    with open(MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def do_undo():
    if not os.path.isfile(MANIFEST):
        sys.exit('No manifest — nothing to undo.')
    with open(MANIFEST, encoding='utf-8') as fh:
        m = json.load(fh)
    n = 0
    for src, dst in m.get('retires', []):
        if os.path.isfile(dst):
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.move(dst, src)
            n += 1
    for src, dst in reversed(m.get('renames', [])):
        if os.path.isfile(dst):
            os.rename(dst, src)
            n += 1
    os.remove(MANIFEST)
    print(f"  restored {n} file(s)")


def main():
    ap = argparse.ArgumentParser(description='Tidy data/ into one convention')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--undo', action='store_true')
    args = ap.parse_args()

    if args.undo:
        do_undo()
        return

    renames, retires, blocked = plan()
    print(f"\n  RENAME {len(renames)} (sparse files to <CODE>.csv)")
    for s, d in renames:
        print(f"    {os.path.relpath(s):<44} -> {os.path.basename(d)}")
    print(f"\n  RETIRE {len(retires)} redundant file(s) -> data/_retired/")
    for f in retires[:8]:
        print(f"    {os.path.relpath(f)}")
    if len(retires) > 8:
        print(f"    … and {len(retires) - 8} more")
    if blocked:
        print(f"\n  KEPT {len(blocked)} (not safe to touch)")
        for f, d, why in blocked:
            print(f"    {os.path.relpath(f) if f else d}  — {why}")

    if not args.apply:
        print("\n  Dry run. Re-run with --apply to make the changes.")
        return
    m = do_apply(renames, retires)
    print(f"\n  Applied: {len(m['renames'])} renamed, {len(m['retires'])} retired.")
    print("  Undo with: python scripts/tidy_data.py --undo")
    print("  Then rebuild: python scripts/refresh_data.py")


if __name__ == '__main__':
    main()
