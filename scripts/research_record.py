"""Archive pre-result challenger forecasts and grade by exact fixture identity.

python -m scripts.research_record record --bundle reports/research/RUN/bundle.joblib --refresh
python -m scripts.research_record grade --record reports/research_records/RUN

Only fixtures strictly after today's UTC date are recorded. This avoids
guessing the source feed's kickoff timezone. Download time is not quote time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd

from scripts import config, data_loader, fixtures
from scripts.research_features import FeatureState, attach_odds, load_raw, replay, season_id
from scripts.research_train import predict_bundle, write_json


def read_fixtures():
    frames = []
    for name in fixtures.SOURCES:
        path = Path(fixtures.FIXTURES_DIR, name+'.csv')
        df = pd.read_csv(path, encoding='utf-8-sig', sep=None, engine='python')
        if name == 'rich':
            df['league'] = df.Div.map(fixtures.DIV_TO_LEAGUE)
        else:
            df['league'] = df.Country.map(fixtures.COUNTRY_TO_LEAGUE)
        df = data_loader.normalize_columns(df)
        df['Date'] = pd.to_datetime(df.Date, dayfirst=True, format='mixed', errors='coerce').dt.normalize()
        df = df.dropna(subset=['Date', 'league', 'HomeTeam', 'AwayTeam'])
        df['SeasonId'] = [season_id(lg, date) for lg, date in zip(df.league, df.Date)]
        df['feed'] = name
        frames.append(attach_odds(df))
    return pd.concat(frames, ignore_index=True).drop_duplicates(['league', 'Date', 'HomeTeam', 'AwayTeam'])


def forecast(bundle, raw, upcoming, day):
    day = pd.Timestamp(day).normalize()
    if pd.Timestamp(bundle['cutoff']) > day:
        raise ValueError('Model contains outcomes after the forecast decision date')
    selected = upcoming[(upcoming.Date > day) & upcoming.league.isin(bundle['leagues'])].copy()
    if selected.empty:
        return []
    history = raw[(raw.Date < day) & raw.league.isin(bundle['leagues'])]
    state = FeatureState(bundle['elo_k'], bundle['season_retention'])
    if len(history):
        _, state = replay(history, state)
    features = pd.DataFrame([state.fixture(row) for row in selected.to_dict('records')], index=selected.index)
    frame = pd.concat([selected, features], axis=1)
    probs = predict_bundle(bundle, frame)
    records = []
    for i, row in enumerate(frame.to_dict('records')):
        record = {k: row[k] for k in ['league', 'HomeTeam', 'AwayTeam', 'feed', 'odds_source']}
        record['Date'] = str(row['Date'].date())
        record['probabilities'] = {name: p[i].tolist() for name, p in probs.items() if np.isfinite(p[i]).all()}
        record['bet365_snapshot_ADH'] = [float(row[c]) if pd.notna(row[c]) else None for c in ['betA', 'betD', 'betH']]
        record['state_last_completed_day'] = str(state.last_day.date()) if state.last_day is not None else None
        records.append(record)
    return records


def record(args):
    bundle_path = Path(args.bundle).resolve()
    bundle = joblib.load(bundle_path)
    if args.refresh:
        fixtures.fetch(force=True)
    now = datetime.now(timezone.utc)
    day = pd.Timestamp(now.date())
    run = Path(config.PROJECT_ROOT, 'reports', 'research_records', now.strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    sources = []
    for name in fixtures.SOURCES:
        path = Path(fixtures.FIXTURES_DIR, name+'.csv')
        payload = path.read_bytes()
        (run/path.name).write_bytes(payload)
        sources.append(dict(feed=name, sha256=hashlib.sha256(payload).hexdigest(),
                            downloaded_at_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                            source_observed_at=None))
    upcoming = read_fixtures()
    raw, raw_sources = load_raw(bundle['leagues'])
    forecasts = forecast(bundle, raw, upcoming, day)
    payload = dict(recorded_at_utc=now.isoformat(), decision_date_utc=str(day.date()),
        bundle_path=str(bundle_path), bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        model_cutoff=bundle['cutoff'], source_files=raw_sources, fixture_feeds=sources,
        forecast_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        class_order=['away', 'draw', 'home'], fixtures=forecasts,
        skipped_current_or_past=int((upcoming.Date <= day).sum()),
        skipped_future_untrained_league=int(((upcoming.Date > day) & ~upcoming.league.isin(bundle['leagues'])).sum()),
        warning='Source quote times are unknown. Download timestamps do not prove executable prices. Forecasts only; no bets placed.')
    write_json(run/'forecasts.json', payload)
    print(f'Archived {len(forecasts)} future-fixture forecasts: {run}')


def grade(args):
    folder = Path(args.record)
    archive = json.loads((folder/'forecasts.json').read_text())
    leagues = sorted({f['league'] for f in archive['fixtures']})
    if not leagues:
        print('No forecasts in this archive.')
        return
    raw, sources = load_raw(leagues)
    lookup = {(r.league, str(r.Date.date()), r.HomeTeam, r.AwayTeam): r.FTR
              for r in raw.itertuples()}
    scores = []
    for f in archive['fixtures']:
        actual = lookup.get((f['league'], f['Date'], f['HomeTeam'], f['AwayTeam']))
        row = dict(league=f['league'], Date=f['Date'], HomeTeam=f['HomeTeam'], AwayTeam=f['AwayTeam'], result=actual)
        row['log_loss'] = ({name: float(-np.log(max(p[{'A': 0, 'D': 1, 'H': 2}[actual]], 1e-12)))
                           for name, p in f['probabilities'].items()} if actual else {})
        scores.append(row)
    result = dict(graded_at_utc=datetime.now(timezone.utc).isoformat(), source_files=sources,
                  completed=sum(r['result'] is not None for r in scores), fixtures=scores)
    # Grading revisions are appended as new files; original forecasts stay fixed.
    dest = folder/f'grade-{uuid.uuid4().hex[:12]}.json'
    write_json(dest, result)
    print(f'{result["completed"]}/{len(scores)} settled: {dest}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('record')
    p.add_argument('--bundle', required=True)
    p.add_argument('--refresh', action='store_true')
    p = sub.add_parser('grade')
    p.add_argument('--record', required=True)
    args = parser.parse_args()
    {'record': record, 'grade': grade}[args.command](args)


if __name__ == '__main__':
    main()
