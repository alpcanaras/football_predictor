#!/usr/bin/env python3
"""
Targeted trainer
================
Train a specific (league, engine) combination without touching the rest.

Reuses the production training primitives in :mod:`scripts.train` but
restricts ``config.MODEL_TYPES`` to the engine the user requested, so only
missing models are created. Existing model files for other engines/leagues
stay on disk untouched.

Typical use case: we have XGB models for all 25 leagues but LightGBM is
missing for Swiss (CH). Rebuilding the full ensemble is overkill.

    python scripts/train_one.py --league CH --engine lgbm

Flags mirror ``scripts/train.py`` where relevant.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import config
from scripts import data_loader
from scripts import train as trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train a specific league/engine combination')
    parser.add_argument('--league', required=True,
                        help='League key from config.LEAGUE_REGISTRY (e.g. CH)')
    parser.add_argument('--engine', required=True, choices=['xgb', 'lgbm'],
                        help='Model engine to train')
    parser.add_argument('--models', nargs='+',
                        choices=config.ALL_MARKETS + ['all'], default=['all'],
                        help='Markets to train (default: all applicable)')
    parser.add_argument('--n-trials', type=int, default=None,
                        help=f'Optuna trials per model (default {trainer.N_TRIALS})')
    parser.add_argument('--model-set', type=str, default='current')
    parser.add_argument('--no-time-decay', action='store_true')
    parser.add_argument('--decay-half-life-days', type=float, default=None)
    args = parser.parse_args()

    if args.league not in config.LEAGUE_REGISTRY:
        print(f"ERROR: '{args.league}' is not in LEAGUE_REGISTRY")
        sys.exit(1)

    print('=' * 60)
    print(f"  TARGETED TRAIN  league={args.league}  engine={args.engine}")
    print('=' * 60)

    print('\nLoading processed data...')
    try:
        all_data = data_loader.load_processed_data()
    except FileNotFoundError:
        print("ERROR: no processed data. Run 'python scripts/refresh_data.py' first.")
        sys.exit(1)

    n = len(all_data[all_data['league'] == args.league])
    print(f"  {n} rows for league '{args.league}'")
    if n < config.MIN_MATCHES_PER_LEAGUE:
        print(f"  ERROR: too few rows (<{config.MIN_MATCHES_PER_LEAGUE}).")
        sys.exit(1)

    models = args.models
    if 'all' in models:
        models = config.ALL_MARKETS

    decay = 0.0 if args.no_time_decay else args.decay_half_life_days

    # Temporarily restrict the engine list so only the requested engine runs.
    original_engines = list(config.MODEL_TYPES)
    config.MODEL_TYPES = [args.engine]
    try:
        t0 = time.time()
        trainer.train_all_models(
            all_data,
            leagues=[args.league],
            models_to_train=models,
            verbose=True,
            fresh=True,
            recent_days=None,
            time_decay_half_life_days=decay,
            model_set=args.model_set,
            n_trials=args.n_trials,
        )
        trainer.clear_checkpoint(model_set=args.model_set)
        elapsed = time.time() - t0
    finally:
        config.MODEL_TYPES = original_engines

    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print('\n' + '=' * 60)
    print(f"  DONE in {h}h {m}m {s}s")
    print('=' * 60)


if __name__ == '__main__':
    main()
