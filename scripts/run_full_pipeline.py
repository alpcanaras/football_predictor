#!/usr/bin/env python3
"""
Full Training Pipeline
======================
Orchestrates the complete V2 training process:

  1. Elo parameter backtesting (finds optimal K, initial, regression, margin)
  2. Data reload with tuned Elo params + new features
  3. Walk-forward validation (60 days holdout)
  4. Production training (all data, XGB + LightGBM, Optuna-tuned)
  5. Save processed data for prediction

Designed to run unattended for 3-4 days on an M2 MacBook Pro.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import config
from scripts import data_loader
from scripts import train as trainer
from scripts.elo_tuner import backtest_elo_params, load_league_data


def main():
    total_start = time.time()

    print("=" * 70)
    print("  KILLER MODEL PIPELINE V2")
    print("=" * 70)

    # ------------------------------------------------------------------
    # PHASE 1: Elo parameter backtesting
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 1: Elo Parameter Backtesting")
    print("=" * 70)

    t0 = time.time()
    raw_by_league = {}
    for league_name in config.LEAGUE_REGISTRY:
        loaded = load_league_data(league_name, verbose=False)
        if not loaded.empty:
            raw_by_league[league_name] = loaded
    print(f"  Loaded {len(raw_by_league)} leagues for Elo backtest.")

    best_elo = backtest_elo_params(raw_by_league, verbose=True)
    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"\n  Phase 1 complete in {m}m {s}s")
    print(f"  Best Elo params: K={best_elo.get('k_factor')}, "
          f"init={best_elo.get('initial_rating')}, "
          f"regression={best_elo.get('season_regression')}, "
          f"margin={best_elo.get('margin_factor')}")

    # ------------------------------------------------------------------
    # PHASE 2: Reload data with tuned Elo + all V2 features
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 2: Feature Engineering (tuned Elo + V2 features)")
    print("=" * 70)

    t0 = time.time()
    all_data = data_loader.load_and_process_all_leagues(verbose=True)
    n = len(all_data)
    d0, d1 = all_data['Date'].min().date(), all_data['Date'].max().date()
    nl = all_data['league'].nunique()
    print(f"\n  {n} matches | {nl} leagues | {d0} to {d1}")

    all_data.to_csv(config.PROCESSED_DATA_FILE, index=False)
    print(f"  Saved -> {config.PROCESSED_DATA_FILE}")

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"  Phase 2 complete in {m}m {s}s")

    # ------------------------------------------------------------------
    # PHASE 3: Walk-forward validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 3: Walk-Forward Validation")
    print("=" * 70)

    t0 = time.time()
    trainer.validate_models(
        all_data,
        validation_days=config.VALIDATION_DAYS,
        n_trials=max(config.OPTUNA_N_TRIALS // 4, 20),
    )
    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n  Phase 3 complete in {h}h {m}m {s}s")

    # ------------------------------------------------------------------
    # PHASE 4: Production training (all data)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 4: Production Training (XGB + LightGBM, Optuna-tuned)")
    print("=" * 70)

    t0 = time.time()
    trainer.train_all_models(
        all_data,
        verbose=True,
        fresh=True,
        n_trials=config.OPTUNA_N_TRIALS,
    )
    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n  Phase 4 complete in {h}h {m}m {s}s")

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------
    total_elapsed = time.time() - total_start
    h, rem = divmod(int(total_elapsed), 3600)
    m, s = divmod(rem, 60)
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE  ({h}h {m}m {s}s total)")
    print("=" * 70)
    print(f"\n  Models saved in: {config.MODELS_TIER1_DIR}")
    print(f"  Processed data:  {config.PROCESSED_DATA_FILE}")
    print(f"\n  To predict: python scripts/predict.py --home <Team> --away <Team>")


if __name__ == '__main__':
    main()
