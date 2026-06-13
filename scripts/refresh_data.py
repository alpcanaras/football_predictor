#!/usr/bin/env python3
"""
Refresh Processed Data
======================
Rebuilds ``full_processed_data.csv`` from whatever is currently under
``data/<league_key>/*.csv`` using the production data_loader pipeline
(Elo + all features). Does NOT retrain any models.

Typical weekly workflow:

    1. Drop newly downloaded CSVs into ``data/<league_key>/``
       (the folder name must match a key in ``config.LEAGUE_REGISTRY``).
    2. Run:  python scripts/refresh_data.py
    3. Restart the notebook kernel (so it reloads the CSV).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import config
from scripts import data_loader


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("  REFRESH PROCESSED DATA  (no retraining)")
    print("=" * 60)
    print(f"  Reading raw CSVs from: {config.DATA_DIR}")

    all_data = data_loader.load_and_process_all_leagues(verbose=True)

    if all_data.empty:
        print("\n  ERROR: no data loaded. Check your data/ folder names match "
              "keys in config.LEAGUE_REGISTRY.")
        sys.exit(1)

    n = len(all_data)
    d0, d1 = all_data['Date'].min().date(), all_data['Date'].max().date()
    nl = all_data['league'].nunique()

    all_data.to_csv(config.PROCESSED_DATA_FILE, index=False)

    elapsed = int(time.time() - t0)
    m, s = divmod(elapsed, 60)
    print("\n" + "=" * 60)
    print(f"  Saved -> {config.PROCESSED_DATA_FILE}")
    print(f"  {n} matches | {nl} leagues | {d0} to {d1}")
    print(f"  Done in {m}m {s}s")
    print("=" * 60)
    print("\n  Next: restart the Predictor_latest.ipynb kernel to load new data.")


if __name__ == '__main__':
    main()
