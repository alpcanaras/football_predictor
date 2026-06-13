"""
Data Loading and Processing
============================
Load CSV data for all leagues, normalise columns, extract odds,
and run the full feature-engineering pipeline.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from . import config
from . import features


def _load_elo_params() -> dict:
    """Load tuned Elo parameters, falling back to config defaults."""
    path = config.ELO_PARAMS_FILE
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


# =============================================================================
# COLUMN HELPERS
# =============================================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename sparse-format columns to standard names & fix known typos."""
    df = df.rename(columns=config.COLUMN_MAPPING)
    if 'B36CA' in df.columns and 'B365CA' not in df.columns:
        df = df.rename(columns={'B36CA': 'B365CA'})
    return df


def extract_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the best available HDA, O/U, and AH odds into uniform columns."""
    df = df.copy()

    # --- HDA odds ---
    df['OddsH'] = np.nan
    df['OddsD'] = np.nan
    df['OddsA'] = np.nan
    for h_col, d_col, a_col in config.ODDS_PRIORITY_HDA:
        if not all(c in df.columns for c in [h_col, d_col, a_col]):
            continue
        mask = df['OddsH'].isna()
        if not mask.any():
            break
        df.loc[mask, 'OddsH'] = pd.to_numeric(df.loc[mask, h_col], errors='coerce')
        df.loc[mask, 'OddsD'] = pd.to_numeric(df.loc[mask, d_col], errors='coerce')
        df.loc[mask, 'OddsA'] = pd.to_numeric(df.loc[mask, a_col], errors='coerce')

    # --- O/U 2.5 odds ---
    df['OddsOver25'] = np.nan
    df['OddsUnder25'] = np.nan
    for o_col, u_col in config.ODDS_PRIORITY_OU:
        if not all(c in df.columns for c in [o_col, u_col]):
            continue
        mask = df['OddsOver25'].isna()
        if not mask.any():
            break
        df.loc[mask, 'OddsOver25'] = pd.to_numeric(df.loc[mask, o_col], errors='coerce')
        df.loc[mask, 'OddsUnder25'] = pd.to_numeric(df.loc[mask, u_col], errors='coerce')

    # --- Asian Handicap odds ---
    df['OddsAHh'] = np.nan
    df['OddsAHH'] = np.nan
    df['OddsAHA'] = np.nan
    for line_col, h_col, a_col in config.ODDS_PRIORITY_AH:
        if not all(c in df.columns for c in [line_col, h_col, a_col]):
            continue
        mask = df['OddsAHH'].isna()
        if not mask.any():
            break
        df.loc[mask, 'OddsAHh'] = pd.to_numeric(df.loc[mask, line_col], errors='coerce')
        df.loc[mask, 'OddsAHH'] = pd.to_numeric(df.loc[mask, h_col], errors='coerce')
        df.loc[mask, 'OddsAHA'] = pd.to_numeric(df.loc[mask, a_col], errors='coerce')

    return df


# =============================================================================
# SINGLE-LEAGUE LOADER
# =============================================================================
def load_league_data(league_name: str, verbose: bool = False) -> pd.DataFrame:
    """Load all CSV files for a league, concat, dedup, and sort."""
    league_info = config.LEAGUE_REGISTRY.get(league_name)
    if not league_info:
        return pd.DataFrame()

    folder = os.path.join(config.DATA_DIR, league_name)
    if not os.path.isdir(folder):
        if verbose:
            print(f"    Folder not found: {folder}")
        return pd.DataFrame()

    csv_files = glob.glob(os.path.join(folder, '*.csv'))
    if not csv_files:
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(
                f,
                delimiter=league_info['delimiter'],
                on_bad_lines='skip',
                encoding='latin1',
            )
            dfs.append(df)
        except Exception as e:
            if verbose:
                print(f"    ERROR loading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = normalize_columns(combined)

    if not all(c in combined.columns for c in config.REQUIRED_COLS):
        if verbose:
            print(f"    {league_name}: missing required columns")
        return pd.DataFrame()

    # Parse types
    combined['Date'] = pd.to_datetime(combined['Date'], dayfirst=True, errors='coerce')
    combined['FTHG'] = pd.to_numeric(combined['FTHG'], errors='coerce')
    combined['FTAG'] = pd.to_numeric(combined['FTAG'], errors='coerce')
    combined = combined.dropna(subset=['Date', 'FTHG', 'FTAG', 'FTR'])
    combined['FTHG'] = combined['FTHG'].astype(int)
    combined['FTAG'] = combined['FTAG'].astype(int)

    # Sort + dedup
    combined = combined.sort_values('Date').reset_index(drop=True)
    combined = combined.drop_duplicates(
        subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'],
        keep='first',
    ).reset_index(drop=True)

    # Coerce rich columns to numeric where they exist
    for col in config.RICH_COLS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')

    # Extract odds
    combined = extract_odds(combined)

    combined['league'] = league_name
    return combined


# =============================================================================
# FULL PIPELINE
# =============================================================================
def load_and_process_all_leagues(verbose: bool = True) -> pd.DataFrame:
    """Load every registered league, compute Elo + features, return combined DF."""
    all_dfs = []

    for league_name, league_info in config.LEAGUE_REGISTRY.items():
        if verbose:
            print(f"  Processing {league_info['display_name']}...")

        raw = load_league_data(league_name, verbose=verbose)
        if raw.empty:
            if verbose:
                print(f"    Skipped (no valid data)")
            continue

        # Keep all data from ELO_WARMUP_YEAR for Elo history
        raw = raw[raw['Date'].dt.year >= config.ELO_WARMUP_YEAR].copy()
        raw = raw.reset_index(drop=True)
        if raw.empty:
            continue

        elo_p = _load_elo_params()
        raw = features.calculate_elo_for_league(
            raw,
            initial_elo=elo_p.get('initial_rating', config.ELO_INITIAL),
            k=elo_p.get('k_factor', config.ELO_K_FACTOR),
            season_regression=elo_p.get('season_regression', config.ELO_SEASON_REGRESSION),
            margin_factor=elo_p.get('margin_factor', config.ELO_MARGIN_FACTOR),
        )

        # All features
        is_rich = league_info['type'] == 'rich'
        raw = features.create_features(raw, is_rich=is_rich)

        if verbose:
            n = len(raw)
            d0 = raw['Date'].min().date()
            d1 = raw['Date'].max().date()
            print(f"    {n} matches ({d0} to {d1})")

        all_dfs.append(raw)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('Date').reset_index(drop=True)
    return combined


# =============================================================================
# CONVENIENCE
# =============================================================================
def load_processed_data(filepath: str = None) -> pd.DataFrame:
    """Load already-processed data from the saved CSV."""
    if filepath is None:
        filepath = config.PROCESSED_DATA_FILE
    return pd.read_csv(filepath, parse_dates=['Date'])


def filter_by_year(df: pd.DataFrame, min_year: int = None) -> pd.DataFrame:
    if min_year is None:
        min_year = config.TRAIN_START_YEAR
    return df[df['Date'].dt.year >= min_year].copy()
