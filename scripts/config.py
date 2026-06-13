"""
Football Predictor Configuration V2
====================================
Centralized configuration for training, prediction, and data processing.
"""

import os

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODELS_TIER1_DIR = os.path.join(MODELS_DIR, 'tier1')
MODELS_EXPERIMENTS_DIR = os.path.join(MODELS_DIR, 'experiments')
PROCESSED_DATA_FILE = os.path.join(PROJECT_ROOT, 'full_processed_data.csv')
ELO_PARAMS_FILE = os.path.join(MODELS_DIR, 'elo_params.json')

# =============================================================================
# LEAGUE REGISTRY
# =============================================================================
LEAGUE_REGISTRY = {
    'belgian':             {'type': 'rich',   'delimiter': ',', 'display_name': 'Belgian Pro League'},
    'british_champ':       {'type': 'rich',   'delimiter': ',', 'display_name': 'English Championship'},
    'british_conference':  {'type': 'rich',   'delimiter': ',', 'display_name': 'English Conference'},
    'british_league1':     {'type': 'rich',   'delimiter': ',', 'display_name': 'English League One'},
    'british_pl':          {'type': 'rich',   'delimiter': ',', 'display_name': 'English Premier League'},
    'dutch':               {'type': 'rich',   'delimiter': ',', 'display_name': 'Dutch Eredivisie'},
    'french':              {'type': 'rich',   'delimiter': ',', 'display_name': 'French Ligue 1'},
    'german':              {'type': 'rich',   'delimiter': ',', 'display_name': 'German Bundesliga'},
    'german_2':            {'type': 'rich',   'delimiter': ',', 'display_name': 'German 2. Bundesliga'},
    'greek':               {'type': 'rich',   'delimiter': ',', 'display_name': 'Greek Super League'},
    'italian':             {'type': 'rich',   'delimiter': ',', 'display_name': 'Italian Serie A'},
    'porto':               {'type': 'rich',   'delimiter': ',', 'display_name': 'Portuguese Liga'},
    'spanish':             {'type': 'rich',   'delimiter': ',', 'display_name': 'Spanish La Liga'},
    'spanish_2':           {'type': 'rich',   'delimiter': ',', 'display_name': 'Spanish Segunda'},
    'turkish':             {'type': 'rich',   'delimiter': ',', 'display_name': 'Turkish Super Lig'},
    'argentina':           {'type': 'sparse', 'delimiter': ',', 'display_name': 'Argentine Liga'},
    'BRAZIL':              {'type': 'sparse', 'delimiter': ',', 'display_name': 'Brazilian Serie A'},
    'CH':                  {'type': 'sparse', 'delimiter': ',', 'display_name': 'Swiss Super League'},
    'DANSK':               {'type': 'sparse', 'delimiter': ',', 'display_name': 'Danish Superliga'},
    'chn':                 {'type': 'sparse', 'delimiter': ',', 'display_name': 'Chinese Super League'},
    'fin':                 {'type': 'sparse', 'delimiter': ',', 'display_name': 'Finnish Veikkausliiga'},
    'japan':               {'type': 'sparse', 'delimiter': ',', 'display_name': 'Japanese J-League'},
    'norsk':               {'type': 'sparse', 'delimiter': ',', 'display_name': 'Norwegian Eliteserien'},
    'irish':               {'type': 'sparse', 'delimiter': ',', 'display_name': 'Irish Premier Division'},
    'mexico':              {'type': 'sparse', 'delimiter': ',', 'display_name': 'Mexican Liga MX'},
    'russian':             {'type': 'sparse', 'delimiter': ',', 'display_name': 'Russian Premier League'},
    'swedish':             {'type': 'sparse', 'delimiter': ',', 'display_name': 'Swedish Allsvenskan'},
    'usa':                 {'type': 'sparse', 'delimiter': ',', 'display_name': 'USA MLS'},
}

# =============================================================================
# FETCH SOURCES (football-data.co.uk download URLs)
# =============================================================================
# Rich European leagues: one file per season at mmz4281/<season>/<div>.csv.
# "Sparse new" leagues: single cumulative multi-season file at new/<code>.csv.
# {season} is a 4-digit code like '2526' (2025/26); computed at runtime in
# scripts/fetch_latest.py::current_season_code().
FETCH_SOURCES = {
    'belgian':             {'url': 'https://www.football-data.co.uk/mmz4281/{season}/B1.csv',  'target': 'B1-2.csv'},
    'british_champ':       {'url': 'https://www.football-data.co.uk/mmz4281/{season}/E1.csv',  'target': 'E1-3.csv'},
    'british_conference':  {'url': 'https://www.football-data.co.uk/mmz4281/{season}/EC.csv',  'target': 'EC-2.csv'},
    'british_league1':     {'url': 'https://www.football-data.co.uk/mmz4281/{season}/E2.csv',  'target': 'E2-2.csv'},
    'british_pl':          {'url': 'https://www.football-data.co.uk/mmz4281/{season}/E0.csv',  'target': 'E0-3.csv'},
    'dutch':               {'url': 'https://www.football-data.co.uk/mmz4281/{season}/N1.csv',  'target': 'N1-3.csv'},
    'french':              {'url': 'https://www.football-data.co.uk/mmz4281/{season}/F1.csv',  'target': 'F1-3.csv'},
    'german':              {'url': 'https://www.football-data.co.uk/mmz4281/{season}/D1.csv',  'target': 'D1-4.csv'},
    'german_2':            {'url': 'https://www.football-data.co.uk/mmz4281/{season}/D2.csv',  'target': 'D2-2.csv'},
    'greek':               {'url': 'https://www.football-data.co.uk/mmz4281/{season}/G1.csv',  'target': 'G1-2.csv'},
    'italian':             {'url': 'https://www.football-data.co.uk/mmz4281/{season}/I1.csv',  'target': 'I1-2.csv'},
    'porto':               {'url': 'https://www.football-data.co.uk/mmz4281/{season}/P1.csv',  'target': 'P1-2.csv'},
    'spanish':             {'url': 'https://www.football-data.co.uk/mmz4281/{season}/SP1.csv', 'target': 'SP1-2.csv'},
    'spanish_2':           {'url': 'https://www.football-data.co.uk/mmz4281/{season}/SP2.csv', 'target': 'SP2-2.csv'},
    'turkish':             {'url': 'https://www.football-data.co.uk/mmz4281/{season}/T1.csv',  'target': 'T1-2.csv'},

    'argentina':           {'url': 'https://www.football-data.co.uk/new/ARG.csv',     'target': 'ARG.csv'},
    'BRAZIL':              {'url': 'https://www.football-data.co.uk/new/BRA.csv',     'target': 'BRA.csv'},
    'CH':                  {'url': 'https://www.football-data.co.uk/new/SWZ.csv',     'target': 'SWZ-2.csv'},
    'DANSK':               {'url': 'https://www.football-data.co.uk/new/DNK.csv',     'target': 'DNK-2.csv'},
    'chn':                 {'url': 'https://www.football-data.co.uk/new/CHN.csv',     'target': 'CHN-2.csv'},
    'fin':                 {'url': 'https://www.football-data.co.uk/new/FIN.csv',     'target': 'FIN-2.csv'},
    'japan':               {'url': 'https://www.football-data.co.uk/new/JPN.csv',     'target': 'JPN-2.csv'},
    'norsk':               {'url': 'https://www.football-data.co.uk/new/NOR.csv',     'target': 'NOR-2.csv'},
    'irish':               {'url': 'https://www.football-data.co.uk/new/IRL.csv',     'target': 'IRL.csv'},
    'mexico':              {'url': 'https://www.football-data.co.uk/new/MEX.csv',     'target': 'MEX.csv'},
    'russian':             {'url': 'https://www.football-data.co.uk/new/RUS.csv',     'target': 'RUS.csv'},
    'swedish':             {'url': 'https://www.football-data.co.uk/new/SWE.csv',     'target': 'SWE-2.csv'},
    'usa':                 {'url': 'https://www.football-data.co.uk/new/USA.csv',     'target': 'USA.csv'},
}


# =============================================================================
# COLUMN MAPPING (sparse format -> standard)
# =============================================================================
COLUMN_MAPPING = {
    'Home': 'HomeTeam',
    'Away': 'AwayTeam',
    'HG': 'FTHG',
    'AG': 'FTAG',
    'Res': 'FTR',
}

REQUIRED_COLS = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Date']

RICH_COLS = ['HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST',
             'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']

# =============================================================================
# FEATURE COLUMNS
# =============================================================================
FEATURES_CORE = [
    # Elo
    'AttackElo_diff',
    'DefenseElo_diff',
    # Form & momentum (window 5)
    'HomeForm',
    'AwayForm',
    'FormDiff',
    'HomeMomentum',
    'AwayMomentum',
    # Rolling goals (window 10)
    'RollingHomeGoals',
    'RollingAwayGoals',
    'RollingHomeConceded',
    'RollingAwayConceded',
    'GoalDiff_Home',
    'GoalDiff_Away',
    # Short-window rolling (window 3)
    'RollingHomeGoals_3',
    'RollingAwayGoals_3',
    'RollingHomeConceded_3',
    'RollingAwayConceded_3',
    'GoalDiff_Home_3',
    'GoalDiff_Away_3',
    # H2H
    'H2H_GD',
    'H2H_HomeWinRate',
    'H2H_AvgTotalGoals',
    # Rest
    'HomeDaysRest',
    'AwayDaysRest',
    'RestAdvantage',
    # Scoring patterns (window 10)
    'HomeCleanSheetPct',
    'AwayCleanSheetPct',
    'HomeFailToScorePct',
    'AwayFailToScorePct',
    'HomeBTTSPct',
    'AwayBTTSPct',
    # League context
    'HomeLeaguePos',
    'AwayLeaguePos',
    'LeaguePosDiff',
    'HomePointsPerGame',
    'AwayPointsPerGame',
    # Venue-specific
    'HomeGoalsAtHome',
    'AwayGoalsAway',
    'HomeConcededAtHome',
    'AwayConcededAway',
    'HomeFormHome',
    'AwayFormAway',
    # Strength of schedule
    'HomeSOS',
    'AwaySOS',
    # Dynamic home advantage
    'LeagueHomeWinRate',
]

FEATURES_RICH = [
    'RollingHomeSoT',
    'RollingAwaySoT',
    'HTGoalRatio_Home',
    'HTGoalRatio_Away',
    'RollingHomeCorners',
    'RollingAwayCorners',
    'CornerDiff',
    'RollingHomeFouls',
    'RollingAwayFouls',
    'RollingHomeCards',
    'RollingAwayCards',
]


def get_features_for_league(league: str, **_kwargs) -> list:
    """Return the feature list for a given league."""
    is_rich = LEAGUE_REGISTRY.get(league, {}).get('type') == 'rich'
    features = FEATURES_CORE.copy()
    if is_rich:
        features.extend(FEATURES_RICH)
    return features


# =============================================================================
# ELO PARAMETERS (defaults; overridden by elo_tuner backtest)
# =============================================================================
ELO_WARMUP_YEAR = 2015
ELO_INITIAL = 1500
ELO_K_FACTOR = 20
ELO_SEASON_REGRESSION = 0.0
ELO_MARGIN_FACTOR = 0.0

ELO_BACKTEST_K = [10, 15, 20, 25, 30, 40]
ELO_BACKTEST_INITIAL = [1400, 1500, 1600]
ELO_BACKTEST_REGRESSION = [0.0, 0.1, 0.2, 0.33]
ELO_BACKTEST_MARGIN = [0.0, 0.1, 0.2, 0.3]

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
TRAIN_START_YEAR = 2015
TRAIN_CUTOFF = None
VALIDATION_DAYS = 60
MIN_MATCHES_PER_LEAGUE = 100
FORM_WINDOW = 5
SHORT_WINDOW = 3
ROLLING_WINDOW = 10
H2H_WINDOW = 6
LEAGUE_HOME_WIN_WINDOW = 150

TRAIN_RECENT_DAYS = None
TIME_DECAY_HALF_LIFE_DAYS = 500

# Time-series CV controls
CV_MAX_SPLITS = 5
CV_MIN_TEST_SIZE = 120

# Probability calibration
CALIBRATION_METHOD = 'auto'
CALIBRATION_HOLDOUT_FRACTION = 0.15
CALIBRATION_MIN_SAMPLES = 120
CALIBRATION_MAX_SAMPLES = 700
CALIBRATION_MIN_FIT_SAMPLES = 400
CALIBRATION_ISOTONIC_MIN_SAMPLES = 500

# =============================================================================
# PERFORMANCE (Apple Silicon M2)
# =============================================================================
_CPU_COUNT = os.cpu_count() or 8
XGB_N_JOBS = max(1, min(8, _CPU_COUNT - 2))
XGB_TREE_METHOD = 'hist'
XGB_MAX_BIN = 256

# =============================================================================
# OPTUNA HYPERPARAMETER SEARCH
# =============================================================================
OPTUNA_N_TRIALS = 200
OPTUNA_TIMEOUT = None
EARLY_STOPPING_ROUNDS = 50
MAX_ESTIMATORS = 3000

MODEL_TYPES = ['xgb', 'lgbm']

# Rich-only markets (only trained for rich leagues)
RICH_ONLY_MARKETS = {'ht1x2', 'htou05'}
ALL_MARKETS = ['1x2', 'ou25', 'ou15', 'ou35', 'btts', 'xg', 'ht1x2', 'htou05']

# =============================================================================
# ODDS EXTRACTION PRIORITY (used by data_loader to parse raw CSV files)
# =============================================================================
ODDS_PRIORITY_HDA = [
    ('AvgH', 'AvgD', 'AvgA'),
    ('B365H', 'B365D', 'B365A'),
    ('PSCH', 'PSCD', 'PSCA'),
    ('AvgCH', 'AvgCD', 'AvgCA'),
    ('B365CH', 'B365CD', 'B365CA'),
    ('MaxCH', 'MaxCD', 'MaxCA'),
]
ODDS_PRIORITY_OU = [
    ('Avg>2.5', 'Avg<2.5'),
    ('B365>2.5', 'B365<2.5'),
    ('P>2.5', 'P<2.5'),
    ('Max>2.5', 'Max<2.5'),
]
ODDS_PRIORITY_AH = [
    ('AHh', 'AvgAHH', 'AvgAHA'),
    ('AHh', 'B365AHH', 'B365AHA'),
    ('AHh', 'PAHH', 'PAHA'),
    ('AHh', 'MaxAHH', 'MaxAHA'),
]
ODDS_PRIORITY = ODDS_PRIORITY_HDA

# =============================================================================
# MODEL PATHS
# =============================================================================
def _sanitize_model_set(model_set: str | None) -> str:
    if not model_set:
        return 'current'
    clean = str(model_set).strip().replace('\\', '_').replace('/', '_')
    clean = clean.replace(' ', '_')
    return clean or 'current'


def get_tier_dir(tier: int = 1, model_set: str | None = 'current') -> str:
    model_set = _sanitize_model_set(model_set)
    if model_set == 'current':
        return MODELS_TIER1_DIR
    return os.path.join(MODELS_EXPERIMENTS_DIR, model_set, 'tier1')


def get_model_path(model_type: str, league: str, tier: int = 1,
                   model_set: str | None = 'current') -> str:
    tier_dir = get_tier_dir(tier=tier, model_set=model_set)
    filename = f"model_{model_type}_{league}.joblib"
    return os.path.join(tier_dir, filename)
