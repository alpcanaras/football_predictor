#!/usr/bin/env python3
"""
Prediction Script V2
====================
Ensemble predictions from XGBoost + LightGBM models.
"""

import argparse
import warnings
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy.stats import poisson

from scripts import config
from scripts import data_loader
from scripts import utils

warnings.filterwarnings('ignore')


# =============================================================================
# ENSEMBLE HELPERS
# =============================================================================
def _load_ensemble(model_type, league, model_set='current'):
    """Load all available engine models for this type/league."""
    models = []
    for eng in config.MODEL_TYPES:
        m = utils.load_model(model_type, league, model_set=model_set, engine=eng)
        if m is not None:
            models.append(m)
    if not models:
        m = utils.load_model(model_type, league, model_set=model_set, engine='xgb')
        if m is not None:
            models.append(m)
    return models


def _ensemble_proba(models, X):
    """Average predict_proba across models."""
    probas = [m.predict_proba(X) for m in models]
    return np.mean(probas, axis=0)


def _ensemble_predict(models, X):
    """Average regression predictions across models."""
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)


# =============================================================================
# CORE PREDICTION
# =============================================================================
def predict_match(home_team, away_team, team_stats, team_to_league,
                  historical_data, include_xg=True, model_set='current',
                  prediction_date=None, **_kwargs):
    if home_team not in team_stats.index:
        raise ValueError(f"Unknown team: {home_team}")
    if away_team not in team_stats.index:
        raise ValueError(f"Unknown team: {away_team}")

    league = team_to_league.get(home_team)
    if not league:
        raise ValueError(f"Could not determine league for {home_team}")

    home_st = team_stats.loc[home_team]
    away_st = team_stats.loc[away_team]
    h2h = utils.compute_h2h_stats(historical_data, home_team, away_team)

    if prediction_date is None:
        prediction_date = pd.Timestamp.now()

    feat = utils.create_fixture_features(
        home_st, away_st, h2h, league,
        prediction_date=prediction_date,
        historical_data=historical_data,
    )

    predictions = {
        'home_team': home_team,
        'away_team': away_team,
        'league': league,
    }
    predictions.update(_predict_all_markets(feat, league, include_xg, model_set))
    _market_anchor(predictions, league, home_team, away_team)
    return predictions


def _market_anchor(predictions: dict, league: str,
                   home_team: str, away_team: str) -> None:
    """Blend the 1X2 probabilities with bookmaker odds when the match is in
    the upcoming-fixtures feed. Weights come from models/blend_weights.json
    (fitted out-of-sample by scripts/blend.py)."""
    if '1x2' not in predictions:
        return
    try:
        from scripts import fixtures as fixtures_mod
        odds = fixtures_mod.lookup_odds(
            fixtures_mod.load_cached(), league, home_team, away_team)
    except Exception:
        return
    if not odds:
        return

    try:
        with open(os.path.join(config.MODELS_DIR, 'blend_weights.json'),
                  encoding='utf-8') as f:
            w = json.load(f)['with_odds']
        w_model, w_book = float(w['model']), float(w['book'])
    except Exception:
        w_model, w_book = 0.0, 1.11

    inv = np.array([1.0 / odds['OddsH'], 1.0 / odds['OddsD'],
                    1.0 / odds['OddsA']])
    book = inv / inv.sum()
    p = predictions['1x2']
    model = np.array([p['home'], p['draw'], p['away']])

    log_p = (w_model * np.log(np.clip(model, 1e-9, 1.0))
             + w_book * np.log(book))
    blended = np.exp(log_p - log_p.max())
    blended /= blended.sum()

    predictions['1x2_model'] = dict(p)
    predictions['1x2'] = {'home': float(blended[0]), 'draw': float(blended[1]),
                          'away': float(blended[2])}
    predictions['market'] = {
        'odds': {'home': odds['OddsH'], 'draw': odds['OddsD'],
                 'away': odds['OddsA']},
        'implied': {'home': float(book[0]), 'draw': float(book[1]),
                    'away': float(book[2])},
    }
    if 'OddsOver25' in odds:
        predictions['market']['ou25_odds'] = {
            'over': odds['OddsOver25'], 'under': odds['OddsUnder25']}
        if 'ou25' in predictions:
            try:
                with open(os.path.join(config.MODELS_DIR,
                                       'blend_weights.json'),
                          encoding='utf-8') as f:
                    w_ou = json.load(f).get('ou25_with_odds')
            except Exception:
                w_ou = None
            if w_ou:
                inv = np.array([1.0 / odds['OddsUnder25'],
                                1.0 / odds['OddsOver25']])
                book_ou = inv / inv.sum()
                pm = predictions['ou25']
                model_ou = np.array([pm['under'], pm['over']])
                log_p = (float(w_ou['model'])
                         * np.log(np.clip(model_ou, 1e-9, 1.0))
                         + float(w_ou['book']) * np.log(book_ou))
                b = np.exp(log_p - log_p.max())
                b /= b.sum()
                predictions['ou25_model'] = dict(pm)
                predictions['ou25'] = {'under': float(b[0]),
                                       'over': float(b[1])}


def _predict_all_markets(fixture_df, league, include_xg, model_set='current'):
    result = {}
    is_rich = config.LEAGUE_REGISTRY.get(league, {}).get('type') == 'rich'
    features = config.get_features_for_league(league)
    X = fixture_df[features]

    ms = _load_ensemble('1x2', league, model_set)
    if ms:
        try:
            p = _ensemble_proba(ms, X)[0]
            result['1x2'] = {'home': float(p[2]), 'draw': float(p[1]),
                             'away': float(p[0])}
        except Exception:
            pass

    ms = _load_ensemble('ou25', league, model_set)
    if ms:
        try:
            p = _ensemble_proba(ms, X)[0]
            result['ou25'] = {'over': float(p[1]), 'under': float(p[0])}
        except Exception:
            pass

    ms = _load_ensemble('ou15', league, model_set)
    if ms:
        try:
            p = _ensemble_proba(ms, X)[0]
            result['ou15'] = {'over': float(p[1]), 'under': float(p[0])}
        except Exception:
            pass

    ms = _load_ensemble('ou35', league, model_set)
    if ms:
        try:
            p = _ensemble_proba(ms, X)[0]
            result['ou35'] = {'over': float(p[1]), 'under': float(p[0])}
        except Exception:
            pass

    ms = _load_ensemble('btts', league, model_set)
    if ms:
        try:
            p = _ensemble_proba(ms, X)[0]
            result['btts'] = {'yes': float(p[1]), 'no': float(p[0])}
        except Exception:
            pass

    if is_rich:
        ms = _load_ensemble('ht1x2', league, model_set)
        if ms:
            try:
                p = _ensemble_proba(ms, X)[0]
                result['ht1x2'] = {'home': float(p[2]), 'draw': float(p[1]),
                                   'away': float(p[0])}
            except Exception:
                pass

        ms = _load_ensemble('htou05', league, model_set)
        if ms:
            try:
                p = _ensemble_proba(ms, X)[0]
                result['htou05'] = {'over': float(p[1]), 'under': float(p[0])}
            except Exception:
                pass

    if include_xg:
        mh = _load_ensemble('xGH', league, model_set)
        ma = _load_ensemble('xGA', league, model_set)
        if mh and ma:
            try:
                hxg = max(float(_ensemble_predict(mh, X)[0]), 0.05)
                axg = max(float(_ensemble_predict(ma, X)[0]), 0.05)
                result['xg'] = {'home': hxg, 'away': axg}

                max_goals = 7
                score_probs = {}
                for i in range(max_goals):
                    for j in range(max_goals):
                        score_probs[(i, j)] = (poisson.pmf(i, hxg) *
                                               poisson.pmf(j, axg))

                scores = [(f"{i}-{j}", float(p))
                          for (i, j), p in score_probs.items()]
                result['top_scores'] = sorted(scores, key=lambda x: x[1],
                                              reverse=True)[:6]

                over25 = sum(p for (i, j), p in score_probs.items()
                             if i + j > 2.5)
                btts_yes = sum(p for (i, j), p in score_probs.items()
                               if i > 0 and j > 0)
                result['poisson'] = {
                    'over25': float(over25),
                    'under25': float(1 - over25),
                    'btts_yes': float(btts_yes),
                    'btts_no': float(1 - btts_yes),
                }
            except Exception:
                pass

    return result


# =============================================================================
# INTERACTIVE MODE
# =============================================================================
def interactive_mode(team_stats, team_to_league, all_teams, historical_data,
                     model_set='current'):
    print("\n" + "=" * 60)
    print("FOOTBALL PREDICTOR - INTERACTIVE MODE")
    print("=" * 60)
    print("Type 'quit' to exit, 'list' to see teams.\n")

    while True:
        print("-" * 40)
        inp = input("Home team: ").strip()
        if inp.lower() in ('quit', 'exit', 'q'):
            break
        if inp.lower() == 'list':
            for t in all_teams:
                print(f"  {t}")
            continue
        home = _fuzzy(inp, all_teams)
        if not home:
            continue
        away = _fuzzy(input("Away team: ").strip(), all_teams)
        if not away:
            continue
        try:
            pred = predict_match(home, away, team_stats, team_to_league,
                                 historical_data, model_set=model_set)
            print("\n" + utils.format_prediction_table(pred))
        except Exception as e:
            print(f"  ERROR: {e}")


def _fuzzy(query, all_teams):
    matches = [t for t in all_teams if query.lower() in t.lower()]
    if not matches:
        print(f"  No match for '{query}'")
        return None
    if len(matches) == 1:
        return matches[0]
    print("  Multiple matches:")
    for i, m in enumerate(matches[:10]):
        print(f"    {i + 1}. {m}")
    try:
        return matches[int(input("  Number: ").strip()) - 1]
    except Exception:
        print("  Invalid")
        return None


# =============================================================================
# BATCH MODE
# =============================================================================
def batch_predict(input_file, output_file, team_stats, team_to_league,
                  historical_data, model_set='current'):
    fixtures = pd.read_csv(input_file)
    rows = []
    for _, r in fixtures.iterrows():
        home = r.get('HomeTeam', r.get('home_team', r.get('Home')))
        away = r.get('AwayTeam', r.get('away_team', r.get('Away')))
        try:
            p = predict_match(home, away, team_stats, team_to_league,
                              historical_data, model_set=model_set)
            out = {'HomeTeam': home, 'AwayTeam': away, 'League': p['league'],
                   'Home': p['1x2']['home'], 'Draw': p['1x2']['draw'],
                   'Away': p['1x2']['away']}
            if 'ou25' in p:
                out['Over25'] = p['ou25']['over']
            if 'btts' in p:
                out['BTTS_Yes'] = p['btts']['yes']
            if 'xg' in p:
                out['xG_Home'] = p['xg']['home']
                out['xG_Away'] = p['xg']['away']
            rows.append(out)
        except Exception as e:
            rows.append({'HomeTeam': home, 'AwayTeam': away, 'Error': str(e)})
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Football match predictions')
    parser.add_argument('--home', type=str)
    parser.add_argument('--away', type=str)
    parser.add_argument('--batch', type=str)
    parser.add_argument('--output', type=str, default='predictions.csv')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--no-xg', action='store_true')
    parser.add_argument('--model-set', type=str, default='current')
    args = parser.parse_args()

    print("Loading data...")
    try:
        hist = data_loader.load_processed_data()
    except FileNotFoundError:
        print("ERROR: Run 'python scripts/train.py' first.")
        sys.exit(1)

    team_stats = utils.get_team_stats_table(hist)
    team_to_league = utils.get_team_to_league_map(hist)
    all_teams = utils.get_all_teams(hist)
    print(f"Loaded {len(all_teams)} teams from "
          f"{len(utils.get_available_leagues(model_set=args.model_set))} leagues")

    if args.batch:
        batch_predict(args.batch, args.output, team_stats, team_to_league, hist,
                      model_set=args.model_set)
        return

    if args.home and args.away:
        try:
            pred = predict_match(args.home, args.away, team_stats,
                                 team_to_league, hist,
                                 include_xg=not args.no_xg,
                                 model_set=args.model_set)
            if args.json:
                def conv(o):
                    if isinstance(o, (np.floating, np.integer)):
                        return float(o)
                    return o
                print(json.dumps(pred, default=conv, indent=2))
            else:
                print(utils.format_prediction_table(pred))
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        return

    interactive_mode(team_stats, team_to_league, all_teams, hist,
                     model_set=args.model_set)


if __name__ == '__main__':
    main()
