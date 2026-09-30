"""
Football Predictor — web app
=============================
A point-and-click interface. No notebooks, no cells.

    pip install streamlit          # one-time
    streamlit run app.py           # opens in your browser

Four tabs:
  * Toto           — two persistent coupons (Turkish + German), system optimiser
  * Fixtures       — every upcoming match with models, anchored to live odds
  * Match          — pick a league + two teams, get the full prediction card
  * Internationals — upcoming national-team matches

Any match in the other tabs can be pushed into either coupon. Coupons are saved
to data/_toto/<game>.txt (override with FOOTBALL_PREDICTOR_TOTO_DIR), so they
survive reloads and new browser tabs until you clear them.
"""

import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st

from scripts import config, data_loader, utils
from scripts import predict as predict_mod
from scripts import toto

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")

# Saved coupons live here. Overridable so tests (and any throwaway session)
# can point at a scratch directory instead of clobbering real coupons.
COUPON_DIR = os.environ.get(
    'FOOTBALL_PREDICTOR_TOTO_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '_toto'))


# ----------------------------------------------------------------------------
# Cached loaders (run once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and models…")
def load_everything():
    hist = data_loader.load_processed_data()
    team_stats = utils.get_team_stats_table(hist)
    team_to_league = utils.get_team_to_league_map(hist)
    leagues = sorted(
        {team_to_league[t] for t in team_to_league},
        key=lambda l: config.LEAGUE_REGISTRY.get(l, {}).get('display_name', l))
    teams_by_league = {}
    for t, lg in team_to_league.items():
        teams_by_league.setdefault(lg, []).append(t)
    for lg in teams_by_league:
        teams_by_league[lg].sort()
    return hist, team_stats, team_to_league, leagues, teams_by_league


def disp(lg):
    return config.LEAGUE_REGISTRY.get(lg, {}).get('display_name', lg)


# ----------------------------------------------------------------------------
# Persistent coupons (disk-backed, one file per game)
# ----------------------------------------------------------------------------
def _coupon_path(game):
    return os.path.join(COUPON_DIR, f"{game}.txt")


def _load_coupon_file(game):
    try:
        with open(_coupon_path(game), encoding='utf-8') as f:
            return f.read()
    except OSError:
        return ''


def _save_coupon_file(game, text):
    """Atomic write: a coupon is a week's work, so never leave it truncated
    if the process dies mid-write."""
    try:
        os.makedirs(COUPON_DIR, exist_ok=True)
        tmp = _coupon_path(game) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(text or '')
        os.replace(tmp, _coupon_path(game))
    except OSError:
        pass


def _fmt_line(home, away, odds=None):
    line = f"{home} - {away}"
    if odds and all(odds.get(k) for k in ('home', 'draw', 'away')):
        line += f"  {odds['home']:.2f} {odds['draw']:.2f} {odds['away']:.2f}"
    return line


def _pair(home, away):
    """Order-insensitive, accent-folded match identity (for duplicate checks)."""
    return frozenset((toto._fold(home), toto._fold(away)))


def _coupon_pairs(text):
    return {_pair(r['home'], r['away'])
            for _, r in toto.parse_lines(text).iterrows()}


def _append_line(game, home, away, odds=None):
    """Append a match to a game's coupon (skipping duplicates). Disk is the
    source of truth, so this is safe even if Streamlit dropped the inactive
    coupon's widget state. Returns 'added' or 'dup'."""
    cur = _load_coupon_file(game).rstrip()
    if _pair(home, away) in _coupon_pairs(cur):
        return 'dup'
    new = (cur + '\n' + _fmt_line(home, away, odds)) if cur \
        else _fmt_line(home, away, odds)
    _save_coupon_file(game, new)
    st.session_state[f'coupon_{game}'] = new
    return 'added'


def _dedup_coupon(game):
    """on_click callback: drop repeated matches (keep first occurrence)."""
    key = f'coupon_{game}'
    seen, out = set(), []
    for raw in st.session_state.get(key, '').splitlines():
        df = toto.parse_lines(raw)
        if df.empty:
            continue
        r = df.iloc[0]
        p = _pair(r['home'], r['away'])
        if p in seen:
            continue
        seen.add(p)
        out.append(raw)
    new = '\n'.join(out)
    st.session_state[key] = new
    _save_coupon_file(game, new)


def _quick_add():
    """on_click callback for the Toto tab's quick-add row."""
    game = st.session_state.get('toto_game') or list(toto.GAMES)[0]
    h = st.session_state.get('qa_home')
    a = st.session_state.get('qa_away')
    if not h or not a or h == a:
        st.session_state['qa_msg'] = ('warning', 'Pick two different teams.')
        return
    odds = None
    parts = (st.session_state.get('qa_odds') or '').replace(',', '.').split()
    if len(parts) == 3:
        try:
            v = [float(x) for x in parts]
            odds = {'home': v[0], 'draw': v[1], 'away': v[2]}
        except ValueError:
            pass
    status = _append_line(game, h, a, odds)
    st.session_state['qa_home'] = None
    st.session_state['qa_away'] = None
    st.session_state['qa_odds'] = ''
    if status == 'added':
        st.session_state['qa_msg'] = (
            'success', f'Added **{h} - {a}** to the {game.capitalize()} coupon.')
    else:
        st.session_state['qa_msg'] = (
            'info', f'**{h} - {a}** is already in the {game.capitalize()} coupon.')


def _parse_slash_odds(s):
    if not s:
        return None
    try:
        h, d, a = [float(x) for x in str(s).split('/')]
        return {'home': h, 'draw': d, 'away': a}
    except Exception:
        return None


def _clear_game(game):
    """Clear the coupon, keeping what was there for a one-click undo — a
    misclick must not cost the week's coupon."""
    old = st.session_state.get(f'coupon_{game}', '') or _load_coupon_file(game)
    if old.strip():
        st.session_state[f'undo_{game}'] = old
    st.session_state[f'coupon_{game}'] = ''
    st.session_state.pop(f'toto_res_{game}', None)
    _save_coupon_file(game, '')


def _undo_clear(game):
    old = st.session_state.pop(f'undo_{game}', '')
    if old:
        st.session_state[f'coupon_{game}'] = old
        _save_coupon_file(game, old)


def _fill_odds(game):
    """on_click: look every odds-less coupon row up in the live feed and
    append its 1X2 odds. Typing odds by hand is the slowest part of building a
    coupon, and where odds exist they dominate the model anyway."""
    from scripts import fixtures as fx_mod
    ck = f'coupon_{game}'
    try:
        fx_mod.fetch()
        feed = fx_mod.load(fetch_if_missing=False)
    except Exception as e:
        st.session_state[f'odds_msg_{game}'] = ('error', f'Feed unavailable: {e}')
        return

    filled = missing = 0
    out = []
    for line in st.session_state.get(ck, '').splitlines():
        parsed_line = toto.parse_lines(line)
        if parsed_line.empty:
            out.append(line)
            continue
        r = parsed_line.iloc[0]
        has = all(pd.notna(r.get(c)) for c in ('o1', 'ox', 'o2'))
        if has or not str(r['away']).strip():
            out.append(line)
            continue
        odds = fx_mod.find_odds_any_league(feed, r['home'], r['away'])
        if odds:
            out.append(f"{line.rstrip()}  {odds['OddsH']:.2f} "
                       f"{odds['OddsD']:.2f} {odds['OddsA']:.2f}")
            filled += 1
        else:
            out.append(line)
            missing += 1
    st.session_state[ck] = '\n'.join(out)
    _save_coupon_file(game, st.session_state[ck])
    st.session_state.pop(f'toto_res_{game}', None)
    if filled:
        st.session_state[f'odds_msg_{game}'] = (
            'success', f"Filled odds for {filled} match(es)."
            + (f" {missing} not in the feed (kick-off too far off, or a league "
               "the feed does not carry)." if missing else ''))
    else:
        st.session_state[f'odds_msg_{game}'] = (
            'info', "No odds found for these matches — the feed only carries "
            "fixtures in the next few days.")


def _fix_name(game, old, key):
    """on_click: replace an unrecognised name in the coupon with the chosen one."""
    new = st.session_state.get(key)
    if not new or new == old:
        return
    ck = f'coupon_{game}'
    lines = []
    for line in st.session_state.get(ck, '').splitlines():
        # only swap the name as a whole side of the fixture, never mid-word
        parts = toto.parse_lines(line)
        if not parts.empty:
            r = parts.iloc[0]
            if str(r['home']).strip() == old:
                line = line.replace(old, new, 1)
            elif str(r['away']).strip() == old:
                idx = line.rfind(old)
                if idx >= 0:
                    line = line[:idx] + new + line[idx + len(old):]
        lines.append(line)
    st.session_state[ck] = '\n'.join(lines)
    _save_coupon_file(game, st.session_state[ck])
    st.session_state.pop(f'toto_res_{game}', None)


def _do_add(ns, by_label):
    """on_click callback: append the multiselect's picks to the chosen game."""
    sel = st.session_state.get(f'{ns}_sel', [])
    game = st.session_state.get(f'{ns}_game') or list(toto.GAMES)[0]
    n = dups = 0
    for lab in sel:
        r = by_label.get(lab)
        if not r:
            continue
        if _append_line(game, r['home'], r['away'], r.get('odds')) == 'added':
            n += 1
        else:
            dups += 1
    st.session_state[f'{ns}_sel'] = []          # clear picks (safe in callback)
    if n or dups:
        st.session_state[f'{ns}_added'] = (n, dups, game)


def _add_controls(rows, ns):
    """Multiselect + game picker + Add button for Fixtures / World Cup slates."""
    if not rows:
        return
    by_label = {r['_label']: r for r in rows}
    st.markdown("**➕ Add to a Toto coupon**")
    cc = st.columns([4, 1])
    cc[0].multiselect("Matches", list(by_label), key=f'{ns}_sel',
                      label_visibility='collapsed',
                      placeholder="Pick matches to add…")
    cc[1].radio("Coupon", list(toto.GAMES), key=f'{ns}_game',
                format_func=lambda g: g.capitalize())
    st.button("Add selected", key=f'{ns}_add', on_click=_do_add,
              args=(ns, by_label))
    added = st.session_state.pop(f'{ns}_added', None)
    if added:
        n, dups, g = added
        msg = f"Added {n} match(es) to the **{g.capitalize()}** coupon — see 🎟️ Toto."
        if dups:
            msg += f" Skipped {dups} already in it."
        st.success(msg)


# ----------------------------------------------------------------------------
hist, team_stats, team_to_league, leagues, teams_by_league = load_everything()


@st.cache_resource(show_spinner=False)
def known_teams():
    """Every selectable team: clubs from the trained leagues + national sides."""
    intl_names = sorted(toto._load_intl()['ratings'])
    return sorted(set(team_to_league) | set(intl_names))


STALE_DAYS = 28


@st.cache_data(ttl=120, show_spinner=False)
def modelled_leagues():
    """Leagues that actually have trained models behind them. A league can be
    in the data (so it appears in the dropdowns) while its models have not been
    trained yet — without this the app just returns nothing and says nothing.

    Short TTL rather than a permanent cache: it is only a directory scan, and
    a league finishing training in the background should show up without
    restarting the app."""
    try:
        return set(utils.get_available_leagues())
    except Exception:
        return set(leagues)


@st.cache_data(show_spinner=False)
def league_freshness():
    """Days since each league's last result — a prediction is only as current
    as the form data behind it."""
    today = pd.Timestamp.today().normalize()
    last = hist.groupby('league')['Date'].max()
    return {lg: (today - d).days for lg, d in last.items()}


def stale_note(lg):
    """Warning text if this league's data has gone cold, else None."""
    days = league_freshness().get(lg)
    if days is None or days <= STALE_DAYS:
        return None
    return (f"⚠️ **{disp(lg)}** data is **{days} days old** (last result "
            f"{(pd.Timestamp.today().normalize() - pd.Timedelta(days=days)).date()}). "
            "Form-based features are out of date — treat this prediction with "
            "caution. Either the league is between seasons, or the new "
            "season's file is not published yet.")


# ----------------------------------------------------------------------------
# Model version: v1 (production tier1) / v2 (frozen bundle) / auto
# ----------------------------------------------------------------------------
MODE_LABELS = {'auto': 'Auto', 'v1': 'v1', 'v2': 'v2'}
MODE_HELP = {
    'auto': "**Auto** — v2 for 🇹🇷 Süper Lig (v1's Turkish model fails on "
            "unseen matches), v1 everywhere else.",
    'v1': "**v1** — the production models everywhere.",
    'v2': "**v2** — the new model everywhere (1X2 only; see V2_RESULTS.md).",
}
if 'model_mode' not in st.session_state:
    _q = st.query_params.get('model', 'auto')
    st.session_state.model_mode = _q if _q in predict_mod.MODEL_MODES else 'auto'
predict_mod.set_model_mode(st.session_state.model_mode)


def model_mode():
    return st.session_state.model_mode


def served(league, v1, v2, mkt=None):
    """(probs [home, draw, away], label) actually used for one fixture:
    the market when there are odds, else v2 or v1 per the current mode."""
    if mkt is not None:
        return mkt, 'odds'
    if v2 is not None and predict_mod.serves_v2(league, model_mode()):
        return v2, 'v2'
    return v1, 'v1'


def pct3(p):
    """[home, draw, away] -> '45 · 28 · 27' (percent), or '—'."""
    return '—' if p is None else ' · '.join(f"{x * 100:.0f}" for x in p)


def _hda(d):
    return None if not d else (d['home'], d['draw'], d['away'])


def prob_bars(rows):
    """Stacked 1 / X / 2 bars, one per (label, probs, highlight) — a quick
    visual of where the models and the market disagree."""
    colors = ('#3b6fb6', '#8f8f8f', '#e07b39')         # home · draw · away
    html = []
    for label, p, hot in rows:
        if p is None:
            continue
        segs = ''.join(
            f"<div style='width:{x * 100:.1f}%;background:{c};color:#fff;"
            f"text-align:center;font-size:12px;line-height:22px;"
            f"overflow:hidden;white-space:nowrap'>{x * 100:.0f}%</div>"
            for x, c in zip(p, colors))
        weight = '700' if hot else '400'
        html.append(
            f"<div style='display:flex;align-items:center;gap:10px;margin:3px 0'>"
            f"<div style='width:92px;font-size:13px;font-weight:{weight}'>{label}</div>"
            f"<div style='flex:1;display:flex;border-radius:5px;overflow:hidden;"
            f"height:22px'>{segs}</div></div>")
    legend = ("<div style='font-size:11px;opacity:.7;margin-top:2px'>"
              "<span style='color:#3b6fb6'>■</span> home &nbsp; "
              "<span style='color:#8f8f8f'>■</span> draw &nbsp; "
              "<span style='color:#e07b39'>■</span> away</div>")
    st.markdown(''.join(html) + legend, unsafe_allow_html=True)


PCT_COL = {k: st.column_config.ProgressColumn(
    k, min_value=0, max_value=100, format="%d%%", width='small')
    for k in ('1', 'X', '2')}


# ----------------------------------------------------------------------------
# Sidebar — model switch, data freshness, one-click refresh
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Model")
    _pick = st.segmented_control(
        "Serve 1X2 from", predict_mod.MODEL_MODES, key='model_mode_w',
        default=st.session_state.model_mode, format_func=MODE_LABELS.get,
        label_visibility='collapsed')
    if _pick and _pick != st.session_state.model_mode:
        st.session_state.model_mode = _pick
        st.query_params['model'] = _pick           # survives reloads/bookmarks
        predict_mod.set_model_mode(_pick)
    st.caption(MODE_HELP[model_mode()] + " With bookmaker odds the market is "
               "used whichever you pick.")
    st.toggle("Show v1 / v2 side by side", value=True, key='show_both')
    try:
        from scripts import v2_bundle as _v2b
        _info = _v2b.serving_info()
    except Exception as _e:                          # never break the app
        _info = {'ok': False, 'error': str(_e)}
    if _info.get('ok'):
        st.caption(f"v2 bundle `{_info['tag']}` · trained through "
                   f"{_info['trained_through']}")
    else:
        st.caption(f"⚠️ v2 unavailable ({_info.get('error', '?')}) — "
                   "everything falls back to v1.")

    st.markdown("### 📡 Data")
    st.caption(f"Club results through **{hist['Date'].max().date()}**")
    try:
        from scripts import international as _intl_mod
        _age_h = (dt.datetime.now().timestamp()
                  - os.path.getmtime(_intl_mod.INTL_FILE)) / 3600
        _age = (f"{_age_h:.0f}h ago" if _age_h < 48
                else f"{_age_h / 24:.0f} days ago")
        st.caption(("⚠️ " if _age_h > 24 * 30 else "")
                   + f"Internationals updated **{_age}**")
    except OSError:
        pass

    _fresh = league_freshness()
    _stale = sorted(((d, lg) for lg, d in _fresh.items() if d > STALE_DAYS),
                    reverse=True)
    if _stale:
        with st.expander(f"⚠️ {len(_stale)} league(s) stale", expanded=False):
            st.caption("No recent results — off-season, or the new season's "
                       "file is not published yet. Predictions for these use "
                       "old form.")
            st.dataframe(
                pd.DataFrame([{'League': disp(lg), 'Days old': d}
                              for d, lg in _stale]),
                hide_index=True, use_container_width=True)
    else:
        st.caption(f"✅ all {len(_fresh)} leagues current")

    if st.button("🔄 Update league data", key="sb_leagues",
                 help="Download the latest results CSVs for all 28 leagues "
                      "from football-data.co.uk, rebuild features and reload "
                      "the app with fresh data (~2–3 min)."):
        import subprocess
        _fetch = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'scripts', 'fetch_latest.py')
        with st.spinner("Downloading CSVs + rebuilding features… (~2–3 min)"):
            proc = subprocess.run([sys.executable, _fetch,
                                   '--apply', '--refresh-processed'],
                                  capture_output=True, text=True)
        out = (proc.stdout or '')
        # Report what actually happened per league rather than a bare exit code:
        # a partial failure (a season not published yet) is normal and must not
        # be reported as a clean success.
        rows, promoted = [], 0
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] in config.FETCH_SOURCES:
                rows.append((parts[0], parts[1]))
            if line.strip().startswith('Promoted '):
                try:
                    promoted = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
        failed = [lg for lg, status in rows if status != 'OK']
        if rows:
            st.toast(f"Updated {promoted} league file(s)")
            if failed:
                st.warning(f"{len(failed)} not updated: "
                           + ', '.join(failed[:8])
                           + ("" if len(failed) <= 8 else ' …')
                           + "  — usually a season that has not started yet. "
                             "Existing data is kept.")
            load_everything.clear()
            known_teams.clear()
            league_freshness.clear()
            modelled_leagues.clear()
            st.rerun()
        else:
            st.error("Update failed")
            st.code(out[-1200:] + '\n' + (proc.stderr or '')[-600:])

    if st.button("🔄 Update internationals / WC", key="sb_intl",
                 help="Pull the latest national-team results + upcoming "
                      "fixtures (same as the World Cup tab's refresh)."):
        from scripts import international as _intl_mod
        try:
            with st.spinner("Downloading latest international results…"):
                _intl_mod.cmd_update(None)
            st.success("Internationals refreshed.")
        except Exception as e:
            st.error(f"Refresh failed: {e}")


st.title("⚽ Football Predictor")
# Toto first: it is the thing that gets used every week.
tab_toto, tab_fix, tab_match, tab_wc = st.tabs(
    ["🎟️ Toto", "📅 Fixtures", "🎯 Match", "🌍 Internationals"])


# ============================================================================
# TAB 1 — single match
# ============================================================================
with tab_match:
    c1, c2, c3 = st.columns(3)
    lg = c1.selectbox("League", leagues, format_func=disp,
                      index=leagues.index('turkish') if 'turkish' in leagues else 0)
    opts = teams_by_league.get(lg, [])
    home = c2.selectbox("Home team", opts, key="home")
    away = c3.selectbox("Away team", opts,
                        index=1 if len(opts) > 1 else 0, key="away")

    if lg not in modelled_leagues():
        st.info(f"**{disp(lg)}** has match data but no trained models yet, so "
                "predictions for it will come back empty. Train it with "
                f"`python scripts/train_one.py --league {lg} --engine xgb` "
                "(then `--engine lgbm`).")

    if st.button("Predict", type="primary") and home != away:
        try:
            pred = predict_mod.predict_match(
                home, away, team_stats, team_to_league, hist,
                include_xg=True, prediction_date=pd.Timestamp.now())
            st.session_state.match_pred = {'home': home, 'away': away,
                                           'pred': pred}
        except Exception as e:
            st.session_state.match_pred = {'home': home, 'away': away,
                                           'error': str(e)}

    mp = st.session_state.get('match_pred')
    if mp and mp.get('error'):
        st.error(f"Could not predict: {mp['error']}")
    elif mp and mp.get('pred'):
        home_p, away_p, pred = mp['home'], mp['away'], mp['pred']
        anchored = 'market' in pred
        st.subheader(f"{home_p} vs {away_p}")
        if anchored:
            st.caption("✅ 1X2 / O-U anchored to live bookmaker odds")
        _sn = stale_note(pred.get('league') or lg)
        if _sn:
            st.warning(_sn)

        p = pred.get('1x2', {})
        if not p:
            st.error(
                f"No 1X2 model produced a result for **{disp(pred.get('league') or lg)}**. "
                "The league's models are missing or failed to load — see the "
                "note above, or run `python scripts/selftest.py` to check.")
        if p:
            mkt = pred.get('market')
            lg_key = pred.get('league') or lg
            v1, v2 = _hda(pred.get('1x2_v1')), _hda(pred.get('1x2_v2'))
            m_imp = _hda(mkt['implied']) if mkt else None
            used, used_src = served(lg_key, v1, v2, m_imp)
            model_p, model_src = served(lg_key, v1, v2)      # the model on its own
            both = st.session_state.get('show_both', True)
            rows = []
            for i, (lbl, key) in enumerate([(f"🏠 {home_p}", 'home'), ("🤝 Draw", 'draw'),
                                            (f"✈️ {away_p}", 'away')]):
                r = {'Outcome': lbl, 'Used': f"{used[i]:.0%}",
                     'Fair odds': f"{1 / used[i]:.2f}" if used[i] > 0 else '—'}
                if both:
                    for name, q in (('v1', v1), ('v2', v2)):
                        head = name + (' ✓' if model_src == name else '')
                        r[head] = f"{q[i]:.0%}" if q else '—'
                else:
                    r[f'Model ({model_src})'] = f"{model_p[i]:.0%}"
                if mkt:
                    r['Book odds'] = f"{mkt['odds'][key]:.2f}"
                    r['Book %'] = f"{m_imp[i]:.0%}"
                    r['Model − Book'] = f"{model_p[i] - m_imp[i]:+.0%}"
                rows.append(r)
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)
            if both:
                prob_bars([('v1' + (' ✓' if model_src == 'v1' else ''), v1, not mkt and model_src == 'v1'),
                           ('v2' + (' ✓' if model_src == 'v2' else ''), v2, not mkt and model_src == 'v2'),
                           ('Market', m_imp, bool(mkt))])
            if v2 is None and both:
                st.caption("v2 has no prediction for this pair (a team it has "
                           "never seen in this league) — v1 is used.")
            if mkt:
                st.caption(
                    "**Used** is the market: odds were found, and they beat both "
                    "models. **✓** marks the model the switch selects. "
                    "**Model − Book** is disagreement, *not* value: on unseen "
                    "matches v1 trailed the market by ~0.03–0.04 log-loss and v2 "
                    "by ~0.002–0.005, and blending either into the odds does not "
                    "reliably help. The models earn their keep on matches with "
                    "no odds, and in Toto, where you play the crowd.")
            else:
                st.caption(f"No live odds for this fixture in the feed — **Used** "
                           f"is model **{model_src}** (✓).")

        cols = st.columns(3)
        if 'ou25' in pred:
            ou = f"{pred['ou25']['over']:.0%}"
            if pred.get('market', {}).get('ou25_odds'):
                ou += f"  (book {pred['market']['ou25_odds']['over']:.2f})"
            cols[0].metric("Over 2.5", ou)
        if 'btts' in pred:
            cols[1].metric("BTTS", f"{pred['btts']['yes']:.0%}")
        if 'xg' in pred:
            cols[2].metric("xG", f"{pred['xg']['home']:.1f} – "
                                 f"{pred['xg']['away']:.1f}")

        ac = st.columns([2, 1, 3])
        gm = ac[0].radio("Add to coupon", list(toto.GAMES), horizontal=True,
                         format_func=lambda g: g.capitalize(), key='match_game')
        if ac[1].button("➕ Add", key='match_add'):
            if _append_line(gm, home_p, away_p,
                            (pred.get('market') or {}).get('odds')) == 'added':
                st.success(f"Added **{home_p} v {away_p}** to the "
                           f"{gm.capitalize()} coupon — see 🎟️ Toto.")
            else:
                st.info(f"**{home_p} v {away_p}** is already in the "
                        f"{gm.capitalize()} coupon.")

        with st.expander("Full breakdown"):
            st.text(utils.format_prediction_table(pred))


# ============================================================================
# TAB 2 — upcoming fixtures
# ============================================================================
with tab_fix:
    days = st.slider("Days ahead", 1, 14, 3)
    if st.button("Load fixtures", type="primary"):
        from scripts import today as today_mod
        with st.spinner("Fetching odds feed and predicting…"):
            st.session_state.fix_rows = today_mod.club_section(
                days, hist, team_stats, team_to_league)

    frows = st.session_state.get('fix_rows')
    if frows is not None:
        if not frows:
            st.info("No club fixtures with odds in this window "
                    "(leagues may be on a break).")
        else:
            all_lgs = sorted({r['League'] for r in frows})
            pick_lgs = st.multiselect(
                "Leagues", all_lgs, default=[], key='fix_lgs',
                placeholder=f"All {len(all_lgs)} leagues "
                            f"({len(frows)} matches) — pick to narrow")
            shown = [r for r in frows
                     if not pick_lgs or r['League'] in pick_lgs]

            both = st.session_state.get('show_both', True)
            table = []
            for r in shown:
                if '_v1' in r:
                    v1, v2, mk, lgk = r['_v1'], r['_v2'], r['_mkt'], r['_league']
                else:                        # rows loaded before this version
                    v1, v2, mk, lgk = (r['P(1)'], r['P(X)'], r['P(2)']), None, None, None
                used, src = served(lgk, v1, v2, mk)
                model_p, _ = served(lgk, v1, v2)
                row = {'Date': r['Date'], 'League': r['League'],
                       'Home': r['Home'], 'Away': r['Away'],
                       '1': round(used[0] * 100), 'X': round(used[1] * 100),
                       '2': round(used[2] * 100),
                       'Pick': '1X2'[int(np.argmax(used))], 'Used': src}
                if both:
                    row['v1'], row['v2'] = pct3(v1), pct3(v2)
                row['Odds'] = r.get('Odds(1/X/2)', '—')
                if r.get('P(O2.5)') is not None:
                    row['O2.5'] = round(r['P(O2.5)'] * 100)
                if r.get('P(BTTS)') is not None:
                    row['BTTS'] = round(r['P(BTTS)'] * 100)
                if mk is not None and model_p is not None:
                    k = int(np.argmax(model_p))
                    row['Edge'] = round((model_p[k] - mk[k]) * 100)
                table.append(row)
            df = pd.DataFrame(table)
            st.caption(f"Showing **{len(shown)}** of {len(frows)} fixtures · "
                       f"model: **{MODE_LABELS[model_mode()]}**.")
            st.dataframe(df, use_container_width=True, hide_index=True,
                         column_config={
                             **PCT_COL,
                             'O2.5': st.column_config.NumberColumn(format="%d%%"),
                             'BTTS': st.column_config.NumberColumn(format="%d%%"),
                             'Edge': st.column_config.NumberColumn(format="%+d pts"),
                         })
            st.caption(
                "**1 · X · 2** are the probabilities used. **Used**: *odds* = live "
                "bookmaker odds (they beat both models), *v1/v2* = the model the "
                "sidebar switch selects. **v1 / v2** show both models as "
                "home · draw · away %. **Edge** = selected model minus market on "
                "the model's pick — disagreement to look into, not a value bet.")
            st.download_button("Download CSV", df.to_csv(index=False),
                               "fixtures.csv")

            add_rows = [{
                '_label': f"{r['Date']} · {r['Home']} - {r['Away']}",
                'home': r['Home'], 'away': r['Away'],
                'odds': _parse_slash_odds(r.get('Odds(1/X/2)')),
            } for r in shown]
            _add_controls(add_rows, 'fix')


# ============================================================================
# TAB 3 — World Cup / internationals
# ============================================================================
with tab_wc:
    wc_days = st.slider("Days ahead ", 1, 21, 5, key="wcdays")
    wca, wcb = st.columns([1, 1])
    load_wc = wca.button("Load upcoming internationals", type="primary")
    if wcb.button("🔄 Refresh results + fixtures", key="wc_refresh",
                  help="Pull the latest international results and upcoming "
                       "fixtures. Knockout matches only appear here once the "
                       "group stage finishes and the bracket is set."):
        from scripts import international as intl_u
        try:
            with st.spinner("Downloading latest international results…"):
                intl_u.cmd_update(None)
            st.success("Data refreshed — now click **Load upcoming internationals**.")
        except Exception as e:
            st.error(f"Refresh failed: {e}")

    if load_wc:
        st.session_state.wc_out = None
        st.session_state.wc_rows = []
        st.session_state.wc_msg = None
        try:
            from scripts import international as intl
            if not os.path.exists(intl.WC_FIXTURES_FILE):
                st.session_state.wc_msg = (
                    "warning", "No WC fixtures file. Run: "
                    "python scripts/international.py update")
            else:
                fx = pd.read_csv(intl.WC_FIXTURES_FILE, parse_dates=['date'])
                now = pd.Timestamp.now().normalize()
                fx = fx[(fx['date'] >= now)
                        & (fx['date'] <= now + pd.Timedelta(days=wc_days))]
                if fx.empty:
                    st.session_state.wc_msg = ("info",
                                               "No WC matches in this window.")
                else:
                    rdf = intl.load_results()
                    ratings, h = intl.run_elo(rdf)
                    model = intl.GoalModel().fit(h)
                    out, add_rows = [], []
                    for _, m in fx.sort_values('date').iterrows():
                        hm, aw = m['home_team'], m['away_team']
                        if hm not in ratings or aw not in ratings:
                            continue
                        pr = model.market_probs(ratings[hm], ratings[aw],
                                                neutral=bool(m['neutral']))
                        out.append({
                            'Date': m['date'].date(),
                            'Match': f"{hm} vs {aw}",
                            '1 %': round(pr['p_home'] * 100),
                            'X %': round(pr['p_draw'] * 100),
                            '2 %': round(pr['p_away'] * 100),
                            'Fair(1)': round(1 / pr['p_home'], 2) if pr['p_home'] > 0 else None,
                            'Fair(X)': round(1 / pr['p_draw'], 2) if pr['p_draw'] > 0 else None,
                            'Fair(2)': round(1 / pr['p_away'], 2) if pr['p_away'] > 0 else None,
                            'O2.5 %': round(pr['p_over25'] * 100),
                            'BTTS %': round(pr['p_btts'] * 100),
                        })
                        add_rows.append({
                            '_label': f"{m['date'].date()} · {hm} - {aw}",
                            'home': hm, 'away': aw, 'odds': None})
                    st.session_state.wc_out = out
                    st.session_state.wc_rows = add_rows
        except Exception as e:
            st.session_state.wc_msg = ("error", f"World Cup section error: {e}")

    msg = st.session_state.get('wc_msg')
    if msg:
        getattr(st, msg[0])(msg[1])
    wc_out = st.session_state.get('wc_out')
    if wc_out:
        st.dataframe(pd.DataFrame(wc_out), use_container_width=True,
                     hide_index=True)
        _add_controls(st.session_state.get('wc_rows', []), 'wc')


# ============================================================================
# TAB 4 — Toto coupon optimizer (two persistent coupons)
# ============================================================================
with tab_toto:
    st.caption("Two persistent coupons — **Turkish** and **German** — saved to "
               "disk until you clear them. Build the coupon three ways: "
               "**quick-add** below (search any club or national team), "
               "**paste** lines into the box (`Home - Away`, optional odds "
               "after), or **push** matches from the other tabs. No odds → the "
               "model prices it (club or World Cup). Duplicates are caught "
               "automatically.")
    cset = st.columns(3)
    game = cset[0].radio("Game", list(toto.GAMES), horizontal=True,
                         format_func=lambda g: g.capitalize(), key='toto_game')
    n_exp, threshold, top_tier = toto.GAMES[game]
    budget = cset[1].number_input("System budget (columns)", 1, 100000, 1,
                                  help="1 = single column. Higher = cover "
                                       "toss-ups with doubles/triples.")
    cset[2].markdown(f"**Coupon**  \n{n_exp} matches · prize at **{threshold}+** "
                     f"correct")

    key = f'coupon_{game}'
    if key not in st.session_state:
        st.session_state[key] = _load_coupon_file(game)

    # --- quick add: searchable over every club + national team ---------------
    qa = st.columns([3, 3, 2, 1])
    qa[0].selectbox("Home", known_teams(), index=None, key='qa_home',
                    placeholder="Home team…", label_visibility='collapsed')
    qa[1].selectbox("Away", known_teams(), index=None, key='qa_away',
                    placeholder="Away team…", label_visibility='collapsed')
    qa[2].text_input("Odds", key='qa_odds', label_visibility='collapsed',
                     placeholder="odds 1 X 2 (optional)")
    qa[3].button("➕ Add", key='qa_btn', on_click=_quick_add)
    qmsg = st.session_state.pop('qa_msg', None)
    if qmsg:
        getattr(st, qmsg[0])(qmsg[1])

    text = st.text_area(
        "Matches (one per line)", key=key, height=320,
        placeholder="Norway - Italy\nTurkey - Spain  1.95 3.40 3.90\n"
                    "Brazil - Morocco\n…")
    _save_coupon_file(game, st.session_state[key])      # persist edits to disk

    parsed = toto.parse_lines(text)

    # --- live validation: count, duplicates, unrecognized names --------------
    n_par = len(parsed)
    tick = '✅' if n_par == n_exp else ('⚠️' if n_par > n_exp else '✏️')
    st.caption(f"{tick} **{n_par}** matches on the coupon "
               f"(this game wants {n_exp}).")
    if not parsed.empty:
        dups, unknown, seen = [], [], set()
        ic = toto._load_intl()
        club = set(team_to_league)
        for _, r in parsed.iterrows():
            p_id = _pair(r['home'], r['away'])
            if p_id in seen:
                dups.append(f"{r['home']} - {r['away']}")
            seen.add(p_id)
            if all(pd.notna(r.get(c)) for c in ('o1', 'ox', 'o2')):
                continue                         # odds present -> always priced
            if toto._fuzzy(r['home'], club) and toto._fuzzy(r['away'], club):
                continue
            if toto._intl_name(r['home'], ic) and toto._intl_name(r['away'], ic):
                continue
            unknown.append(f"{r['home']} - {r['away']}")
        if dups:
            dc1, dc2 = st.columns([4, 1])
            dc1.warning("Duplicate matches: " + " · ".join(dups))
            dc2.button("🧹 Remove duplicates", on_click=_dedup_coupon,
                       args=(game,))
        if unknown:
            st.warning("Not recognized — these will default to 1/3 each unless "
                       "you add odds or fix the name.")
            known_all = toto.all_known_names(team_to_league)
            club = set(team_to_league)
            ic2 = toto._load_intl()
            seen_bad = []
            for _, r in parsed.iterrows():
                for side in ('home', 'away'):
                    nm = str(r[side]).strip()
                    if not nm or nm in seen_bad:
                        continue
                    if toto._fuzzy(nm, club) or toto._intl_name(nm, ic2):
                        continue
                    seen_bad.append(nm)
            for nm in seen_bad[:8]:
                opts = toto.suggest_names(nm, known_all, 6)
                fc = st.columns([2, 3, 1])
                fc[0].markdown(f"`{nm}`")
                if opts:
                    key = f'fix_{game}_{nm}'
                    fc[1].selectbox("Did you mean", opts, key=key,
                                    label_visibility='collapsed')
                    fc[2].button("Fix", key=f'btn_{key}', on_click=_fix_name,
                                 args=(game, nm, key))
                else:
                    fc[1].caption("no close match — add odds for this row")

    ca1, ca2, ca3 = st.columns([1, 1, 1])
    analyze = ca1.button("Analyze coupon", type="primary")
    ca2.button("📡 Fill odds from feed", on_click=_fill_odds, args=(game,),
               help="Look each odds-less match up in the live bookmaker feed "
                    "and fill in 1X2 odds. Where odds exist they carry the "
                    "prediction, so this is the single best thing you can do "
                    "to a coupon.")
    ca3.button("🗑️ Clear this coupon", on_click=_clear_game, args=(game,))
    if st.session_state.get(f'undo_{game}'):
        st.button("↩️ Undo clear", key=f'undo_btn_{game}',
                  on_click=_undo_clear, args=(game,))
    _om = st.session_state.pop(f'odds_msg_{game}', None)
    if _om:
        getattr(st, _om[0])(_om[1])

    if analyze:
        if parsed.empty:
            st.info("Add some matches first — quick-add above, paste, or push "
                    "from the other tabs.")
        else:
            ctx = {'hist': hist, 'team_stats': team_stats,
                   'team_to_league': team_to_league,
                   'teams': set(team_to_league),
                   'weights': toto._load_blend_weights()}
            out, sorted_probs, unmatched, picks = [], [], [], []
            for _, r in parsed.iterrows():
                p, src, model = toto.match_probs(r, ctx)
                if p is None:
                    p = np.array([0.34, 0.33, 0.33]); src = 'no data'
                    unmatched.append(f"{r['home']} v {r['away']}")
                order = np.argsort(-p)
                flag = ''
                has_odds = all(pd.notna(r.get(c)) for c in ('o1', 'ox', 'o2'))
                if model is not None and src == 'blend' and has_odds:
                    if np.argmax(model) != np.argmax(
                            toto._devig(r['o1'], r['ox'], r['o2'])):
                        flag = '⚠ contrarian'
                # say where the numbers came from, in plain words
                det = ctx.get('model_detail', {}).get((str(r['home']), str(r['away'])))
                shown_src = src
                if src in ('blend', 'odds'):
                    shown_src = 'odds'
                elif src == 'model' and det:
                    shown_src = 'feed odds' if det['feed_odds'] else det['version']
                out.append({'Match': f"{r['home']} v {r['away']}",
                            '1': round(p[0] * 100), 'X': round(p[1] * 100),
                            '2': round(p[2] * 100),
                            'Pick': toto.OUTCOMES[order[0]],
                            'Fair': round(1 / p[order[0]], 2)
                            if p[order[0]] > 0 else None,
                            'Src': shown_src,
                            'v1': pct3(_hda(det['v1'])) if det else '—',
                            'v2': pct3(_hda(det['v2'])) if det else '—',
                            'Note': flag})
                sorted_probs.append(np.sort(p)[::-1])
                picks.append({'home': str(r['home']), 'away': str(r['away']),
                              'p1': float(p[0]), 'px': float(p[1]),
                              'p2': float(p[2]),
                              'pick': toto.OUTCOMES[order[0]], 'src': src})
            st.session_state[f'toto_res_{game}'] = {
                'text': text, 'out': out, 'sorted_probs': sorted_probs,
                'unmatched': unmatched, 'picks': picks, 'mode': model_mode()}

    # --- results persist across reruns; system section follows the budget ----
    res = st.session_state.get(f'toto_res_{game}')
    if res:
        if res['text'].strip() != text.strip():
            st.info("✏️ The coupon changed since this analysis — hit "
                    "**Analyze coupon** to refresh.")
        elif res.get('mode', 'auto') != model_mode():
            st.info(f"🧠 The model switch changed to **{MODE_LABELS[model_mode()]}** "
                    "since this analysis — hit **Analyze coupon** to re-price "
                    "the rows priced by a model.")
        out, sorted_probs = res['out'], res['sorted_probs']

        _odf = pd.DataFrame(out)
        if not st.session_state.get('show_both', True):
            _odf = _odf.drop(columns=['v1', 'v2'], errors='ignore')
        st.dataframe(_odf, use_container_width=True, hide_index=True,
                     column_config=PCT_COL)
        st.caption("Fair = fair decimal odds for the pick (1 ÷ probability) "
                   "— bet it only if a book pays more. **Src** (where 1 · X · 2 "
                   "came from): **odds** (typed on the coupon) · **feed odds** "
                   "(found in the live bookmaker feed) · **v1 / v2** (club model, "
                   "per the sidebar switch) · **euro** (cross-league clubs via "
                   "global club Elo) · **intl** (national teams) · **no data** "
                   "(1∕3 each). **v1 / v2** show both club models as "
                   "home · draw · away %.")
        if res['unmatched']:
            st.warning("Couldn't match — defaulted to 1/3 each. Check the "
                       "spelling, or add odds (`… 2.10 3.30 3.40`):\n\n- "
                       + "\n- ".join(res['unmatched']))

        q_single = [sp[0] for sp in sorted_probs]
        d = toto.pb_distribution(q_single)
        st.markdown(f"**Single column** — expected correct ≈ "
                    f"{sum(q_single):.1f}/{len(out)}")
        tiers = st.columns(min(4, top_tier - threshold + 1))
        for k, t in enumerate(range(threshold, top_tier + 1)):
            if t < len(d) and k < len(tiers):
                tiers[k].metric(f"P(≥{t})", f"{d[t:].sum():.1%}")

        p_single = d[threshold:].sum()
        sys_cov = None
        if budget > 1:
            cov, cols, p_thr = toto.optimize_system(
                sorted_probs, threshold, int(budget))
            sys_cov = (cov, cols, p_thr)
            st.markdown(f"**Best system in {budget} columns** — uses "
                        f"{cols} columns; **P(≥{threshold}) = {p_thr:.1%}** "
                        f"(vs {p_single:.1%} single)")
            ups = [(i, cov[i]) for i in range(len(cov)) if cov[i] > 1]
            if ups:
                st.markdown("Cover these least-predictable matches:")
                for i, c in ups:
                    kind = "**TRIPLE** (play 1, X and 2)" if c == 3 \
                        else "**double** (top 2 outcomes)"
                    st.markdown(f"- {out[i]['Match']} → {kind}")

        # --- how far does each extra column actually get you? ----------------
        with st.expander("💸 What does another column buy me?"):
            ladder = []
            prev = None
            for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                cv, cl, pt = toto.optimize_system(sorted_probs, threshold, b)
                if prev is not None and cl == prev:
                    continue            # this budget cannot buy anything new
                prev = cl
                ladder.append({
                    'Columns': cl,
                    f'P(≥{threshold})': f"{pt:.2%}",
                    'vs single': (f"×{pt / p_single:.1f}"
                                  if p_single > 0 else '—'),
                    'Doubles': sum(1 for c in cv if c == 2),
                    'Triples': sum(1 for c in cv if c == 3),
                })
            st.dataframe(pd.DataFrame(ladder), use_container_width=True,
                         hide_index=True)
            st.caption(
                "Cost rises with the column count while P(prize) rises much "
                "more slowly, so the last doubling is always the worst value — "
                "this is for choosing a budget you are happy to lose, not for "
                "finding a profitable one.")

        # --- the actual ticket: which symbols to fill in ---------------------
        with st.expander("🎫 Build my ticket", expanded=True):
            from scripts import ticket as ticket_mod
            st.caption(
                "Turns the coupon and your budget into the ticket you actually "
                "fill in — `1`, `1X`, `12`, `1X2` per match. Optionally paste "
                "the crowd percentages (*oynanma yüzdesi*) as `68/21/11` per "
                "line, in coupon order, and it will also lean toward outcomes "
                "the field is under-backing — everyone with "
                f"{threshold}+ splits the pot, so the weeks worth winning are "
                "the ones the crowd loses.")
            tc1, tc2 = st.columns([2, 3])
            tilt = tc1.slider(
                "Crowd tilt", 0.0, 1.0, ticket_mod.DEFAULT_TILT, 0.05,
                key=f'tilt_{game}',
                help="0 = pure hit chance. Higher leans into unpopular "
                     "outcomes for a bigger share when you win. Past ~0.75 "
                     "the hit rate collapses — check the trade-off table.")
            crowd_txt = tc2.text_area(
                "Crowd % per match (optional)", key=f'crowd_{game}', height=110,
                placeholder="68/21/11\n55/25/20\n…")
            crowd_arr = ticket_mod.parse_crowd(crowd_txt)
            if crowd_txt.strip() and (crowd_arr is None
                                      or len(crowd_arr) != len(out)):
                st.warning(f"Need one `1/X/2` line per match — got "
                           f"{0 if crowd_arr is None else len(crowd_arr)} for "
                           f"{len(out)} matches. Ignoring the crowd numbers.")
                crowd_arr = None

            # Full-precision probabilities, not the table's rounded percents —
            # rounding to whole numbers can change which ticket is optimal.
            full_probs = [np.array([m['p1'], m['px'], m['p2']])
                          for m in res.get('picks', [])] or \
                [np.array([r['1'], r['X'], r['2']]) / 100.0 for r in out]
            bt1, bt2 = st.columns([1, 1])
            if bt1.button("🎫 Build ticket", key=f'mk_{game}'):
                with st.spinner("Choosing symbols…"):
                    st.session_state[f'ticket_{game}'] = ticket_mod.build(
                        full_probs,
                        threshold, int(budget), crowd=crowd_arr, tilt=tilt)
            if crowd_arr is not None and bt2.button("⚖️ Show trade-off",
                                                    key=f'sw_{game}'):
                with st.spinner("Comparing tilts…"):
                    st.session_state[f'sweep_{game}'] = ticket_mod.sweep(
                        full_probs, threshold, int(budget), crowd_arr)

            tk = st.session_state.get(f'ticket_{game}')
            if tk and len(tk['labels']) == len(out):
                tdf = pd.DataFrame([{
                    '#': i + 1, 'Match': out[i]['Match'],
                    'PLAY': tk['labels'][i],
                    'cols': ticket_mod.popcount(tk['masks'][i]),
                    '1': out[i]['1'], 'X': out[i]['X'], '2': out[i]['2'],
                } for i in range(len(out))])
                st.dataframe(tdf, use_container_width=True, hide_index=True,
                             column_config=PCT_COL)
                mk1, mk2, mk3 = st.columns(3)
                mk1.metric("Columns used", f"{tk['columns']} / {int(budget)}")
                mk2.metric(f"P(≥{threshold})", f"{tk['p_hit']:.2%}")
                mk3.metric("Optimised for",
                           "crowd-weighted" if tk['objective'] == 'payout'
                           else "hit chance")
                st.download_button(
                    "Download ticket", tdf.to_csv(index=False),
                    f"ticket_{game}.csv", key=f'dl_{game}')

            sw = st.session_state.get(f'sweep_{game}')
            if sw:
                st.markdown("**Hit chance vs payout** (payout relative to "
                            "tilt 0, all scored on one yardstick)")
                st.dataframe(pd.DataFrame([{
                    'tilt': r['tilt'], f'P(≥{threshold})': f"{r['p_hit']:.2%}",
                    'payout': f"{r['vs_base']:.2f}x",
                    'hits every': f"{1 / max(r['p_hit'], 1e-9):.0f} wk",
                    'ticket': r['ticket'],
                } for r in sw]), use_container_width=True, hide_index=True)
                st.caption("Take the largest tilt whose hit rate you can live "
                           "with. Past the cliff it is a lottery ticket. "
                           "'payout' is a crowd-weighted preference score for "
                           "comparing tickets — not expected money: it counts one "
                           "win per ticket and ignores separate prize tiers, "
                           "ticket cost and rollovers.")

        # --- record it, so "does this work?" becomes a number ----------------
        sv1, sv2 = st.columns([1, 3])
        note = sv2.text_input("Note (optional)", key=f'note_{game}',
                              placeholder="e.g. week 34, played 48 columns",
                              label_visibility='collapsed')
        # What gets saved must be the ticket you actually play, graded against
        # the probability of THAT ticket. Previously the singles were saved
        # alongside the system's P(prize), so history compared two different
        # tickets. Priority: a ticket from 🎫 Build my ticket, then the system
        # recommended above, then the single column.
        base = res.get('picks', [])
        built = st.session_state.get(f'ticket_{game}')
        if built and len(built.get('labels', [])) == len(base):
            played = [{**m, 'pick': lbl} for m, lbl in zip(base, built['labels'])]
            save_p, save_cols, save_kind = (built['p_hit'], built['columns'],
                                            'ticket')
        elif sys_cov is not None and len(sys_cov[0]) == len(base):
            cov, cols, p_thr = sys_cov
            played = []
            for m, c in zip(base, cov):
                pr = [m['p1'], m['px'], m['p2']]
                top = sorted(range(3), key=lambda k: -pr[k])[:c]
                played.append({**m, 'pick': ''.join('1X2'[k] for k in sorted(top))})
            save_p, save_cols, save_kind = p_thr, cols, 'system'
        else:
            played, save_p, save_cols, save_kind = base, p_single, 1, 'single'
        sv1.caption(f"Will save: **{save_kind}**, {save_cols} column(s), "
                    f"P(≥{threshold}) {save_p:.1%}")

        if sv1.button("💾 Save to history", key=f'save_{game}'):
            from scripts import track
            track.save_coupon(game, played, budget=int(budget),
                              threshold=threshold, p_threshold=float(save_p),
                              note=note or '', columns=int(save_cols),
                              kind=save_kind)
            st.session_state.pop('graded', None)
            st.success("Saved. Once these matches are played, grade them in "
                       "**📈 History** below.")

    # ------------------------------------------------------------------ history
    with st.expander("📈 History & calibration"):
        from scripts import track
        entries = track.load_history()
        if not entries:
            st.caption("No coupons saved yet. Analyse one above and hit "
                       "**💾 Save to history** — after the matches are played "
                       "this grades itself, so you can see your real hit rate "
                       "against what the model promised.")
        else:
            if st.button("🔄 Grade against results", key='grade_btn'):
                st.session_state['graded'] = track.grade_all()
            graded = st.session_state.get('graded')
            if graded is None:
                st.caption(f"{len(entries)} saved coupon(s). "
                           "Hit **Grade against results**.")
            else:
                rows, done, hits = [], 0, 0
                tot_c = tot_e = 0.0
                for g in graded:
                    pending = g['graded'] < g['n']
                    won = (g['threshold'] is not None
                           and g['correct'] >= g['threshold'])
                    if not pending and g['graded']:
                        done += 1; hits += int(won)
                        tot_c += g['correct']; tot_e += g['expected']
                    rows.append({
                        'Saved': (g['saved_at'] or '')[:16].replace('T', ' '),
                        'Game': (g['game'] or '').capitalize(),
                        'Cols': g.get('columns') or '—',
                        'Result': (f"{g['correct']}/{g['graded']}"
                                   + (f" (of {g['n']})" if pending else '')),
                        'Expected': round(g['expected'], 1),
                        'Need': g['threshold'],
                        'Prize': ('⏳' if pending else ('✅' if won else '—')),
                        'P(prize) said': (
                            f"{g['p_threshold']:.1%}"
                            if g.get('p_threshold') is not None else '—'),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
                if done:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Coupons completed", done)
                    m2.metric("Cleared the threshold", f"{hits}/{done}")
                    m3.metric("Correct vs expected",
                              f"{tot_c:.0f} / {tot_e:.1f}",
                              delta=f"{tot_c - tot_e:+.1f}")
                    st.caption(
                        "**Expected** is the Poisson-binomial mean — the honest "
                        "benchmark. Landing consistently below it means the "
                        "probabilities are running optimistic; around it means "
                        "they are calibrated and the rest is variance.")


st.caption(f"Loaded {len(team_to_league)} teams · {len(leagues)} leagues · "
           f"data through {hist['Date'].max().date()} · "
           f"{dt.date.today()}")
