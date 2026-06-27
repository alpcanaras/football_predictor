"""
Football Predictor — web app
=============================
A point-and-click interface. No notebooks, no cells.

    pip install streamlit          # one-time
    streamlit run app.py           # opens in your browser

Four tabs:
  * Match      — pick a league + two teams, get the full prediction card
  * Fixtures   — every upcoming match with models, anchored to live odds
  * World Cup  — today's / upcoming national-team matches
  * Toto       — two persistent coupons (Turkish + German), system optimizer

Any match in the first three tabs can be pushed into either coupon. Coupons are
saved to data/_toto/<game>.txt, so they survive reloads and new browser tabs
until you clear them.
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

COUPON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'data', '_toto')


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
    try:
        os.makedirs(COUPON_DIR, exist_ok=True)
        with open(_coupon_path(game), 'w', encoding='utf-8') as f:
            f.write(text or '')
    except OSError:
        pass


def _fmt_line(home, away, odds=None):
    line = f"{home} - {away}"
    if odds and all(odds.get(k) for k in ('home', 'draw', 'away')):
        line += f"  {odds['home']:.2f} {odds['draw']:.2f} {odds['away']:.2f}"
    return line


def _append_line(game, home, away, odds=None):
    """Append a match to a game's coupon. Disk is the source of truth, so this
    is safe even if Streamlit dropped the inactive coupon's widget state."""
    cur = _load_coupon_file(game).rstrip()
    new = (cur + '\n' + _fmt_line(home, away, odds)) if cur \
        else _fmt_line(home, away, odds)
    _save_coupon_file(game, new)
    st.session_state[f'coupon_{game}'] = new


def _parse_slash_odds(s):
    if not s:
        return None
    try:
        h, d, a = [float(x) for x in str(s).split('/')]
        return {'home': h, 'draw': d, 'away': a}
    except Exception:
        return None


def _clear_game(game):
    st.session_state[f'coupon_{game}'] = ''
    _save_coupon_file(game, '')


def _do_add(ns, by_label):
    """on_click callback: append the multiselect's picks to the chosen game."""
    sel = st.session_state.get(f'{ns}_sel', [])
    game = st.session_state.get(f'{ns}_game') or list(toto.GAMES)[0]
    n = 0
    for lab in sel:
        r = by_label.get(lab)
        if r:
            _append_line(game, r['home'], r['away'], r.get('odds'))
            n += 1
    st.session_state[f'{ns}_sel'] = []          # clear picks (safe in callback)
    if n:
        st.session_state[f'{ns}_added'] = (n, game)


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
        n, g = added
        st.success(f"Added {n} match(es) to the **{g.capitalize()}** coupon "
                   f"— see 🎟️ Toto.")


# ----------------------------------------------------------------------------
hist, team_stats, team_to_league, leagues, teams_by_league = load_everything()

st.title("⚽ Football Predictor")
tab_match, tab_fix, tab_wc, tab_toto = st.tabs(
    ["🎯 Match", "📅 Fixtures", "🌍 World Cup", "🎟️ Toto"])


# ============================================================================
# TAB 1 — single match
# ============================================================================
with tab_match:
    c1, c2, c3 = st.columns(3)
    lg = c1.selectbox("League", leagues, format_func=disp)
    opts = teams_by_league.get(lg, [])
    home = c2.selectbox("Home team", opts, key="home")
    away = c3.selectbox("Away team", opts,
                        index=1 if len(opts) > 1 else 0, key="away")

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

        p = pred.get('1x2', {})
        if p:
            # Stats-first table: probability, bookmaker odds, implied %, edge
            mkt = pred.get('market')
            rows = []
            for lbl, key in [(f"🏠 {home_p}", 'home'), ("🤝 Draw", 'draw'),
                             (f"✈️ {away_p}", 'away')]:
                r = {'Outcome': lbl, 'Model': f"{p[key]:.0%}",
                     'Fair odds': f"{(1/p[key]):.2f}" if p[key] > 0 else '—'}
                if mkt:
                    imp = mkt['implied'][key]
                    r['Book odds'] = f"{mkt['odds'][key]:.2f}"
                    r['Book %'] = f"{imp:.0%}"
                    edge = p[key] - imp
                    r['Edge'] = f"{edge:+.0%}"
                    r['Value'] = '✅' if edge > 0.03 else ''
                rows.append(r)
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)
            if mkt:
                st.caption("Edge = model − bookmaker implied probability. "
                           "✅ marks a model edge over 3% (treat as a *lean*, "
                           "not a sure thing — the book is sharp).")
            else:
                st.caption("No live odds for this fixture in the feed — "
                           "showing model probabilities only.")

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
            _append_line(gm, home_p, away_p,
                         (pred.get('market') or {}).get('odds'))
            st.success(f"Added **{home_p} v {away_p}** to the "
                       f"{gm.capitalize()} coupon — see 🎟️ Toto.")

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
            st.session_state.fix_rows = today_mod.club_section(days)

    frows = st.session_state.get('fix_rows')
    if frows is not None:
        if not frows:
            st.info("No club fixtures with odds in this window "
                    "(leagues may be on a break).")
        else:
            df = pd.DataFrame(frows)
            for c in ['P(1)', 'P(X)', 'P(2)', 'Conf', 'P(O2.5)', 'P(BTTS)',
                      'Edge']:
                if c in df.columns:
                    df[c] = (df[c] * 100).round(0)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Anchored = 1X2 blended with live odds. Edge = model − "
                       "market on the model's pick (only meaningful where the "
                       "model beats the book).")
            st.download_button("Download CSV", df.to_csv(index=False),
                               "fixtures.csv")

            add_rows = [{
                '_label': f"{r['Date']} · {r['Home']} - {r['Away']}",
                'home': r['Home'], 'away': r['Away'],
                'odds': _parse_slash_odds(r.get('Odds(1/X/2)')),
            } for r in frows]
            _add_controls(add_rows, 'fix')


# ============================================================================
# TAB 3 — World Cup / internationals
# ============================================================================
with tab_wc:
    wc_days = st.slider("Days ahead ", 1, 21, 5, key="wcdays")
    wca, wcb = st.columns([1, 1])
    load_wc = wca.button("Load World Cup matches", type="primary")
    if wcb.button("🔄 Refresh results + fixtures", key="wc_refresh",
                  help="Pull the latest international results and upcoming "
                       "fixtures. Knockout matches only appear here once the "
                       "group stage finishes and the bracket is set."):
        from scripts import international as intl_u
        try:
            with st.spinner("Downloading latest international results…"):
                intl_u.cmd_update(None)
            st.success("Data refreshed — now click **Load World Cup matches**.")
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
    st.caption("Two persistent coupons — **Turkish** and **German**. They're "
               "saved to disk, so they survive reloads and new browser tabs "
               "until you clear them. Paste matches (one `Home - Away` per "
               "line, optional odds after), or push them in from the other "
               "tabs. No odds → the model fills it in (club or World Cup).")
    cset = st.columns(3)
    game = cset[0].radio("Game", list(toto.GAMES), horizontal=True,
                         format_func=lambda g: g.capitalize(), key='toto_game')
    n_exp, threshold, top_tier = toto.GAMES[game]
    budget = cset[1].number_input("System budget (columns)", 1, 100000, 1,
                                  help="1 = single column. Higher = cover "
                                       "toss-ups with doubles/triples.")
    cset[2].metric("Coupon", f"{n_exp} matches · prize {threshold}+")

    key = f'coupon_{game}'
    if key not in st.session_state:
        st.session_state[key] = _load_coupon_file(game)
    st.button("🗑️ Clear this coupon", on_click=_clear_game, args=(game,))

    text = st.text_area(
        "Matches (one per line)", key=key, height=320,
        placeholder="Norway - Italy\nTurkey - Spain  1.95 3.40 3.90\n"
                    "Brazil - Morocco\n…")
    _save_coupon_file(game, st.session_state[key])      # persist edits to disk

    parsed = toto.parse_lines(text)
    st.caption(f"Parsed **{len(parsed)}** matches (this game wants {n_exp}).")

    if st.button("Analyze coupon", type="primary"):
        if parsed.empty:
            st.info("Paste some matches first — one `Home - Away` per line.")
        else:
            ctx = {'hist': hist, 'team_stats': team_stats,
                   'team_to_league': team_to_league,
                   'teams': set(team_to_league),
                   'weights': toto._load_blend_weights()}
            out, sorted_probs, unmatched = [], [], []
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
                out.append({'Match': f"{r['home']} v {r['away']}",
                            '1': round(p[0] * 100), 'X': round(p[1] * 100),
                            '2': round(p[2] * 100),
                            'Pick': toto.OUTCOMES[order[0]],
                            'Fair': round(1 / p[order[0]], 2)
                            if p[order[0]] > 0 else None,
                            'Src': src, 'Note': flag})
                sorted_probs.append(np.sort(p)[::-1])

            st.dataframe(pd.DataFrame(out), use_container_width=True,
                         hide_index=True)
            st.caption("Fair = fair decimal odds for the pick (1 ÷ probability) "
                       "— bet it only if a book pays more. Src: **blend** "
                       "(odds+model) · **odds** · **model** (club) · **intl** "
                       "(World Cup model) · **no data** (defaulted to 1∕3 each).")
            if unmatched:
                st.warning("Couldn't match — defaulted to 1/3 each. Check the "
                           "spelling, or add odds (`… 2.10 3.30 3.40`):\n\n- "
                           + "\n- ".join(unmatched))

            q_single = [sp[0] for sp in sorted_probs]
            d = toto.pb_distribution(q_single)
            st.markdown(f"**Single column** — expected correct ≈ "
                        f"{sum(q_single):.1f}/{len(parsed)}")
            tiers = st.columns(min(4, top_tier - threshold + 1))
            for k, t in enumerate(range(threshold, top_tier + 1)):
                if t < len(d) and k < len(tiers):
                    tiers[k].metric(f"P(≥{t})", f"{d[t:].sum():.1%}")

            if budget > 1:
                cov, cols, p_thr = toto.optimize_system(
                    sorted_probs, threshold, int(budget))
                st.markdown(f"**Best system in {budget} columns** — uses "
                            f"{cols} columns; **P(≥{threshold}) = {p_thr:.1%}** "
                            f"(vs {d[threshold:].sum():.1%} single)")
                ups = [(i, cov[i]) for i in range(len(cov)) if cov[i] > 1]
                if ups:
                    st.markdown("Cover these least-predictable matches:")
                    for i, c in ups:
                        kind = "**TRIPLE** (play 1, X and 2)" if c == 3 \
                            else "**double** (top 2 outcomes)"
                        st.markdown(f"- {out[i]['Match']} → {kind}")


st.caption(f"Loaded {len(team_to_league)} teams · {len(leagues)} leagues · "
           f"data through {hist['Date'].max().date()} · "
           f"{dt.date.today()}")
