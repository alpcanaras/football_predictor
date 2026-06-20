"""
Football Predictor — web app
=============================
A point-and-click interface. No notebooks, no cells.

    pip install streamlit          # one-time
    streamlit run app.py           # opens in your browser

Three tabs:
  * Match      — pick a league + two teams, get the full prediction card
  * Fixtures   — every upcoming match with models, anchored to live odds
  * World Cup  — today's / upcoming national-team matches
"""

import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from scripts import config, data_loader, utils
from scripts import predict as predict_mod

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")


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


def prob_bar(label, p):
    st.markdown(f"**{label}** — {p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))


def _blank_coupon(n):
    return pd.DataFrame({'home': [''] * n, 'away': [''] * n,
                         'o1': [None] * n, 'ox': [None] * n, 'o2': [None] * n})


def _add_to_coupon(home, away, pred):
    """Drop a match (with live odds if the prediction carries them) into the
    Toto coupon — fills the first blank row, else appends."""
    if 'coupon' not in st.session_state:
        st.session_state.coupon = _blank_coupon(15)
        st.session_state.coupon_ver = 0
    odds = ((pred or {}).get('market') or {}).get('odds') or {}
    new = {'home': home, 'away': away, 'o1': odds.get('home'),
           'ox': odds.get('draw'), 'o2': odds.get('away')}
    df = st.session_state.coupon.copy()
    blanks = df.index[df['home'].astype(str).str.strip() == '']
    if len(blanks):
        df.loc[blanks[0]] = new
    else:
        df.loc[len(df)] = new
    st.session_state.coupon = df
    st.session_state.coupon_ver = st.session_state.get('coupon_ver', 0) + 1


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

        if st.button("➕ Add to Toto coupon"):
            _add_to_coupon(home_p, away_p, pred)
            st.success(f"Added **{home_p} v {away_p}** to the Toto coupon — "
                       "see the 🎟️ tab.")

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
            rows = today_mod.club_section(days)
        if not rows:
            st.info("No club fixtures with odds in this window "
                    "(leagues may be on a break).")
        else:
            df = pd.DataFrame(rows)
            for c in ['P(1)', 'P(X)', 'P(2)', 'Conf', 'P(O2.5)', 'P(BTTS)', 'Edge']:
                if c in df.columns:
                    df[c] = (df[c] * 100).round(0)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Anchored = 1X2 blended with live odds. Edge = model − "
                       "market on the model's pick (only meaningful where the "
                       "model beats the book).")
            st.download_button("Download CSV", df.to_csv(index=False),
                               "fixtures.csv")


# ============================================================================
# TAB 3 — World Cup / internationals
# ============================================================================
with tab_wc:
    wc_days = st.slider("Days ahead ", 1, 21, 5, key="wcdays")
    if st.button("Load World Cup matches", type="primary"):
        try:
            from scripts import international as intl
            if not os.path.exists(intl.WC_FIXTURES_FILE):
                st.warning("No WC fixtures file. Run: "
                           "python scripts/international.py update")
            else:
                fx = pd.read_csv(intl.WC_FIXTURES_FILE, parse_dates=['date'])
                now = pd.Timestamp.now().normalize()
                fx = fx[(fx['date'] >= now)
                        & (fx['date'] <= now + pd.Timedelta(days=wc_days))]
                if fx.empty:
                    st.info("No WC matches in this window.")
                else:
                    rdf = intl.load_results()
                    ratings, h = intl.run_elo(rdf)
                    model = intl.GoalModel().fit(h)
                    out = []
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
                    st.dataframe(pd.DataFrame(out), use_container_width=True,
                                 hide_index=True)
        except Exception as e:
            st.error(f"World Cup section error: {e}")

# ============================================================================
# TAB 4 — Toto coupon optimizer
# ============================================================================
with tab_toto:
    import numpy as np
    from scripts import toto

    st.caption("Enter this week's coupon. Bookmaker 1X2 odds (o1/ox/o2) give "
               "the most accurate probabilities; where odds are blank the model "
               "fills in — club teams via the league ensembles, national teams "
               "(World Cup) via the international model. You can also push a "
               "match here from the 🎯 Match tab.")
    cset = st.columns(3)
    game = cset[0].radio("Game", list(toto.GAMES), horizontal=True,
                         format_func=lambda g: g.capitalize())
    n_exp, threshold, top_tier = toto.GAMES[game]
    budget = cset[1].number_input("System budget (columns)", 1, 100000, 1,
                                  help="1 = single column. Higher = cover "
                                       "toss-ups with doubles/triples.")
    cset[2].metric("Coupon", f"{n_exp} matches · prize {threshold}+")

    if 'coupon' not in st.session_state:
        st.session_state.coupon = _blank_coupon(n_exp)
        st.session_state.coupon_ver = 0

    bc1, bc2, _ = st.columns([1, 1, 5])
    if bc1.button("➕ Row"):
        df = st.session_state.coupon.copy()
        df.loc[len(df)] = {'home': '', 'away': '', 'o1': None,
                           'ox': None, 'o2': None}
        st.session_state.coupon = df
        st.session_state.coupon_ver += 1
    if bc2.button("🗑️ Clear"):
        st.session_state.coupon = _blank_coupon(n_exp)
        st.session_state.coupon_ver += 1

    edited = st.data_editor(st.session_state.coupon, num_rows="dynamic",
                            use_container_width=True,
                            key=f"toto_ed_{st.session_state.coupon_ver}")
    st.session_state.coupon = edited

    if st.button("Analyze coupon", type="primary"):
        ctx = {'hist': hist, 'team_stats': team_stats,
               'team_to_league': team_to_league,
               'teams': set(team_to_league),
               'weights': toto._load_blend_weights()}
        rows = edited.dropna(subset=['home', 'away'])
        rows = rows[(rows['home'].astype(str).str.strip() != '')]
        if rows.empty:
            st.info("Add some matches first.")
        else:
            out, sorted_probs = [], []
            for _, r in rows.iterrows():
                p, src, model = toto.match_probs(r, ctx)
                if p is None:
                    p = np.array([0.34, 0.33, 0.33]); src = 'no data'
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

            q_single = [sp[0] for sp in sorted_probs]
            d = toto.pb_distribution(q_single)
            st.markdown(f"**Single column** — expected correct ≈ "
                        f"{sum(q_single):.1f}/{len(rows)}")
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
