// Match: any two teams -> the full card (v1 / v2 / market, goals, fair odds).
import { h, pct, fair, probBar, badge, legend, Autocomplete } from './ui.js';
import { api, store } from './api.js';
import { addToCoupon } from './toto.js';

export class MatchPage {
  constructor(root) {
    this.root = root;
    const last = JSON.parse(localStorage.getItem('fp.match') || '{}');
    this.home = last.home || 'Galatasaray'; this.away = last.away || 'Fenerbahce';
    this.unsub = store.on((w) => { if (w === 'mode' || w === 'compare') this.predict(); });
    this.build(); this.predict();
  }
  destroy() { this.unsub(); }

  build() {
    const mk = (val, ph) => {
      const inp = h('input', { class: 'txt', value: val, placeholder: ph, spellcheck: false });
      const w = h('div', {}, inp);
      new Autocomplete(inp, { source: () => store.teams, onPick: () => this.predict() });
      inp.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !w.querySelector('.ac-list')) this.predict(); });
      return [inp, w];
    };
    const [hIn, hw] = mk(this.home, 'Home team');
    const [aIn, aw] = mk(this.away, 'Away team');
    this.hIn = hIn; this.aIn = aIn;
    const swap = h('button', { class: 'iconbtn', title: 'Swap home and away', onclick: () => { [this.hIn.value, this.aIn.value] = [this.aIn.value, this.hIn.value]; this.predict(); } }, '⇄');
    this.card = h('div', { class: 'matchcard' });
    this.root.replaceChildren(
      h('div', { class: 'page-head' }, h('h1', {}, 'Match predictor'), h('span', { class: 'small muted' }, 'Any two clubs or national teams')),
      h('div', { class: 'card card-pad' }, h('div', { class: 'picker' }, hw, swap, aw,
        h('button', { class: 'btn primary', onclick: () => this.predict() }, 'Predict'))),
      this.card);
  }

  async predict() {
    const home = this.hIn.value.trim(); const away = this.aIn.value.trim();
    if (!home || !away) return;
    localStorage.setItem('fp.match', JSON.stringify({ home, away }));
    this.card.classList.add('busy');
    try {
      const m = await api.get(`/api/match?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);
      this.render(m);
    } catch (e) {
      this.card.replaceChildren(h('div', { class: 'card empty' }, `Could not predict: ${e.message}`));
    } finally { this.card.classList.remove('busy'); }
  }

  render(m) {
    if (!m.used) {
      const sug = ['home', 'away'].flatMap((s) => (m.known?.[s] ? [] : [h('div', {}, `“${m[s]}” not recognised`, ...(m.suggest?.[s] || []).map((o) => h('button', { class: 'chip', style: { marginLeft: '6px' }, onclick: () => { (s === 'home' ? this.hIn : this.aIn).value = o; this.predict(); } }, o)))]));
      this.card.replaceChildren(h('div', { class: 'card card-pad' }, h('b', {}, 'No prediction for this pair.'), ...sug));
      return;
    }
    const lines = [];
    const add = (label, p, used, extra) => { if (p) lines.push(h('div', { class: 'cmp' }, h('div', { class: `lbl${used ? ' used' : ''}` }, label), probBar(p), h('div', { class: 'fair' }, extra || `fair ${p.map(fair).join(' · ')}`))); };
    let src = m.kind;
    if (m.kind === 'club') {
      src = m.market ? 'feed' : m.model_version;
      if (store.compare) {
        add('v1', m.v1, !m.market && m.model_version === 'v1');
        add('v2', m.v2, !m.market && m.model_version === 'v2');
      } else add(`Model ${m.model_version}`, m.model_version === 'v2' ? m.v2 : m.v1, !m.market);
      if (m.market) add('Market', m.market, true, `odds ${m.odds.map((x) => x.toFixed(2)).join(' · ')}`);
    } else {
      add(m.kind === 'intl' ? 'Nations' : 'Euro Elo', m.used, true);
    }
    const g = m.goals_used; const g1 = m.goals_v1; const g2 = m.goals_v2;
    const cmp = (k, fmt) => (store.compare && g1 && g2 ? `v1 ${fmt(g1[k])} · v2 ${fmt(g2[k])}` : '');
    const xg = (v) => (v ? `${v[0].toFixed(1)}–${v[1].toFixed(1)}` : '—');
    const metrics = g ? h('div', { class: 'metrics' },
      ...[['Over 1.5', 'ou15'], ['Over 2.5', 'ou25'], ['Over 3.5', 'ou35'], ['Both score', 'btts']].map(([t, k]) => h('div', { class: 'metric' },
        h('div', { class: 't' }, t), h('div', { class: 'v' }, pct(g[k])), h('div', { class: 's' }, cmp(k, (x) => pct(x)) || `fair ${g[k] ? fair(g[k]) : '—'}`))),
      h('div', { class: 'metric' }, h('div', { class: 't' }, 'Expected goals'), h('div', { class: 'v' }, xg(g.xg)), h('div', { class: 's' }, cmp('xg', xg)))) : '';
    const age = m.data_age_days;
    const stale = age != null && age > 28 ? h('div', { class: 'warnbox', style: { marginTop: '10px' } }, `${m.league_name} data is ${age} days old — between seasons or not published yet; form is out of date.`) : '';
    const used = m.used;
    const k = used.indexOf(Math.max(...used));
    const pickText = [`${m.home} win`, 'Draw', `${m.away} win`][k];
    this.card.replaceChildren(h('div', { class: 'card card-pad' },
      h('div', { class: 'mhead' }, h('h2', {}, `${m.home} – ${m.away}`), h('span', { class: 'muted' }, m.league_name || ''), badge(src), h('span', { class: 'spacer' }),
        ...Object.entries(store.status.games).map(([game, info]) => h('button', { class: 'btn sm', onclick: () => addToCoupon(game, m.home, m.away, m.odds) }, `+ ${info.flag} ${info.label}`))),
      h('div', { class: 'row', style: { margin: '12px 0 4px' } },
        h('div', {}, h('div', { class: 'muted small' }, 'Most likely'), h('div', { class: 'big' }, `${pickText} `, h('small', {}, `${pct(used[k])} · fair odds ${fair(used[k])}`))), h('span', { class: 'spacer' }), legend()),
      ...lines, metrics, stale,
      h('div', { class: 'note' }, m.market
        ? 'Odds were found for this match, so the market is used: it beats both models. The model rows show where they disagree — disagreement, not value.'
        : m.kind === 'club' ? `No odds in the feed — using model ${m.model_version} (the sidebar switch decides which).` : '')));
  }
}
