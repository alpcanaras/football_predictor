// The full card for one match: v1, v2 and the market side by side, goal
// markets, fair odds. Opened by clicking a fixture or a coupon row.
import { h, pct, fair, probBar, legend, badge, modal } from './ui.js';
import { api, store } from './api.js';

export async function showDetails(home, away, coupon, { odds } = {}) {
  const body = h('div', {}, h('div', { class: 'blank' }, h('span', { class: 'spin-sm' }), ' Loading…'));
  const m = modal('Match details', body, [], { wide: true });
  try {
    const d = await api.get(`/api/match?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);
    body.replaceChildren(render(d, coupon, m, odds));
  } catch (e) {
    body.replaceChildren(h('div', { class: 'warnbox' }, `Could not load: ${e.message}`));
  }
}

function render(d, coupon, m, oddsIn) {
  if (!d.used) {
    return h('div', {}, h('b', {}, 'No prediction for this pair.'), h('p', { class: 'small muted' }, 'One of the names is not recognised — close this and fix it on the coupon.'));
  }
  const home = d.home_display || d.home; const away = d.away_display || d.away;
  const k = d.used.indexOf(Math.max(...d.used));
  const lines = [];
  const add = (label, p, used, extra) => { if (p) lines.push(h('div', { class: 'cmp-row' }, h('div', { class: `lbl${used ? ' used' : ''}` }, label), probBar(p), h('div', { class: 'fairs' }, extra || `fair ${p.map(fair).join(' · ')}`))); };
  let src = d.kind;
  if (d.kind === 'club') {
    src = d.market ? 'odds' : d.model_version;
    add('v2', d.v2, !d.market && d.model_version === 'v2');
    add('v1', d.v1, !d.market && d.model_version === 'v1');
    if (d.market) add('Odds', d.market, true, `${d.odds.map((x) => x.toFixed(2)).join(' · ')}`);
  } else add(d.kind === 'intl' ? 'Nations' : 'Club Elo', d.used, true);

  const g = d.goals_used; const g1 = d.goals_v1; const g2 = d.goals_v2;
  const cmp = (key, fmt) => (g1 && g2 ? `v1 ${fmt(g1[key])} · v2 ${fmt(g2[key])}` : '');
  const xg = (v) => (v ? `${v[0].toFixed(1)}–${v[1].toFixed(1)}` : '—');
  const metrics = g ? h('div', { class: 'metrics' },
    ...[['Over 1.5', 'ou15'], ['Over 2.5', 'ou25'], ['Over 3.5', 'ou35'], ['Both score', 'btts']].map(([t, key]) => h('div', { class: 'metric' },
      h('div', { class: 't' }, t), h('div', { class: 'v' }, pct(g[key])), h('div', { class: 's' }, cmp(key, (x) => pct(x)) || (g[key] ? `fair ${fair(g[key])}` : '')))),
    h('div', { class: 'metric' }, h('div', { class: 't' }, 'Expected goals'), h('div', { class: 'v' }, xg(g.xg)), h('div', { class: 's' }, cmp('xg', xg)))) : '';
  const age = d.data_age_days;
  const stale = age != null && age > 28 ? h('div', { class: 'warnbox', style: { marginTop: '10px' } }, `${d.league_name} data is ${age} days old — between seasons or not published yet, so form is out of date.`) : '';
  const onCoupon = coupon.has(d.home, d.away);
  const toggle = h('button', { class: `btn ${onCoupon ? '' : 'primary'}`, onclick: () => {
    const row = coupon.find(d.home, d.away);
    if (row) coupon.remove(row); else coupon.add({ home: d.home, away: d.away, odds: oddsIn || d.odds });
    m.close();
  } }, onCoupon ? '✓ On the coupon — remove' : '＋ Add to coupon');
  return h('div', { class: 'details' },
    h('div', { class: 'mhead' }, h('h2', {}, `${home} – ${away}`), h('span', { class: 'muted' }, d.league_name || ''), badge(src), h('span', { class: 'spacer' }), toggle),
    h('div', { class: 'row', style: { margin: '12px 0 2px' } },
      h('div', {}, h('div', { class: 'muted small' }, 'Most likely'),
        h('div', { class: 'big' }, [`${home} win`, 'Draw', `${away} win`][k], h('small', {}, `  ${pct(d.used[k])} · fair odds ${fair(d.used[k])}`))),
      h('span', { class: 'spacer' }), legend()),
    ...lines, metrics, stale,
    h('div', { class: 'note' }, d.market
      ? 'Odds were found for this match, so they are used: they beat both models. The model rows show where the models disagree — disagreement, not value.'
      : d.kind === 'club' ? `No odds in the feed — using model ${d.model_version}.` : ''));
}
