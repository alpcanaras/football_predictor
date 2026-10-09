// The full card for one match. Opened by clicking a fixture or a coupon row.
import { h, pct, fair, modal, SRC_TEXT } from './ui.js';
import { icon } from './icons.js';
import { api, store } from './api.js';

export async function showDetails(home, away, coupon, { odds, ou_odds } = {}) {
  const body = h('div', {}, h('div', { class: 'blank' }, h('span', { class: 'spin-sm' }), ' Loading…'));
  const m = modal('Match', body, [], { wide: true });
  try {
    const d = await api.post('/api/match', { home, away, odds: odds || null, ou_odds: ou_odds || null });
    body.replaceChildren(render(d, coupon, m, odds));
  } catch (e) {
    body.replaceChildren(h('div', { class: 'box warn' }, `Could not load: ${e.message}`));
  }
}

function render(d, coupon, m, oddsIn) {
  if (!d.ok) {
    return h('div', {}, h('b', {}, 'No prediction for this pair.'),
      h('p', { class: 'small muted', style: { marginTop: '6px' } }, 'One of the names is not recognised. Close this and correct it on the coupon.'));
  }
  const home = d.home_display; const away = d.away_display;
  const p = d.p1x2; const top = p.indexOf(Math.max(...p));
  const onCoupon = coupon.has(d.home, d.away);
  const toggle = h('button', { class: `btn ${onCoupon ? '' : 'primary'}`, onclick: () => {
    const row = coupon.find(d.home, d.away);
    if (row) coupon.remove(row); else coupon.add({ home: d.home, away: d.away, odds: oddsIn || d.odds });
    m.close();
  } }, icon(onCoupon ? 'check' : 'plus', 15), onCoupon ? 'On the coupon · remove' : 'Add to coupon');
  const src = SRC_TEXT[d.src] || d.src;

  const cell = (i, label) => {
    const model = d.model_1x2 && d.market_1x2 ? d.model_1x2[i] : null;
    return h('div', { class: `res${i === top ? ' top' : ''}` },
      h('div', { class: 't' }, label),
      h('div', { class: 'v' }, pct(p[i])),
      h('div', { class: 's' }, `fair odds ${fair(p[i])}`),
      model != null ? h('div', { class: 's' }, `model alone ${pct(model)}`) : '');
  };
  const goalsRow = h('div', { class: 'metrics' },
    ...[['Over 1.5', d.ou['1.5']], ['Over 2.5', d.ou['2.5']], ['Over 3.5', d.ou['3.5']], ['Both teams score', d.btts]].map(([t, v]) => h('div', { class: 'metric' },
      h('div', { class: 't' }, t), h('div', { class: 'v' }, pct(v)))),
    h('div', { class: 'metric' }, h('div', { class: 't' }, 'Expected goals'),
      h('div', { class: 'v' }, `${d.xg[0].toFixed(1)} – ${d.xg[1].toFixed(1)}`)));

  const mx = Math.max(...d.grid.flat());
  const heat = h('table', { class: 'heat', 'aria-label': 'Score probabilities' },
    h('tr', {}, h('th', {}), ...[0, 1, 2, 3, 4].map((j) => h('th', {}, j))),
    ...d.grid.map((row, i) => h('tr', {}, h('th', {}, i),
      ...row.map((v) => h('td', { style: { background: `color-mix(in srgb, var(--accent) ${Math.round(70 * v / mx)}%, var(--surface-2))`,
        color: v / mx > 0.55 ? 'var(--on-accent)' : 'var(--text)', fontWeight: v / mx > 0.55 ? 650 : 400 } }, v >= 0.005 ? Math.round(v * 100) : '·')))));
  const tbl = h('table', { class: 'scoretbl' },
    h('tr', {}, h('th', {}, 'Score'), h('th', {}, 'Chance'), h('th', {}, 'Fair odds')),
    ...d.scores.map((s) => h('tr', {}, h('td', {}, h('b', {}, s.score.replace('-', ' – '))), h('td', {}, pct(s.p, 1)), h('td', {}, fair(s.p)))));

  const age = d.data_age_days;
  const warn = age != null && age > 28
    ? h('div', { class: 'box warn', style: { marginTop: '14px' } }, `${d.league_name} data is ${age} days old. The league is between seasons or the new season is not published yet, so recent form is missing.`) : '';
  const how = {
    odds: 'The bookmaker odds are used for the result, because on matches the models had not seen the odds were more accurate. Over/under combines the model with the over/under odds when they exist.',
    model: 'No odds were found, so the model gives the result. The score table and the over/under numbers come from the same scoreline model and agree with it.',
    national: 'National teams are priced from international ratings and a goals model.',
    'club elo': 'These clubs play in different leagues, so they are priced from global club ratings.',
  }[d.src] || '';
  return h('div', {},
    h('div', { class: 'd-head' }, h('h2', {}, `${home} v ${away}`), h('span', { class: `tag ${d.src === 'odds' ? 'odds' : ''}` }, src),
      h('span', { class: 'spacer' }), toggle),
    h('div', { class: 'd-sub' }, [d.league_name, d.odds ? `odds ${d.odds.map((o) => o.toFixed(2)).join(' / ')}` : null].filter(Boolean).join('  ·  ')),
    h('div', { class: 'sect' }, h('h4', {}, 'Result'), h('div', { class: 'res3' }, cell(0, home), cell(1, 'Draw'), cell(2, away))),
    h('div', { class: 'sect' }, h('h4', {}, 'Goals'), goalsRow),
    h('div', { class: 'sect' }, h('h4', {}, 'Likely scores'),
      h('div', { class: 'scorewrap' }, tbl, h('div', {}, heat, h('div', { class: 'axis small muted', style: { textAlign: 'center', marginTop: '6px' } }, `${home} goals down, ${away} goals across. Cells show %.`)))),
    warn, h('div', { class: 'note' }, how));
}
