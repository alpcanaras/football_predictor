// Internationals: upcoming national-team fixtures when the source lists them,
// and a quick "any two nations" pricer (the Toto coupon uses the same model).
import { h, pct, fair, fmtDate, probBar, legend, Autocomplete } from './ui.js';
import { api, store } from './api.js';
import { addToCoupon } from './toto.js';

export class IntlPage {
  constructor(root) { this.root = root; this.build(); this.load(); }
  destroy() {}

  build() {
    const nat = (t) => t.kind === 'national';
    const mk = (ph) => { const i = h('input', { class: 'txt', placeholder: ph }); const w = h('div', {}, i); new Autocomplete(i, { source: () => store.teams, filter: nat, onPick: () => this.quick() }); return [i, w]; };
    const [hi, hw] = mk('Home nation'); const [ai, aw] = mk('Away nation');
    this.hi = hi; this.ai = ai;
    this.neutral = h('input', { type: 'checkbox' });
    this.quickOut = h('div');
    this.list = h('div');
    this.info = h('span', { class: 'small muted' });
    this.root.replaceChildren(
      h('div', { class: 'page-head' }, h('h1', {}, 'Internationals'), this.info),
      h('div', { class: 'card card-pad', style: { marginBottom: '14px' } },
        h('div', { class: 'small muted', style: { marginBottom: '8px' } }, 'Price any two national teams (Nations League, qualifiers, friendlies on the Spor Toto coupon).'),
        h('div', { class: 'picker' }, hw, h('label', { class: 'small row' }, this.neutral, 'neutral'), aw,
          h('button', { class: 'btn primary', onclick: () => this.quick() }, 'Price it')),
        this.quickOut),
      this.list);
  }

  async quick() {
    const home = this.hi.value.trim(); const away = this.ai.value.trim();
    if (!home || !away) return;
    const r = await api.post('/api/analyze', { home, away });
    if (!r.probs) { this.quickOut.replaceChildren(h('div', { class: 'warnbox', style: { marginTop: '10px' } }, 'Not recognised — pick the names from the list.')); return; }
    this.quickOut.replaceChildren(h('div', { class: 'cmp', style: { marginTop: '12px' } }, h('div', { class: 'lbl' }, `${home} – ${away}`), probBar(r.probs), h('div', { class: 'fair' }, `fair ${r.probs.map(fair).join(' · ')}`)),
      h('div', { class: 'row' }, h('span', { class: 'spacer' }), ...Object.entries(store.status.games).map(([g, info]) => h('button', { class: 'btn sm', onclick: () => addToCoupon(g, home, away) }, `+ ${info.flag} ${info.label}`))));
  }

  async load() {
    this.list.replaceChildren(h('div', { class: 'empty' }, h('span', { class: 'spin-sm' }), ' Loading…'));
    try {
      const d = await api.get('/api/internationals?days=21');
      this.info.textContent = `results through ${d.results_through}`;
      if (!d.fixtures.length) {
        this.list.replaceChildren(h('div', { class: 'card empty' }, 'The results source lists no upcoming national-team fixtures right now. Use the pricer above — or 📡 Data → Update internationals later.'));
        return;
      }
      this.list.replaceChildren(h('div', { class: 'row', style: { marginBottom: '8px' } }, h('span', { class: 'spacer' }), legend()),
        h('div', { class: 'card' }, ...d.fixtures.map((f) => h('div', { class: 'frow', style: { gridTemplateColumns: '120px minmax(200px,1.5fr) minmax(180px,1.2fr) 160px 110px 92px' } },
          h('div', { class: 'lg' }, fmtDate(f.date)),
          h('div', {}, h('div', { class: 'teams' }, `${f.home} – ${f.away}`), h('div', { class: 'tiny faint' }, f.tournament)),
          probBar(f.probs), h('div', { class: 'mini' }, `fair ${f.probs.map(fair).join(' · ')}`),
          h('div', { class: 'mini' }, `O2.5 ${pct(f.ou25)} · BTTS ${pct(f.btts)}`),
          h('div', { class: 'addbtns' }, ...Object.entries(store.status.games).map(([g, info]) => h('button', { class: 'btn sm', onclick: () => addToCoupon(g, f.home, f.away) }, `+${info.flag}`)))))));
    } catch (e) { this.list.replaceChildren(h('div', { class: 'empty' }, `Failed: ${e.message}`)); }
  }
}
