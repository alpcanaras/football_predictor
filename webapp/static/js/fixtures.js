// Fixtures: every upcoming club match in the odds feed, filterable, with v1 /
// v2 / market side by side and one-click "add to coupon".
import { h, pct, fold, fmtDate, probBar, badge, legend } from './ui.js';
import { api, store } from './api.js';
import { addToCoupon } from './toto.js';

function served(f) {
  if (f.market) return [f.market, 'odds'];
  const mode = store.status.model_mode;
  const v2 = mode === 'v2' || (mode === 'auto' && store.status.v2_auto_leagues.includes(f.league));
  if (v2 && f.v2) return [f.v2, 'v2'];
  return [f.v1, 'v1'];
}
const mini = (p) => (p ? p.map((x) => Math.round(x * 100)).join('·') : '—');

export class FixturesPage {
  constructor(root) {
    this.root = root; this.days = Number(localStorage.getItem('fp.fixdays') || 3);
    this.q = ''; this.leagues = new Set(); this.data = null;
    this.unsub = store.on((w) => { if (w === 'mode') this.load(true); if (w === 'compare') this.paint(); });
    this.build(); this.load();
  }
  destroy() { this.unsub(); }

  build() {
    this.dayBtns = [1, 3, 7, 14].map((d) => h('button', { onclick: () => { this.days = d; localStorage.setItem('fp.fixdays', d); this.load(); } }, d === 1 ? 'Today' : `${d} days`));
    const search = h('input', { class: 'txt', placeholder: 'Search team or league…' });
    search.addEventListener('input', () => { this.q = fold(search.value); this.paint(); });
    this.info = h('span', { class: 'small muted' });
    this.lgBox = h('div', { class: 'lgchips' });
    this.list = h('div');
    this.root.replaceChildren(
      h('div', { class: 'page-head' }, h('h1', {}, 'Upcoming fixtures'), this.info),
      h('div', { class: 'filters' }, h('div', { class: 'seg' }, ...this.dayBtns), search, h('span', { class: 'spacer' }), legend()),
      this.lgBox, this.list);
  }

  async load(force = false) {
    this.dayBtns.forEach((b, i) => b.classList.toggle('on', [1, 3, 7, 14][i] === this.days));
    this.list.replaceChildren(h('div', { class: 'empty' }, h('span', { class: 'spin-sm' }), ' Loading the odds feed and predicting…'));
    try {
      this.data = await api.get(`/api/fixtures?days=${this.days}${force ? `&t=${Date.now()}` : ''}`);
      this.paint();
    } catch (e) { this.list.replaceChildren(h('div', { class: 'empty' }, `Could not load fixtures: ${e.message}`)); }
  }

  paint() {
    if (!this.data) return;
    const all = this.data.fixtures;
    const lgs = [...new Map(all.map((f) => [f.league, f.league_name])).entries()].sort((a, b) => a[1].localeCompare(b[1]));
    this.lgBox.replaceChildren(...lgs.map(([k, name]) => h('button', {
      class: `chip${this.leagues.has(k) ? ' on' : ''}`,
      onclick: () => { if (this.leagues.has(k)) this.leagues.delete(k); else this.leagues.add(k); this.paint(); },
    }, `${name} · ${all.filter((f) => f.league === k).length}`)));
    const shown = all.filter((f) => (!this.leagues.size || this.leagues.has(f.league))
      && (!this.q || fold(`${f.home} ${f.away} ${f.league_name}`).includes(this.q)));
    this.info.textContent = `${shown.length} of ${all.length} matches · next ${this.days} day${this.days > 1 ? 's' : ''}`;
    if (!all.length) {
      this.list.replaceChildren(h('div', { class: 'card empty' }, 'No club fixtures in the feed for this window — leagues may be on an international break. The feed publishes a few days ahead.'));
      return;
    }
    const byDay = new Map();
    shown.forEach((f) => { if (!byDay.has(f.date)) byDay.set(f.date, []); byDay.get(f.date).push(f); });
    const head = h('div', { class: 'frow head' }, h('div', {}, 'League'), h('div', {}, 'Match'), h('div', {}, 'Used 1 · X · 2'), h('div', {}, 'From'),
      h('div', {}, store.compare ? 'v1' : ''), h('div', {}, store.compare ? 'v2' : ''), h('div', {}, 'O2.5 · BTTS'), h('div', {}, 'Add'));
    const blocks = [];
    for (const [day, fs] of byDay) {
      blocks.push(h('div', { class: 'day' }, fmtDate(day)));
      blocks.push(h('div', { class: 'card' }, head.cloneNode(true), ...fs.map((f) => this.rowEl(f))));
    }
    this.list.replaceChildren(...blocks);
  }

  rowEl(f) {
    const [p, src] = served(f);
    const mode = store.status.model_mode;
    const goalsV2 = mode === 'v2' || (mode === 'auto' && store.status.v2_auto_leagues.includes(f.league));
    const g = (goalsV2 ? f.goals_v2 : f.goals_v1) || {};
    const odds = f.odds ? `${f.odds.map((x) => x.toFixed(2)).join(' / ')}` : '';
    return h('div', { class: 'frow' },
      h('div', { class: 'lg', title: f.league_name }, f.league_name),
      h('div', {}, h('div', { class: 'teams' }, `${f.home} – ${f.away}`), odds ? h('div', { class: 'tiny faint num' }, `odds ${odds}`) : ''),
      probBar(p),
      badge(src),
      h('div', { class: 'mini' }, store.compare && f.v1 ? h('span', {}, h('b', {}, 'v1 '), mini(f.v1)) : ''),
      h('div', { class: 'mini' }, store.compare && f.v2 ? h('span', {}, h('b', {}, 'v2 '), mini(f.v2)) : ''),
      h('div', { class: 'mini' }, `${pct(g.ou25 ?? f.ou25)} · ${pct(g.btts ?? f.btts)}`),
      h('div', { class: 'addbtns' },
        ...Object.entries(store.status.games).map(([game, info]) => h('button', {
          class: 'btn sm', title: `Add to the ${info.label} coupon`,
          onclick: () => addToCoupon(game, f.home, f.away, f.odds),
        }, `+${info.flag}`))));
  }
}
