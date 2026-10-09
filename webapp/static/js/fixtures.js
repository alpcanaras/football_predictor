// The fixtures table: everything in the odds feed, grouped by day, with the
// numbers in columns so they can be scanned. Filters are all visible; the add
// button appears on hover. It repaints in place while matches are still being
// priced, so the page is usable from the first second.
import { h, norm, pct, kickoff, ymd, hhmm, dayLabel, textField, toast, modal, SRC_TEXT } from './ui.js';
import { icon } from './icons.js';
import { api, store } from './api.js';
import { Coupon, pairKey } from './coupon.js';
import { showDetails } from './details.js';

// label, [first day, last day] counted from today
const WINDOWS = {
  today: ['Today', [0, 0]], tomorrow: ['Tomorrow', [1, 1]], d3: ['3 days', [0, 3]], d7: ['7 days', [0, 7]], all: ['All', [0, 14]],
};
const SORTS = { time: 'Kickoff', conf: 'Most likely result', even: 'Closest matches' };
const PRESETS = [
  ['Top 5 in Europe', ['british_pl', 'spanish', 'german', 'italian', 'french']],
  ['Süper Lig', ['turkish']],
  ['Bundesliga 1 + 2', ['german', 'german_2']],
];
const CELLS = ['1', 'X', '2'];

export class FixturesView {
  constructor(root, coupon) {
    this.root = root; this.coupon = coupon;
    const saved = localStorage.getItem('fp.window');
    this.win = WINDOWS[saved] ? saved : 'd3';
    this.sort = localStorage.getItem('fp.sort') || 'time';
    this.oddsOnly = localStorage.getItem('fp.oddsonly') === '1';
    this.q = ''; this.leagues = new Set(); this.showStarted = false;
    this.data = { state: 'loading', rows: [], done: 0, total: 0 };
    this.els = new Map();
    this.timer = null;
    this.unsub = [coupon.on((w) => { if (w === 'rows') this.markAdded(); })];
    this.build();
    this.load();
    this.onKey = (e) => {
      if (e.key === '/' && !/input|textarea|select/i.test(document.activeElement?.tagName || '')) {
        e.preventDefault(); this.search.input.focus();
      }
    };
    document.addEventListener('keydown', this.onKey);
  }
  destroy() { this.unsub.forEach((u) => u()); clearTimeout(this.timer); document.removeEventListener('keydown', this.onKey); }

  // ----------------------------------------------------------------- layout
  build() {
    this.winSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Days' },
      ...Object.entries(WINDOWS).map(([k, [label]]) => h('button', { 'data-k': k, onclick: () => this.setWindow(k) }, label)));
    this.search = textField({ placeholder: 'Search a team or league', cls: '', onInput: (v) => { this.q = norm(v); this.paint(); } });
    this.search.wrap.classList.add('withicon');
    this.search.wrap.prepend(h('span', { class: 'lead' }, icon('search', 15)));
    this.lgBtn = h('button', { class: 'btn', onclick: (e) => this.leagueMenu(e.currentTarget) });
    this.sortSel = h('select', { class: 'txt', 'aria-label': 'Sort', style: { width: '170px' },
      onchange: (e) => { this.sort = e.target.value; localStorage.setItem('fp.sort', this.sort); this.paint(); } },
    ...Object.entries(SORTS).map(([k, v]) => h('option', { value: k, selected: k === this.sort }, `Sort: ${v}`)));
    this.oddsChk = h('input', { type: 'checkbox', checked: this.oddsOnly,
      onchange: (e) => { this.oddsOnly = e.target.checked; localStorage.setItem('fp.oddsonly', this.oddsOnly ? '1' : '0'); this.paint(); } });
    this.count = h('span', { class: 'small muted' });
    this.chips = h('div', { class: 'chips' });
    this.reset = h('button', { class: 'link small', onclick: () => this.resetFilters(), hidden: true }, 'Reset filters');
    this.addAll = h('button', { class: 'btn sm', onclick: () => this.addShown(), hidden: true });
    this.refresh = h('button', { class: 'iconbtn sm', title: 'Reload odds and predictions', onclick: () => this.load(true) }, icon('refresh', 15));
    this.list = h('div');
    const header = h('div', { class: 'fgrid fhead panel', style: { borderBottomLeftRadius: 0, borderBottomRightRadius: 0, borderBottom: '1px solid var(--border)', marginTop: '6px' } },
      h('div', {}, 'Time'), h('div', {}, 'Match'),
      h('div', { class: 'r', title: 'Home win, %' }, '1 %'), h('div', { class: 'r', title: 'Draw, %' }, 'X %'), h('div', { class: 'r', title: 'Away win, %' }, '2 %'),
      h('div', { class: 'r', title: 'Three or more goals, %' }, 'O 2.5 %'), h('div', { class: 'r', title: 'Both teams score, %' }, 'BTTS %'),
      h('div', { class: 'r', title: 'Most likely score' }, 'Score'), h('div'));
    this.root.replaceChildren(
      h('div', { class: 'fx-head' },
        h('div', { class: 'fx-title' }, h('h1', {}, 'Fixtures'), this.count, h('span', { class: 'spacer' }), this.refresh),
        h('div', { class: 'fx-tools' }, this.winSeg, this.search.wrap, this.lgBtn, this.sortSel),
        h('div', { class: 'fx-sub' }, h('label', { class: 'check' }, this.oddsChk, 'Only matches with odds'),
          this.chips, this.reset, h('span', { class: 'spacer' }), this.addAll)),
      header, this.list);
    this.syncWindow();
  }

  setWindow(k) { this.win = k; localStorage.setItem('fp.window', k); this.syncWindow(); this.paint(); }
  syncWindow() { [...this.winSeg.children].forEach((b) => b.classList.toggle('on', b.dataset.k === this.win)); }
  resetFilters() {
    this.q = ''; this.search.set(''); this.leagues.clear(); this.oddsOnly = false; this.oddsChk.checked = false;
    localStorage.setItem('fp.oddsonly', '0'); this.paint();
  }

  // ------------------------------------------------------------------ data
  async load(refresh = false) {
    clearTimeout(this.timer);
    try {
      this.data = await api.get(`/api/fixtures${refresh ? '?refresh=1' : ''}`);
      this.paint();
      if (this.data.state === 'loading') this.timer = setTimeout(() => this.load(), 900);
    } catch (e) {
      if (/warming up/i.test(e.message)) { this.timer = setTimeout(() => this.load(), 1500); return; }
      this.list.replaceChildren(h('div', { class: 'blank panel' }, `Could not load fixtures: ${e.message}`));
    }
  }

  prepared() {
    const now = new Date();
    return this.data.rows.map((f) => {
      const ko = kickoff(f.date, f.time);
      return { f, ko, day: ymd(ko.date), started: ko.hasTime && ko.date < now,
        over: ko.hasTime && ko.date.getTime() + 150 * 60000 < now.getTime() };
    }).filter((x) => !x.over);
  }

  inWindow(x) {
    const [, [a, b]] = WINDOWS[this.win];
    const t = new Date();
    const lo = ymd(new Date(t.getFullYear(), t.getMonth(), t.getDate() + a));
    const hi = ymd(new Date(t.getFullYear(), t.getMonth(), t.getDate() + b));
    return x.day >= lo && x.day <= hi;
  }

  haystack(f) {
    const keys = (n) => (store.byName.get(n)?.keys || []).join(' ');
    return norm(`${f.home_display} ${f.away_display} ${f.league_name} ${keys(f.home)} ${keys(f.away)}`);
  }

  shown() {
    const topP = (f) => (f.probs ? Math.max(...f.probs) : 0);
    let xs = this.prepared().filter((x) => this.inWindow(x)
      && (this.showStarted || !x.started)
      && (!this.leagues.size || this.leagues.has(x.f.league))
      && (!this.oddsOnly || x.f.odds)
      && (!this.q || this.haystack(x.f).includes(this.q)));
    if (this.sort === 'time') xs = xs.sort((a, b) => a.ko.date - b.ko.date || a.f.league_name.localeCompare(b.f.league_name));
    else if (this.sort === 'conf') xs = xs.sort((a, b) => topP(b.f) - topP(a.f));
    else xs = xs.sort((a, b) => (topP(a.f) || 9) - (topP(b.f) || 9));
    return xs;
  }

  // ------------------------------------------------------------------ paint
  paint() {
    const d = this.data;
    const shown = this.shown();
    const priced = d.rows.filter((r) => !r.pending).length;
    const hiddenStarted = this.showStarted ? 0 : this.prepared().filter((x) => x.started && this.inWindow(x)).length;
    this.count.replaceChildren(
      d.state === 'loading' && d.total ? `Pricing ${priced} of ${d.total}…` : `${shown.length} match${shown.length === 1 ? '' : 'es'}`,
      hiddenStarted ? h('span', {}, ' · ', h('button', { class: 'link', onclick: () => { this.showStarted = true; this.paint(); } }, `show ${hiddenStarted} already started`)) : '',
      this.showStarted ? h('span', {}, ' · ', h('button', { class: 'link', onclick: () => { this.showStarted = false; this.paint(); } }, 'hide started')) : '');
    const nLg = this.leagues.size;
    this.lgBtn.replaceChildren(icon('filter', 15), nLg ? `Leagues · ${nLg}` : 'All leagues', icon('chevron', 14));
    this.lgBtn.classList.toggle('on', nLg > 0);
    const names = (k) => this.data.rows.find((r) => r.league === k)?.league_name || k;
    this.chips.replaceChildren(...[...this.leagues].map((k) => h('span', { class: 'chip' }, names(k),
      h('button', { title: 'Remove this filter', 'aria-label': `Remove ${names(k)}`, onclick: () => { this.leagues.delete(k); this.paint(); } }, icon('x', 12)))));
    this.reset.hidden = !(this.q || nLg || this.oddsOnly);
    const addable = shown.filter(({ f }) => !this.coupon.has(f.home, f.away));
    const narrowed = this.q || nLg || this.oddsOnly || this.win === 'today' || this.win === 'tomorrow';
    this.addAll.hidden = !(narrowed && addable.length > 1 && addable.length <= 30);
    this.addAll.replaceChildren(icon('plus', 14), `Add all ${addable.length}`);

    if (!shown.length) {
      const msg = d.state === 'loading' ? 'Loading fixtures and prices…'
        : this.prepared().length ? 'No matches fit these filters.' : 'The odds feed has no fixtures right now. Leagues may be on a break.';
      this.list.replaceChildren(h('div', { class: 'blank panel', style: { borderTopLeftRadius: 0, borderTopRightRadius: 0, borderTop: 0 } },
        d.state === 'loading' ? h('span', { class: 'spin-sm' }) : '', ' ', msg,
        !this.prepared().length || d.state === 'loading' ? '' : h('div', { style: { marginTop: '8px' } },
          h('button', { class: 'link', onclick: () => this.resetFilters() }, 'Reset filters'),
          this.win !== 'all' ? h('span', {}, ' · ', h('button', { class: 'link', onclick: () => this.setWindow('all') }, 'Show all days')) : '')));
      return;
    }
    const blocks = [];
    if (this.sort === 'time') {
      const byDay = new Map();
      shown.forEach((x) => { if (!byDay.has(x.day)) byDay.set(x.day, []); byDay.get(x.day).push(x); });
      let first = true;
      for (const [, xs] of byDay) {
        blocks.push(h('div', { class: 'day' }, h('b', {}, dayLabel(xs[0].ko.date)), `${xs.length} match${xs.length > 1 ? 'es' : ''}`));
        blocks.push(h('div', { class: 'panel fx-table', style: first ? { borderTopLeftRadius: 0, borderTopRightRadius: 0, borderTop: 0 } : {} }, ...xs.map((x) => this.rowEl(x))));
        first = false;
      }
    } else {
      blocks.push(h('div', { class: 'panel fx-table', style: { borderTopLeftRadius: 0, borderTopRightRadius: 0, borderTop: 0 } }, ...shown.map((x) => this.rowEl(x, true))));
    }
    this.list.replaceChildren(...blocks);
    this.markAdded();
  }

  rowEl(x, withDay = false) {
    const f = x.f;
    const sig = JSON.stringify([f.probs, f.src, f.odds, f.pending, f.ou25, f.btts, f.score, x.started, withDay]);
    const hit = this.els.get(f.id);
    if (hit && hit.sig === sig) return hit.el;
    const el = this.buildRow(x, withDay);
    this.els.set(f.id, { el, sig });
    return el;
  }

  buildRow({ f, ko, started }, withDay) {
    const time = ko.hasTime ? hhmm(ko.date) : '—';
    const dayShort = withDay ? ko.date.toLocaleDateString(undefined, { weekday: 'short' }) : '';
    const top = f.probs ? f.probs.indexOf(Math.max(...f.probs)) : -1;
    const num = (v, cls = '') => h('div', { class: `fnum num ${cls}` }, v == null ? '—' : Math.round(v * 100));
    const src = SRC_TEXT[f.src] || f.src;
    const meta = [h('span', {}, f.league_name)];
    if (f.src) meta.push(h('span', { class: `src ${f.src === 'odds' ? 'odds' : ''}` }, ` · ${src}`));
    if (f.odds) meta.push(h('span', { class: 'num' }, ` · ${f.odds.map((o) => o.toFixed(2)).join('  ')}`));
    const add = h('button', { class: 'addbtn', 'aria-label': 'Add to coupon', title: 'Add to coupon',
      onclick: (e) => { e.stopPropagation(); this.toggle(f); } }, icon('plus', 16, 'plusi'), icon('check', 16, 'tick'));
    const open = () => showDetails(f.home, f.away, this.coupon, { odds: f.odds, ou_odds: f.ou_odds });
    return h('div', { class: 'fgrid frow', tabindex: 0, 'data-key': pairKey(f.home, f.away), title: 'Open the full breakdown',
      onclick: open,
      onkeydown: (e) => { if (e.key === 'Enter') open(); if (e.key === ' ' || e.key === '+') { e.preventDefault(); this.toggle(f); } } },
    h('div', { class: 'ftime num' }, withDay ? `${dayShort} ${time}` : time, started ? h('i', {}, 'started') : ''),
    h('div', { class: 'fmain' },
      h('div', { class: 'fteams' }, f.home_display, h('span', { class: 'vs' }, 'v'), f.away_display),
      h('div', { class: 'fmeta' }, ...meta)),
    ...CELLS.map((_, i) => num(f.probs?.[i], `${i === top ? 'top' : ''}${f.pending && !f.probs ? ' pending' : ''}`)),
    num(f.ou25), num(f.btts),
    h('div', { class: 'fnum num' }, f.score || '—', f.score ? h('span', { class: 'sub' }, `${Math.round(f.score_p * 100)}%`) : ''),
    add);
  }

  markAdded() {
    for (const { el } of this.els.values()) {
      if (!el.isConnected) continue;
      const added = this.coupon.rows.some((r) => r.complete && r.key === el.dataset.key);
      el.classList.toggle('added', added);
      el.querySelector('.addbtn')?.setAttribute('title', added ? 'Remove from coupon' : 'Add to coupon');
    }
  }

  toggle(f) {
    const row = this.coupon.find(f.home, f.away);
    if (row) { this.coupon.remove(row); return; }
    this.coupon.add(Coupon.fromFixture(f));
  }

  addShown() {
    const todo = this.shown().map((x) => x.f).filter((f) => !this.coupon.has(f.home, f.away));
    const run = () => { todo.forEach((f) => this.coupon.add(Coupon.fromFixture(f), { silent: true })); toast(`Added ${todo.length} matches. The coupon has ${this.coupon.n}.`, { key: 'add' }); };
    if (this.coupon.n + todo.length > 15) {
      modal('Add all of them?', h('p', {}, `The coupon would have ${this.coupon.n + todo.length} matches. Spor Toto has 15. You can remove matches afterwards.`),
        [['Cancel'], [`Add ${todo.length}`, run, 'primary']]);
    } else run();
  }

  // The league menu groups leagues by country, shows how many matches each has
  // in the chosen days, and has one-click presets.
  leagueMenu(anchor) {
    const rows = this.prepared().filter((x) => this.inWindow(x));
    const counts = new Map();
    rows.forEach(({ f }) => counts.set(f.league, { name: f.league_name, n: (counts.get(f.league)?.n || 0) + 1 }));
    const country = (name) => name.split(' ')[0];
    const box = h('div', { class: 'popover pop-leagues' });
    const q = textField({ placeholder: 'Find a league' });
    const list = h('div', { class: 'pop-list' });
    const draw = () => {
      const f = norm(q.input.value);
      const groups = new Map();
      [...counts.entries()].filter(([, v]) => !f || norm(v.name).includes(f)).forEach(([k, v]) => {
        const c = country(v.name);
        if (!groups.has(c)) groups.set(c, []);
        groups.get(c).push([k, v]);
      });
      const out = [];
      [...groups.keys()].sort().forEach((c) => {
        out.push(h('div', { class: 'pop-group' }, c));
        groups.get(c).sort((a, b) => a[1].name.localeCompare(b[1].name)).forEach(([k, v]) => {
          const cb = h('input', { type: 'checkbox', checked: this.leagues.has(k) });
          cb.addEventListener('change', () => { if (cb.checked) this.leagues.add(k); else this.leagues.delete(k); this.paint(); });
          out.push(h('label', { class: 'pop-item' }, cb, h('span', {}, v.name.replace(`${c} `, '')), h('span', { class: 'faint num' }, String(v.n))));
        });
      });
      list.replaceChildren(...(out.length ? out : [h('div', { class: 'blank' }, 'No league matches')]));
    };
    q.input.addEventListener('input', draw); draw();
    const presets = h('div', { class: 'row', style: { marginTop: '10px' } },
      ...PRESETS.map(([label, ks]) => h('button', { class: 'btn sm', onclick: () => {
        const have = ks.filter((k) => counts.has(k));
        if (!have.length) { toast(`No ${label} matches in these days`); return; }
        this.leagues = new Set(have); this.paint(); draw();
      } }, label)));
    box.append(q.wrap, presets, list, h('div', { class: 'row end', style: { marginTop: '10px' } },
      h('button', { class: 'btn sm quiet', onclick: () => { this.leagues.clear(); this.paint(); draw(); } }, 'Clear'),
      h('button', { class: 'btn sm primary', onclick: () => box.remove() }, 'Done')));
    const r = anchor.getBoundingClientRect();
    Object.assign(box.style, { left: `${Math.min(r.left, window.innerWidth - 360)}px`, top: `${r.bottom + 6}px` });
    document.body.append(box);
    q.input.focus();
    const off = (e) => { if (!box.contains(e.target) && !anchor.contains(e.target)) { box.remove(); document.removeEventListener('mousedown', off); } };
    setTimeout(() => document.addEventListener('mousedown', off), 0);
  }
}
