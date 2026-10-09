// The fixtures list: everything in the odds feed, grouped by day, filterable,
// with a "+" that appears on hover. It repaints in place while matches are
// still being priced, so the page is usable from the first second.
import { h, norm, pct, pct3, fair, kickoff, ymd, hhmm, dayLabel, outcomeTiles, badge, textField, toast, modal } from './ui.js';
import { api, store } from './api.js';
import { Coupon, pairKey } from './coupon.js';
import { showDetails } from './details.js';

const WINDOWS = [[0, 'Today'], [1, '+1'], [3, '+3'], [7, '+7'], [14, 'All']];

export class FixturesView {
  constructor(root, coupon) {
    this.root = root; this.coupon = coupon;
    this.window = Number(localStorage.getItem('fp.window') ?? 3);
    this.q = ''; this.leagues = new Set(); this.showStarted = false;
    this.data = { state: 'loading', rows: [], done: 0, total: 0 };
    this.els = new Map();                           // fixture id -> {el, sig}
    this.timer = null;
    this.unsub = [coupon.on((w) => { if (w === 'rows') this.markAdded(); }),
      store.on((w) => { if (w === 'mode') this.load(); })];
    this.build();
    this.load();
    this.onKey = (e) => {
      if (e.key === '/' && !/input|textarea/i.test(document.activeElement?.tagName || '')) {
        e.preventDefault(); this.search.input.focus();
      }
    };
    document.addEventListener('keydown', this.onKey);
  }
  destroy() { this.unsub.forEach((u) => u()); clearTimeout(this.timer); document.removeEventListener('keydown', this.onKey); }

  // ----------------------------------------------------------------- layout
  build() {
    this.winBtns = WINDOWS.map(([d, label]) => h('button', { class: 'seg-btn', 'data-d': d, onclick: () => this.setWindow(d), title: d === 14 ? 'Everything in the feed (two weeks)' : d === 0 ? 'Matches today' : `Today and the next ${d} day${d > 1 ? 's' : ''}` }, label));
    this.search = textField({ placeholder: 'Search team or league…  ( / )', cls: 'search', onInput: (v) => { this.q = norm(v); this.paint(); } });
    this.lgBtn = h('button', { class: 'btn', onclick: (e) => this.leagueMenu(e.currentTarget) }, 'Leagues ▾');
    this.addAll = h('button', { class: 'btn primary', onclick: () => this.addShown(), hidden: true });
    this.status = h('span', { class: 'small muted' });
    this.refresh = h('button', { class: 'iconbtn', title: 'Refresh odds and predictions', onclick: () => this.load(true) }, '↻');
    this.chips = h('div', { class: 'chips' });
    this.list = h('div', { class: 'fx-list' });
    this.root.replaceChildren(
      h('div', { class: 'fx-head' },
        h('div', { class: 'fx-title' }, h('h1', {}, 'Fixtures'), this.status, h('span', { class: 'spacer' }), this.refresh),
        h('div', { class: 'fx-tools' },
          h('div', { class: 'seg' }, ...this.winBtns), this.search.wrap, this.lgBtn, this.addAll),
        this.chips),
      this.list);
    this.syncWindow();
  }

  setWindow(d) { this.window = d; localStorage.setItem('fp.window', d); this.syncWindow(); this.paint(); }
  syncWindow() { this.winBtns.forEach((b) => b.classList.toggle('on', Number(b.dataset.d) === this.window)); }

  // ------------------------------------------------------------------ data
  async load(refresh = false) {
    clearTimeout(this.timer);
    try {
      this.data = await api.get(`/api/fixtures${refresh ? '?refresh=1' : ''}`);
      this.paint();
      if (this.data.state === 'loading') this.timer = setTimeout(() => this.load(), 900);
    } catch (e) {
      if (/warming up/i.test(e.message)) { this.timer = setTimeout(() => this.load(), 1500); return; }
      this.list.replaceChildren(h('div', { class: 'blank card' }, `Could not load fixtures: ${e.message}`));
    }
  }

  // prepared rows: kickoff in the viewer's time zone
  rows() {
    const now = new Date();
    return this.data.rows.map((f) => {
      const ko = kickoff(f.date, f.time);
      return { f, ko, day: ymd(ko.date), started: ko.hasTime && ko.date < now };
    }).filter((x) => !(x.ko.hasTime && x.ko.date.getTime() + 150 * 60000 < now.getTime())     // already over
      && (this.showStarted || !x.started));                                                    // already kicked off
  }

  startedCount() {
    const now = new Date(); const t = new Date(); const hi = ymd(new Date(t.getFullYear(), t.getMonth(), t.getDate() + this.window));
    return this.data.rows.filter((f) => { const ko = kickoff(f.date, f.time); return ko.hasTime && ko.date < now && ko.date.getTime() + 150 * 60000 >= now.getTime() && ymd(ko.date) <= hi; }).length;
  }

  shown() {
    const today = new Date(); const last = new Date(today.getFullYear(), today.getMonth(), today.getDate() + this.window);
    const lo = ymd(today); const hi = ymd(last);
    return this.rows().filter(({ f, day }) => day >= lo && day <= hi
      && (!this.leagues.size || this.leagues.has(f.league))
      && (!this.q || norm(`${f.home_display} ${f.away_display} ${f.home} ${f.away} ${f.league_name}`).includes(this.q)))
      .sort((a, b) => a.ko.date - b.ko.date || a.f.league_name.localeCompare(b.f.league_name));
  }

  // ------------------------------------------------------------------ paint
  paint() {
    const d = this.data;
    const all = this.rows();
    const shown = this.shown();
    const priced = d.rows.filter((r) => !r.pending).length;
    const hidden = this.showStarted ? 0 : this.startedCount();
    this.status.replaceChildren(d.state === 'loading' && d.total
      ? `pricing ${priced}/${d.total}…` : `${shown.length} match${shown.length === 1 ? '' : 'es'}`,
    hidden ? h('button', { class: 'linkbtn', style: { marginLeft: '10px' }, onclick: () => { this.showStarted = true; this.paint(); } }, `${hidden} already started — show`) : '',
    this.showStarted ? h('button', { class: 'linkbtn', style: { marginLeft: '10px' }, onclick: () => { this.showStarted = false; this.paint(); } }, 'hide started') : '');
    this.lgBtn.textContent = this.leagues.size ? `Leagues (${this.leagues.size}) ▾` : 'Leagues ▾';
    this.lgBtn.classList.toggle('on', this.leagues.size > 0);
    this.chips.replaceChildren(...[...this.leagues].map((k) => h('button', { class: 'chip on', title: 'Remove this filter', onclick: () => { this.leagues.delete(k); this.paint(); } },
      `${all.find((x) => x.f.league === k)?.f.league_name || k}  ×`)));
    const addable = shown.filter(({ f }) => !this.coupon.has(f.home, f.away));
    const filtered = this.leagues.size > 0 || this.q;
    this.addAll.hidden = !(filtered && addable.length > 1 && addable.length <= 40);
    this.addAll.textContent = `＋ Add all ${addable.length}`;

    if (!shown.length) {
      const msg = d.state === 'loading' ? 'Loading the odds feed…'
        : all.length ? 'Nothing matches these filters. Widen the days or clear the search.'
          : 'No fixtures in the odds feed right now (leagues on a break?). Add any match by name on the right.';
      this.list.replaceChildren(h('div', { class: 'blank card' }, d.state === 'loading' ? h('span', { class: 'spin-sm' }) : '', ' ', msg));
      return;
    }
    const byDay = new Map();
    shown.forEach((x) => { if (!byDay.has(x.day)) byDay.set(x.day, []); byDay.get(x.day).push(x); });
    const blocks = [];
    for (const [, xs] of byDay) {
      blocks.push(h('div', { class: 'day' }, dayLabel(xs[0].ko.date), h('span', { class: 'faint' }, `  ${xs.length}`)));
      blocks.push(h('div', { class: 'card fx-card' }, ...xs.map((x) => this.rowEl(x))));
    }
    this.list.replaceChildren(...blocks);
    this.markAdded();
  }

  // a row is rebuilt only when what it shows has changed
  rowEl(x) {
    const f = x.f;
    const sig = JSON.stringify([f.probs, f.src, f.v1, f.v2, f.odds, f.pending, f.ou25, f.btts, x.started]);
    const hit = this.els.get(f.id);
    if (hit && hit.sig === sig) return hit.el;
    const el = this.buildRow(x);
    this.els.set(f.id, { el, sig });
    return el;
  }

  buildRow({ f, ko, started }) {
    const time = ko.hasTime ? hhmm(ko.date) : '—';
    const odds = f.odds ? `odds ${f.odds.map((o) => o.toFixed(2)).join(' / ')}` : null;
    const goals = f.ou25 != null ? `O2.5 ${pct(f.ou25)} · BTTS ${pct(f.btts)}` : null;
    const detail = [
      h('span', { class: 'lg' }, f.league_name),
      f.src ? badge(f.src) : '',
      odds ? h('span', {}, odds) : '',
      f.v2 ? h('span', { class: 'cmp' }, h('b', {}, 'v2 '), pct3(f.v2)) : '',
      f.v1 ? h('span', { class: 'cmp' }, h('b', {}, 'v1 '), pct3(f.v1)) : '',
      goals ? h('span', {}, goals) : '',
    ].filter(Boolean);
    const plus = h('button', { class: 'plus', 'aria-label': 'Add to coupon', onclick: (e) => { e.stopPropagation(); this.toggle(f); } }, h('span', { class: 'p-plus' }, '+'), h('span', { class: 'p-tick' }, '✓'));
    const el = h('div', { class: 'frow', tabindex: 0, 'data-key': pairKey(f.home, f.away), title: 'Click for the full breakdown',
      onclick: () => showDetails(f.home, f.away, this.coupon, { odds: f.odds }),
      onkeydown: (e) => { if (e.key === 'Enter') showDetails(f.home, f.away, this.coupon, { odds: f.odds }); if (e.key === '+' || e.key === ' ') { e.preventDefault(); this.toggle(f); } } },
    h('div', { class: 'ftime' }, time, started ? h('i', {}, 'started') : ''),
    h('div', { class: 'fmain' },
      h('div', { class: 'fteams' }, h('span', {}, f.home_display), h('em', {}, '–'), h('span', {}, f.away_display)),
      h('div', { class: 'fdetail' }, ...detail)),
    f.probs ? outcomeTiles(f.probs, { src: f.src }) : h('div', { class: `tiles3 pending` }, ...['1', 'X', '2'].map((s) => h('div', { class: 't3 empty' }, h('b', {}, s), h('span', {}, '·')))),
    plus);
    return el;
  }

  markAdded() {
    for (const { el } of this.els.values()) {
      if (!el.isConnected) continue;
      el.classList.toggle('added', this.coupon.rows.some((r) => r.complete && r.key === el.dataset.key));
    }
  }

  toggle(f) {
    const row = this.coupon.find(f.home, f.away);
    if (row) { this.coupon.remove(row); return; }
    this.coupon.add(Coupon.fromFixture(f));
  }

  addShown() {
    const todo = this.shown().map((x) => x.f).filter((f) => !this.coupon.has(f.home, f.away));
    const run = () => { todo.forEach((f) => this.coupon.add(Coupon.fromFixture(f), { silent: true })); toast(`Added ${todo.length} matches (${this.coupon.n} on the coupon)`); };
    if (this.coupon.n + todo.length > 15) {
      modal('Add them all?', h('p', {}, `That would put ${this.coupon.n + todo.length} matches on the coupon — Spor Toto has 15. You can remove extras afterwards.`),
        [['Cancel'], [`Add ${todo.length}`, run, 'primary']]);
    } else run();
  }

  leagueMenu(anchor) {
    const rows = this.rows().filter((x) => { const t = new Date(); return x.day >= ymd(t) && x.day <= ymd(new Date(t.getFullYear(), t.getMonth(), t.getDate() + this.window)); });
    const counts = new Map();
    rows.forEach(({ f }) => counts.set(f.league, { name: f.league_name, n: (counts.get(f.league)?.n || 0) + 1 }));
    const entries = [...counts.entries()].sort((a, b) => a[1].name.localeCompare(b[1].name));
    const box = h('div', { class: 'popover pop-leagues' });
    const q = textField({ placeholder: 'Find a league…' });
    const list = h('div', { class: 'pop-list' });
    const draw = () => {
      const f = norm(q.input.value);
      list.replaceChildren(...entries.filter(([, v]) => !f || norm(v.name).includes(f)).map(([k, v]) => {
        const cb = h('input', { type: 'checkbox', checked: this.leagues.has(k) });
        cb.addEventListener('change', () => { if (cb.checked) this.leagues.add(k); else this.leagues.delete(k); this.paint(); });
        return h('label', { class: 'pop-item' }, cb, h('span', {}, v.name), h('span', { class: 'faint' }, String(v.n)));
      }));
    };
    q.input.addEventListener('input', draw); draw();
    box.append(q.wrap, list, h('div', { class: 'row end', style: { marginTop: '8px' } },
      h('button', { class: 'btn sm', onclick: () => { this.leagues.clear(); this.paint(); box.remove(); } }, 'Clear'),
      h('button', { class: 'btn sm primary', onclick: () => box.remove() }, 'Done')));
    const r = anchor.getBoundingClientRect();
    Object.assign(box.style, { position: 'fixed', left: `${Math.min(r.left, window.innerWidth - 320)}px`, top: `${r.bottom + 6}px` });
    document.body.append(box);
    q.input.focus();
    const off = (e) => { if (!box.contains(e.target) && e.target !== anchor) { box.remove(); document.removeEventListener('mousedown', off); } };
    setTimeout(() => document.addEventListener('mousedown', off), 0);
  }
}
