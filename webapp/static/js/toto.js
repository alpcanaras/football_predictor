// Toto: a live coupon editor + ticket optimiser.
//
// Rows re-price themselves when a team or the odds change (debounced, cached,
// a few requests at a time); the ticket re-optimises whenever probabilities,
// budget, objective or locks change; everything autosaves to the same coupon
// files the Streamlit app uses. Inputs are built once per row and never
// re-rendered while you type.
import { h, debounce, pct, fold, fair, probBar, setBar, badge, toast, modal, Autocomplete, legend } from './ui.js';
import { api, store } from './api.js';

const PRESETS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256];
const LABELS = ['1', 'X', '2'];
const MASK = { '1': 1, X: 2, '2': 4, '1X': 3, '12': 5, X2: 6, '1X2': 7 };
const LABEL = Object.fromEntries(Object.entries(MASK).map(([k, v]) => [v, k]));
const UNIFORM = [1 / 3, 1 / 3, 1 / 3];

let uid = 0;
const cache = new Map();                     // analysis cache: key -> result
const queue = []; let inflight = 0;          // polite request queue

function enqueue(job) { queue.push(job); pump(); }
function pump() {
  while (inflight < 3 && queue.length) {
    const job = queue.shift(); inflight += 1;
    job().finally(() => { inflight -= 1; pump(); });
  }
}

const oddsOk = (o) => Array.isArray(o) && o.length === 3 && o.every((x) => x > 1);
const parseNum = (s) => { const v = parseFloat(String(s).replace(',', '.')); return Number.isFinite(v) ? v : null; };
const pairKey = (r) => [fold(r.home), fold(r.away)].sort().join('|');

class Row {
  constructor(d = {}) {
    this.id = ++uid; this.home = d.home || ''; this.away = d.away || '';
    this.odds = oddsOk(d.odds) ? d.odds : null; this.crowd = d.crowd || null; this.lock = d.lock || null;
    this.an = null; this.akey = null; this.view = null;
  }
  get empty() { return !this.home.trim() && !this.away.trim(); }
  get complete() { return !!(this.home.trim() && this.away.trim()); }
  analysisKey() { return `${fold(this.home)}|${fold(this.away)}|${this.odds ? this.odds.join(',') : ''}|${store.status?.model_mode}`; }
  probs() { return this.an?.probs || UNIFORM; }
}

export class TotoPage {
  constructor(root) {
    this.root = root;
    this.game = localStorage.getItem('fp.game') || 'turkish';
    this.rows = []; this.budget = 1; this.tilt = 0.5; this.objective = 'hit';
    this.showCrowd = false; this.result = null; this.seq = 0; this.loaded = false;
    this.save = debounce(() => this.persist(), 600);
    this.retick = debounce(() => this.computeTicket(), 250);
    this.unsub = store.on((what) => { if (what === 'mode') this.reanalyzeAll(true); if (what === 'compare') this.rows.forEach((r) => this.paintRow(r)); });
    this.build();
    this.load(this.game);
  }
  destroy() { this.unsub(); this.save.flush?.(); }

  // ------------------------------------------------------------------ layout
  build() {
    const games = store.status.games;
    this.gameBtns = Object.entries(games).map(([g, info]) => {
      const cnt = h('span', { class: 'count' }, '');
      const b = h('button', { class: 'game', onclick: () => this.switchGame(g) },
        h('span', { class: 'flag' }, info.flag || '🎟️'),
        h('span', {}, h('b', {}, info.label || g), h('span', { class: 'sub' }, `${info.n} matches · prize at ${info.threshold}+`)),
        cnt);
      b._cnt = cnt; b._game = g; return b;
    });

    this.statusEl = h('span', { class: 'status' });
    this.savedEl = h('span', { class: 'tiny faint' });
    const toolbar = h('div', { class: 'toolbar' },
      h('button', { class: 'btn sm', onclick: () => this.pasteDialog(), title: 'Paste a whole coupon, one match per line' }, '📋 Paste coupon'),
      h('button', { class: 'btn sm', onclick: () => this.fillOdds(), title: 'Look every match without odds up in the live bookmaker feed' }, '📡 Fill odds from feed'),
      this.crowdBtn = h('button', { class: 'btn sm', onclick: () => this.toggleCrowd(), title: 'Crowd percentages (oynanma yüzdesi) — lets the ticket lean away from what everyone else plays' }, '👥 Crowd %'),
      this.crowdPaste = h('button', { class: 'btn sm', onclick: () => this.pasteCrowd(), title: 'Paste one “68/21/11” line per match, in coupon order' }, '📋 Paste crowd %'),
      h('button', { class: 'btn sm ghost danger', onclick: () => this.clearCoupon() }, '🗑 Clear'),
      this.statusEl, this.savedEl);

    this.table = h('div', { class: 'coupon' });
    const left = h('div', { class: 'card' }, toolbar, this.table);

    this.panel = h('div', { class: 'panel' });
    this.root.replaceChildren(
      h('div', { class: 'games' }, ...this.gameBtns),
      h('div', { class: 'grid-toto' }, left, this.panel),
      this.printArea = h('div', { class: 'print-area print-only' }));
    this.buildPanel();
  }

  header() {
    const cells = ['#', 'Home', 'Away', 'Odds 1 · X · 2', 'Probability', ...(this.showCrowd ? ['Crowd %'] : []), 'Play', ''];
    return h('div', { class: 'crow head' }, ...cells.map((c) => h('div', {}, c)));
  }

  renderTable() {
    this.table.classList.toggle('with-crowd', this.showCrowd);
    this.crowdBtn.classList.toggle('primary', this.showCrowd);
    this.crowdPaste.hidden = !this.showCrowd;
    const els = [this.header()];
    this.rows.forEach((r, i) => { r.view = this.rowView(r, i); els.push(r.view.el); });
    this.ghost = new Row(); this.ghost.view = this.rowView(this.ghost, this.rows.length, true);
    els.push(this.ghost.view.el);
    this.table.replaceChildren(...els);
    this.rows.forEach((r) => this.paintRow(r));
    this.updateStatus();
  }

  rowView(r, i, ghost = false) {
    const mkInput = (side) => {
      const inp = h('input', { class: 'cell', value: r[side], placeholder: side === 'home' ? (ghost ? '+ Add a match — home team' : 'Home') : 'Away', spellcheck: false });
      const wrap = h('div', {}, inp);
      inp.addEventListener('input', () => { if (ghost && inp.value.trim()) this.promoteGhost(r); r[side] = inp.value; this.onEdit(r); });
      inp.addEventListener('change', () => { r[side] = inp.value.trim(); inp.value = r[side]; this.onEdit(r, true); });
      inp.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !wrap.querySelector('.ac-list')) {
          e.preventDefault();
          if (side === 'home') v.away.focus(); else this.focusNext(r);
        }
      });
      new Autocomplete(inp, { source: () => store.teams });
      return [inp, wrap];
    };
    const [home, homeW] = mkInput('home');
    const [away, awayW] = mkInput('away');
    const odds = [0, 1, 2].map((k) => {
      const inp = h('input', { class: 'cell', value: r.odds ? r.odds[k].toFixed(2) : '', placeholder: LABELS[k], inputmode: 'decimal' });
      inp.addEventListener('change', () => {
        const vals = odds.map((o) => parseNum(o.value));
        r.odds = vals.every((x) => x && x > 1) ? vals : null;
        if (!vals.some((x) => x)) r.odds = null;
        this.onEdit(r, true);
      });
      inp.addEventListener('paste', (e) => {           // "1.95 3.40 3.90" in one go
        const t = e.clipboardData.getData('text').trim().split(/[\s/;]+/).map(parseNum);
        if (t.length === 3 && t.every((x) => x && x > 1)) {
          e.preventDefault(); odds.forEach((o, j) => { o.value = t[j].toFixed(2); });
          r.odds = t; this.onEdit(r, true);
        }
      });
      return inp;
    });
    const bar = probBar(null);
    const src = h('span');
    const mini = h('div', { class: 'tiny muted' });
    const prob = h('div', { class: 'probcell' }, h('div', { style: { flex: 1, minWidth: 0 } }, bar, mini), src);
    const crowd = h('input', { class: 'cell', value: r.crowd ? r.crowd.map((x) => Math.round(x)).join('/') : '', placeholder: '1/X/2 %' });
    crowd.addEventListener('change', () => {
      const t = crowd.value.trim().split(/[\s/;,]+/).map(parseNum).filter((x) => x != null);
      r.crowd = t.length === 3 && t.reduce((a, b) => a + b, 0) > 0 ? t : null;
      crowd.value = r.crowd ? r.crowd.map((x) => Math.round(x)).join('/') : '';
      this.onEdit(r, false, true);
    });
    const play = h('div', { class: 'play' }, ...LABELS.map((s) => h('button', {
      class: 'sym', title: `Click to force ${s} on this match (the rest of the ticket re-optimises around it)`,
      onclick: () => this.toggleSymbol(r, s),
    }, s)), h('button', {
      class: 'lock', title: 'Locked by you — click to hand this match back to the optimiser',
      onclick: () => { r.lock = null; this.paintPlay(r); this.save(); this.retick(); },
    }, '🔒'));
    const menu = h('button', { class: 'iconbtn', title: 'Row options', onclick: (e) => this.rowMenu(r, e.currentTarget) }, '⋯');
    const idx = h('div', { class: 'idx' }, ghost ? '+' : String(i + 1));
    const cells = [idx, homeW, awayW, h('div', { class: 'odds3' }, ...odds), prob,
      ...(this.showCrowd ? [h('div', { class: 'crowd1' }, crowd)] : []), play, ghost ? h('div') : menu];
    const el = h('div', { class: `crow${ghost ? ' ghost-row' : ''}` }, ...cells);
    const sug = h('div', { class: 'suggest' });
    const wrapper = h('div', {}, el, sug);
    const v = { el: wrapper, row: el, idx, home, away, odds, bar, src, mini, crowd, play, sug };
    if (ghost) { [play, ...odds, crowd].forEach((x) => { x.style.visibility = 'hidden'; }); }
    return v;
  }

  focusNext(r) {
    const i = this.rows.indexOf(r);
    const next = this.rows[i + 1] || this.ghost;
    next?.view?.home.focus();
  }

  promoteGhost(r) {
    if (r !== this.ghost) return;
    this.rows.push(r);
    const v = r.view;
    v.idx.textContent = String(this.rows.length);
    v.row.classList.remove('ghost-row');
    [v.play, ...v.odds, v.crowd].forEach((x) => { x.style.visibility = ''; });
    v.home.placeholder = 'Home';
    const menu = h('button', { class: 'iconbtn', title: 'Row options', onclick: (e) => this.rowMenu(r, e.currentTarget) }, '⋯');
    v.row.lastChild.replaceWith(menu);
    this.ghost = new Row(); this.ghost.view = this.rowView(this.ghost, this.rows.length, true);
    this.table.append(this.ghost.view.el);
  }

  // ------------------------------------------------------------------ edits
  onEdit(r, commit = false, crowdOnly = false) {
    if (r === this.ghost) return;
    this.save();
    if (crowdOnly) { this.updateStatus(); this.retick(); return; }
    this.markDuplicates();
    if (commit) this.analyze(r);
    else { r.soon = r.soon || debounce(() => this.analyze(r), 450); r.soon(); }
    this.updateStatus();
  }

  analyze(r, force = false) {
    if (!r.complete) { r.an = null; r.akey = null; this.paintRow(r); this.retick(); return; }
    const key = r.analysisKey();
    if (!force && key === r.akey && r.an) return;
    r.akey = key;
    if (!force && cache.has(key)) { r.an = cache.get(key); this.paintRow(r); this.retick(); return; }
    setBar(r.view.bar, r.an?.probs, true);
    enqueue(async () => {
      try {
        const res = await api.post('/api/analyze', { home: r.home, away: r.away, odds: r.odds });
        cache.set(key, res);
        if (r.akey === key) { r.an = res; this.paintRow(r); this.retick(); this.updateStatus(); }
      } catch (e) {
        toast(`Could not price ${r.home} – ${r.away}: ${e.message}`, { error: true });
        setBar(r.view.bar, r.an?.probs, false);
      }
    });
  }

  reanalyzeAll(force = false) {
    if (force) cache.clear();
    this.rows.forEach((r) => this.analyze(r, force));
  }

  paintRow(r) {
    const v = r.view; if (!v) return;
    const a = r.an;
    setBar(v.bar, a?.probs || null, false);
    v.src.replaceChildren(a ? badge(a.src) : '');
    const known = a?.known || {};
    v.home.classList.toggle('bad', !!a && r.home && !known.home);
    v.away.classList.toggle('bad', !!a && r.away && !known.away);
    // v1 / v2 side by side where both exist
    const bits = [];
    if (store.compare && a) {
      if (a.v1) bits.push(`v1 ${a.v1.map((x) => Math.round(x * 100)).join('·')}`);
      if (a.v2) bits.push(`v2 ${a.v2.map((x) => Math.round(x * 100)).join('·')}`);
      if (a.league_name) bits.push(a.league_name);
    }
    v.mini.textContent = bits.join('  ·  ');
    v.bar.title = a?.probs ? `1 ${pct(a.probs[0], 1)} · X ${pct(a.probs[1], 1)} · 2 ${pct(a.probs[2], 1)}` : '';
    // suggestions for names we could not resolve
    const sug = [];
    for (const side of ['home', 'away']) {
      if (a && !known[side] && r[side]) {
        const opts = a.suggest?.[side] || [];
        sug.push(h('span', {}, `“${r[side]}” not recognised`), opts.length ? ' → ' : ' — add odds for this row',
          ...opts.map((o) => h('button', { class: 'chip', onclick: () => { r[side] = o; v[side].value = o; this.onEdit(r, true); } }, o)));
      }
    }
    v.sug.replaceChildren(...sug);
    v.row.classList.toggle('unknown', sug.length > 0);
    this.paintPlay(r);
  }

  paintPlay(r) {
    const v = r.view; if (!v) return;
    const i = this.rows.indexOf(r);
    const label = r.lock || this.result?.labels?.[i] || '';
    [...v.play.querySelectorAll('button.sym')].forEach((b, k) => b.classList.toggle('sel', label.includes(LABELS[k])));
    v.play.classList.toggle('locked', !!r.lock);
    v.play.title = r.lock ? 'Locked by you — click a symbol to change, or ⋯ → Unlock' : 'Chosen by the optimiser — click a symbol to lock your own pick';
  }

  toggleSymbol(r, s) {
    const cur = r.lock || this.result?.labels?.[this.rows.indexOf(r)] || '';
    let set = new Set(cur.split('').filter(Boolean));
    if (!r.lock) set = new Set([s]);              // first click: lock exactly this
    else if (set.has(s)) set.delete(s); else set.add(s);
    const lab = LABELS.filter((x) => set.has(x)).join('');
    r.lock = lab && MASK[lab] ? lab : null;
    this.paintPlay(r); this.save(); this.retick();
  }

  markDuplicates() {
    const seen = new Map();
    this.rows.forEach((r) => {
      const dup = r.complete && seen.has(pairKey(r));
      if (r.complete && !dup) seen.set(pairKey(r), r);
      r.view?.row.classList.toggle('dup', dup);
      r.dup = dup;
    });
  }

  rowMenu(r, anchor) {
    const i = this.rows.indexOf(r);
    const items = [
      ['↑ Move up', i > 0, () => this.move(r, -1)],
      ['↓ Move down', i < this.rows.length - 1, () => this.move(r, 1)],
      ['Unlock pick', !!r.lock, () => { r.lock = null; this.paintPlay(r); this.save(); this.retick(); }],
      ['Clear odds', !!r.odds, () => { r.odds = null; r.view.odds.forEach((o) => { o.value = ''; }); this.onEdit(r, true); }],
      ['Delete row', true, () => this.remove(r)],
    ];
    const m = h('div', { class: 'ac-list', style: { left: 'auto', right: 0, minWidth: '150px' } },
      ...items.filter((x) => x[1]).map(([t, , fn]) => h('div', { class: 'ac-item', onmousedown: (e) => { e.preventDefault(); m.remove(); fn(); } }, t)));
    anchor.parentElement.style.position = 'relative';
    anchor.parentElement.append(m);
    const off = (e) => { if (!m.contains(e.target)) { m.remove(); document.removeEventListener('mousedown', off); } };
    setTimeout(() => document.addEventListener('mousedown', off), 0);
  }

  move(r, d) {
    const i = this.rows.indexOf(r); const j = i + d;
    [this.rows[i], this.rows[j]] = [this.rows[j], this.rows[i]];
    this.renderTable(); this.save(); this.retick();
  }

  remove(r) {
    const i = this.rows.indexOf(r);
    this.rows.splice(i, 1);
    this.renderTable(); this.save(); this.retick();
    toast(`Removed ${r.home || '?'} – ${r.away || '?'}`, { action: 'Undo', onAction: () => { this.rows.splice(i, 0, r); this.renderTable(); this.save(); this.retick(); } });
  }

  updateStatus() {
    const info = store.status.games[this.game];
    const done = this.rows.filter((r) => r.complete);
    const unknown = done.filter((r) => r.an && (!r.an.known.home || !r.an.known.away)).length;
    const dups = done.filter((r) => r.dup).length;
    const odds = done.filter((r) => r.an && (r.an.src === 'odds' || r.an.src === 'feed')).length;
    const parts = [`${done.length}/${info.n} matches`];
    if (odds) parts.push(`${odds} priced from odds`);
    if (unknown) parts.push(`⚠ ${unknown} unrecognised`);
    if (dups) parts.push(`⚠ ${dups} duplicate`);
    this.statusEl.textContent = parts.join(' · ');
    for (const b of this.gameBtns) {
      if (b._game === this.game) {
        b._cnt.textContent = `${done.length}/${info.n}`;
        b._cnt.classList.toggle('full', done.length === info.n);
      }
    }
    const allCrowd = done.length > 0 && done.every((r) => r.crowd);
    if (this.objSeg) {
      this.objSeg.querySelector('[data-o=crowd]').disabled = !allCrowd;
      if (!allCrowd && this.objective === 'crowd') { this.objective = 'hit'; this.paintObjective(); }
    }
  }

  // ------------------------------------------------------------------ io
  async load(game) {
    this.game = game; localStorage.setItem('fp.game', game);
    this.gameBtns.forEach((b) => b.classList.toggle('on', b._game === game));
    const c = await api.get(`/api/coupon/${game}`);
    this.rows = c.rows.map((d) => new Row(d));
    this.budget = c.budget || 1; this.tilt = c.tilt ?? 0.5; this.objective = c.objective || 'hit';
    this.showCrowd = this.rows.some((r) => r.crowd);
    this.result = null; this.loaded = true;
    this.renderTable(); this.markDuplicates(); this.paintPanelInputs();
    this.rows.forEach((r) => this.analyze(r));
    this.retick();
    // counts on the other game's card
    for (const b of this.gameBtns) {
      if (b._game !== game) {
        api.get(`/api/coupon/${b._game}`).then((o) => { b._cnt.textContent = `${o.rows.length}/${o.n}`; b._cnt.classList.toggle('full', o.rows.length === o.n); });
      }
    }
  }

  async switchGame(g) {
    if (g === this.game) return;
    this.save.flush(); await new Promise((r) => setTimeout(r, 50));
    this.load(g);
  }

  async persist() {
    if (!this.loaded) return;
    const body = {
      rows: this.rows.filter((r) => !r.empty).map((r) => ({ home: r.home, away: r.away, odds: r.odds, crowd: r.crowd, lock: r.lock })),
      budget: this.budget, tilt: this.tilt, objective: this.objective,
    };
    try {
      await api.put(`/api/coupon/${this.game}`, body);
      this.savedEl.textContent = 'Saved ✓';
      setTimeout(() => { this.savedEl.textContent = ''; }, 1500);
    } catch (e) { toast(`Could not save the coupon: ${e.message}`, { error: true }); }
  }

  pasteDialog() {
    const ta = h('textarea', { class: 'txt', rows: 12, placeholder: 'Galatasaray - Fenerbahce\nBayern Munich - Dortmund  1.50 4.60 6.00\nTurkey - Spain\n…' });
    const note = h('div', { class: 'small muted', style: { margin: '6px 0 8px' } },
      'One match per line: “Home - Away”, optionally followed by the three odds. Also accepts “v”, “vs” or “:”.');
    const apply = async (mode) => {
      const { rows } = await api.post('/api/parse', { text: ta.value });
      if (!rows.length) { toast('Nothing recognisable to add'); return false; }
      const fresh = rows.map((d) => new Row(d));
      if (mode === 'replace') { this.snapshot(); this.rows = fresh; } else { this.rows.push(...fresh); }
      this.renderTable(); this.markDuplicates(); this.save();
      this.rows.forEach((r) => this.analyze(r)); this.retick();
      toast(`${mode === 'replace' ? 'Replaced the coupon with' : 'Added'} ${rows.length} match${rows.length > 1 ? 'es' : ''}`);
      return true;
    };
    modal('Paste a coupon', h('div', {}, note, ta), [
      ['Cancel'], ['Append', () => apply('append')], ['Replace coupon', () => apply('replace'), 'primary']]);
    setTimeout(() => ta.focus(), 30);
  }

  async fillOdds() {
    const targets = this.rows.filter((r) => r.complete && !r.odds);
    if (!targets.length) { toast('Every match already has odds'); return; }
    toast('Looking the matches up in the live odds feed…');
    try {
      const { odds } = await api.post('/api/fill-odds', { rows: targets.map((r) => ({ home: r.home, away: r.away })) });
      let n = 0;
      targets.forEach((r, k) => {
        if (odds[k]) { r.odds = odds[k]; r.view.odds.forEach((o, j) => { o.value = odds[k][j].toFixed(2); }); n += 1; this.analyze(r); }
      });
      this.save();
      toast(n ? `Filled odds for ${n} of ${targets.length} match${targets.length > 1 ? 'es' : ''}` + (n < targets.length ? ' — the rest are not in the feed yet (it only covers the next few days)' : '')
        : 'None of these are in the feed yet — it only covers the next few days');
    } catch (e) { toast(`Feed unavailable: ${e.message}`, { error: true }); }
  }

  pasteCrowd() {
    const rows = this.rows.filter((r) => r.complete);
    const ta = h('textarea', { class: 'txt', rows: Math.min(Math.max(rows.length, 6), 16), placeholder: '68/21/11\n45/30/25\n…' });
    modal('Paste crowd percentages', h('div', {},
      h('div', { class: 'small muted', style: { marginBottom: '8px' } },
        `One line per match in coupon order (${rows.length} matches): the share of players on 1, X and 2 — “68/21/11”, “68 21 11” or “68;21;11”.`), ta), [
      ['Cancel'],
      ['Apply', () => {
        const lines = ta.value.split(/\n/).map((l) => l.trim().split(/[\s/;,]+/).map(parseNum).filter((x) => x != null)).filter((t) => t.length >= 3);
        if (!lines.length) { toast('No “1/X/2” lines found'); return false; }
        rows.forEach((r, i) => { if (lines[i]) { r.crowd = lines[i].slice(-3); if (r.view?.crowd) r.view.crowd.value = r.crowd.map((x) => Math.round(x)).join('/'); } });
        this.save(); this.updateStatus(); this.paintObjective(); this.retick();
        toast(`Crowd % set for ${Math.min(lines.length, rows.length)} of ${rows.length} matches` + (lines.length !== rows.length ? ` — ${lines.length} lines for ${rows.length} matches, check the order` : ''));
        return true;
      }, 'primary']]);
    setTimeout(() => ta.focus(), 30);
  }

  toggleCrowd() {
    this.showCrowd = !this.showCrowd;
    this.renderTable();
  }

  snapshot() { this.undoRows = this.rows.map((r) => ({ home: r.home, away: r.away, odds: r.odds, crowd: r.crowd, lock: r.lock })); }

  clearCoupon() {
    if (!this.rows.length) return;
    this.snapshot();
    const keep = this.undoRows;
    this.rows = []; this.result = null;
    this.renderTable(); this.save(); this.retick();
    toast('Coupon cleared', { action: 'Undo', onAction: () => { this.rows = keep.map((d) => new Row(d)); this.renderTable(); this.markDuplicates(); this.save(); this.rows.forEach((r) => this.analyze(r)); this.retick(); } });
  }

  // ------------------------------------------------------------------ ticket
  buildPanel() {
    const info = () => store.status.games[this.game];
    this.budgetIn = h('input', { class: 'txt', type: 'number', min: 1, max: 100000, value: this.budget });
    this.budgetIn.addEventListener('input', () => { const b = parseInt(this.budgetIn.value, 10); if (b >= 1) { this.budget = b; this.paintPresets(); this.save(); this.retick(); } });
    this.presets = h('div', { class: 'presets' }, ...PRESETS.map((p) => h('button', { onclick: () => { this.budget = p; this.budgetIn.value = p; this.paintPresets(); this.save(); this.retick(); } }, String(p))));
    this.priceIn = h('input', { class: 'txt', type: 'number', min: 0, step: '0.01', placeholder: 'price / column', style: { width: '120px' } });
    this.priceIn.addEventListener('input', () => { localStorage.setItem(`fp.price.${this.game}`, this.priceIn.value); this.paintResult(); });

    this.objSeg = h('div', { class: 'seg' },
      h('button', { 'data-o': 'hit', onclick: () => this.setObjective('hit'), title: 'Maximise the chance of reaching the prize' }, 'Hit chance'),
      h('button', { 'data-o': 'crowd', onclick: () => this.setObjective('crowd'), title: 'Lean toward outcomes the crowd under-plays (needs crowd % on every row)' }, 'Crowd-aware'));
    this.tiltIn = h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: this.tilt, style: { width: '100%' } });
    this.tiltLbl = h('span', { class: 'small muted' });
    this.tiltIn.addEventListener('input', () => { this.tilt = parseFloat(this.tiltIn.value); this.tiltLbl.textContent = `tilt ${this.tilt.toFixed(2)}`; this.save(); this.retick(); });
    this.tiltBox = h('div', { class: 'small', style: { marginTop: '8px' } }, h('div', { class: 'row' }, h('span', { class: 'muted' }, 'Lean on the crowd'), h('span', { class: 'spacer' }), this.tiltLbl), this.tiltIn,
      h('div', { class: 'tiny faint' }, '0 = pure hit chance · above ~0.75 the hit rate collapses'));

    this.resultBox = h('div');
    this.ladderBox = h('div', { class: 'small muted' }, 'Open to compute.');
    this.ladder = h('details', { class: 'fold' }, h('summary', {}, '💸 What does more budget buy?'), this.ladderBox);
    this.ladder.addEventListener('toggle', () => { if (this.ladder.open) this.loadLadder(); });
    this.sweepBox = h('div', { class: 'small muted' });
    this.sweep = h('details', { class: 'fold' }, h('summary', {}, '⚖️ Hit chance vs crowd payout'), this.sweepBox);
    this.sweep.addEventListener('toggle', () => { if (this.sweep.open) this.loadSweep(); });

    const actions = h('div', { class: 'row', style: { marginTop: '12px' } },
      h('button', { class: 'btn sm', onclick: () => this.copyTicket() }, '📋 Copy'),
      h('button', { class: 'btn sm', onclick: () => this.printTicket() }, '🖨 Print'),
      h('span', { class: 'spacer' }),
      h('button', { class: 'btn sm primary', onclick: () => this.saveHistory() }, '💾 Save to history'));

    this.panel.replaceChildren(
      h('div', { class: 'card card-pad' },
        h('h3', {}, 'Budget'),
        h('div', { class: 'budget-row' }, this.budgetIn, h('span', { class: 'muted' }, 'columns'), h('span', { class: 'spacer' }), this.priceIn),
        this.presets,
        h('div', { style: { marginTop: '12px' } }, this.objSeg), this.tiltBox),
      h('div', { class: 'card card-pad' }, h('h3', {}, 'Your ticket'), this.resultBox, actions, this.ladder, this.sweep));
    this.paintPanelInputs();
    this.paintResult();
    void info;
  }

  paintPanelInputs() {
    if (!this.budgetIn) return;
    this.budgetIn.value = this.budget;
    this.tiltIn.value = this.tilt; this.tiltLbl.textContent = `tilt ${Number(this.tilt).toFixed(2)}`;
    this.priceIn.value = localStorage.getItem(`fp.price.${this.game}`) || '';
    this.paintPresets(); this.paintObjective(); this.updateStatus();
  }
  paintPresets() { [...this.presets.children].forEach((b) => b.classList.toggle('on', Number(b.textContent) === this.budget)); }
  setObjective(o) { this.objective = o; this.paintObjective(); this.save(); this.retick(); }
  paintObjective() {
    [...this.objSeg.children].forEach((b) => b.classList.toggle('on', b.dataset.o === this.objective));
    this.tiltBox.hidden = this.objective !== 'crowd';
    this.sweep.hidden = !this.rows.some((r) => r.crowd);
  }

  ticketBody() {
    const rows = this.rows.filter((r) => r.complete);
    return {
      rows,
      body: {
        game: this.game, budget: this.budget, objective: this.objective, tilt: this.tilt,
        probs: rows.map((r) => r.probs()), locks: rows.map((r) => r.lock),
        crowd: rows.every((r) => r.crowd) ? rows.map((r) => r.crowd) : null,
      },
    };
  }

  async computeTicket() {
    const { rows, body } = this.ticketBody();
    this.tRows = rows;
    if (!rows.length) { this.result = null; this.paintResult(); this.rows.forEach((r) => this.paintPlay(r)); return; }
    const seq = ++this.seq;
    this.resultBox.classList.add('busy');
    try {
      const res = await api.post('/api/ticket', body);
      if (seq !== this.seq) return;                // a newer request is on its way
      // labels are per complete row; map them back onto all rows
      const byRow = new Map(rows.map((r, i) => [r, res.labels[i]]));
      res.rowLabels = byRow;
      this.result = { ...res, labels: this.rows.map((r) => byRow.get(r) || '') };
      this.paintResult();
      this.rows.forEach((r) => this.paintPlay(r));
      if (this.ladder.open) this.loadLadder();
      this.sweepBox.replaceChildren(h('span', { class: 'small muted' }, 'Open again to recompute.'));
    } catch (e) {
      if (seq === this.seq) toast(`Ticket: ${e.message}`, { error: true });
    } finally {
      if (seq === this.seq) this.resultBox.classList.remove('busy');
    }
  }

  paintResult() {
    const info = store.status.games[this.game];
    const r = this.result;
    const rows = this.rows.filter((x) => x.complete);
    if (!r || !rows.length) {
      this.resultBox.replaceChildren(h('div', { class: 'empty' }, 'Add matches to the coupon — the ticket builds itself.'));
      return;
    }
    const partial = rows.length !== info.n;
    const missesAllowed = info.n - info.threshold;
    const thrLabel = partial ? `≤${missesAllowed} wrong` : `${info.threshold}+`;
    const every = r.p_prize > 0 ? Math.round(1 / r.p_prize) : null;
    const price = parseFloat(this.priceIn?.value || '');
    const tiers = Object.entries(r.tiers).map(([k, p]) => {
      const lbl = partial ? `${k}/${rows.length}` : (Number(k) === info.top ? `${k}` : `${k}+`);
      return h('div', { class: 'tier' }, h('span', {}, lbl),
        h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: `${Math.max(p * 100, 0.4)}%` } })),
        h('span', { class: 'v' }, pct(p, p < 0.01 ? 2 : 1)));
    });
    const unknown = rows.filter((x) => x.an && x.an.src === 'none').length;
    const warn = [];
    if (rows.length < info.n) warn.push(h('div', { class: 'infobox' }, `${rows.length} of ${info.n} matches so far — previewing the same tolerance as the real game (at most ${missesAllowed} wrong).`));
    if (rows.length > info.n) warn.push(h('div', { class: 'warnbox' }, `${rows.length} matches — ${info.label} has ${info.n}. Remove ${rows.length - info.n} (⋯ → Delete row).`));
    if (unknown) warn.push(h('div', { class: 'warnbox' }, `${unknown} match${unknown > 1 ? 'es are' : ' is'} counted as ⅓ each — fix the name or add odds.`));
    if (r.over_budget) warn.push(h('div', { class: 'warnbox' }, `Your locked picks alone need ${r.columns} columns — more than the budget of ${r.budget}.`));
    this.resultBox.replaceChildren(
      h('div', { class: 'muted small' }, `P(${thrLabel})`),
      h('div', { class: 'big' }, pct(r.p_prize, r.p_prize < 0.1 ? 2 : 1),
        every && every > 1 ? h('small', {}, `  ≈ once every ${every} coupons`) : ''),
      h('div', { class: 'small muted' }, `single column: ${pct(r.single.p_prize, 2)} · ×${r.single.p_prize > 0 ? (r.p_prize / r.single.p_prize).toFixed(1) : '—'}`),
      h('div', { class: 'tiers' }, ...tiers),
      h('dl', { class: 'kv' },
        h('dt', {}, 'Columns'), h('dd', {}, `${r.columns} / ${r.budget}` + (price > 0 ? `  ·  ${(r.columns * price).toFixed(2)}` : '')),
        h('dt', {}, 'Doubles · triples'), h('dd', {}, `${r.n_double} · ${r.n_triple}`),
        h('dt', {}, 'Expected correct'), h('dd', {}, `${r.expected.toFixed(1)} / ${rows.length}`),
        h('dt', {}, 'Optimised for'), h('dd', {}, r.objective === 'payout' ? 'crowd-aware' : 'hit chance')),
      ...warn.map((w) => h('div', { style: { marginTop: '8px' } }, w)));
  }

  async loadLadder() {
    const { rows, body } = this.ticketBody();
    if (!rows.length) { this.ladderBox.replaceChildren('Add matches first.'); return; }
    this.ladderBox.replaceChildren(h('span', { class: 'spin-sm' }), ' computing…');
    try {
      const { ladder } = await api.post('/api/ladder', body);
      const base = ladder[0]?.p_prize || 0;
      this.ladderBox.replaceChildren(h('table', { class: 'simple' },
        h('tr', {}, h('th', {}, 'Columns'), h('th', { class: 'r' }, 'P(prize)'), h('th', { class: 'r' }, 'vs single'), h('th', { class: 'r' }, '2× · 3×'), h('th', {})),
        ...ladder.map((x) => h('tr', {},
          h('td', {}, String(x.columns)), h('td', { class: 'r' }, pct(x.p_prize, 2)),
          h('td', { class: 'r' }, base > 0 ? `×${(x.p_prize / base).toFixed(1)}` : '—'),
          h('td', { class: 'r' }, `${x.doubles} · ${x.triples}`),
          h('td', { class: 'r' }, h('button', { class: 'btn sm ghost', onclick: () => { this.budget = x.columns; this.budgetIn.value = x.columns; this.paintPresets(); this.save(); this.retick(); } }, 'use'))))),
      h('div', { class: 'note' }, 'Cost grows with the columns, the chance of a prize much more slowly — pick a budget you are happy to lose. (Ignores locks and the crowd.)'));
    } catch (e) { this.ladderBox.replaceChildren(`Failed: ${e.message}`); }
  }

  async loadSweep() {
    const { rows, body } = this.ticketBody();
    if (!body.crowd) { this.sweepBox.replaceChildren('Needs crowd % on every match.'); return; }
    this.sweepBox.replaceChildren(h('span', { class: 'spin-sm' }), ' simulating (a few seconds)…');
    try {
      const { sweep } = await api.post('/api/sweep', body);
      this.sweepBox.replaceChildren(h('table', { class: 'simple' },
        h('tr', {}, h('th', {}, 'Tilt'), h('th', { class: 'r' }, 'P(prize)'), h('th', { class: 'r' }, 'Payout'), h('th', { class: 'r' }, 'Hits every')),
        ...sweep.map((x) => h('tr', {}, h('td', {}, x.tilt.toFixed(2)), h('td', { class: 'r' }, pct(x.p_hit, 2)),
          h('td', { class: 'r' }, `${x.payout.toFixed(2)}×`), h('td', { class: 'r' }, x.p_hit > 0 ? `${Math.round(1 / x.p_hit)} wk` : '—')))),
      h('div', { class: 'note' }, 'Payout is a crowd-weighted preference score relative to tilt 0, not money. Take the largest tilt whose hit rate you can live with.'));
      void rows;
    } catch (e) { this.sweepBox.replaceChildren(`Failed: ${e.message}`); }
  }

  ticketLines() {
    const rows = this.rows.filter((r) => r.complete);
    return rows.map((r, i) => ({ i: i + 1, home: r.home, away: r.away, play: r.lock || this.result?.rowLabels?.get(r) || '', probs: r.probs(), src: r.an?.src }));
  }

  copyTicket() {
    const info = store.status.games[this.game];
    const lines = this.ticketLines();
    const w = Math.max(...lines.map((l) => `${l.home} - ${l.away}`.length), 10);
    const txt = [`${info.label} — ${this.result?.columns || 1} columns · P(prize) ${pct(this.result?.p_prize, 2)}`,
      ...lines.map((l) => `${String(l.i).padStart(2)}  ${`${l.home} - ${l.away}`.padEnd(w)}  ${l.play}`)].join('\n');
    navigator.clipboard.writeText(txt).then(() => toast('Ticket copied to the clipboard'), () => toast('Copy failed', { error: true }));
  }

  printTicket() {
    const info = store.status.games[this.game];
    const lines = this.ticketLines();
    this.printArea.replaceChildren(
      h('h2', {}, `${info.flag} ${info.label} — ${this.result?.columns || 1} columns`),
      h('p', {}, `P(prize) ${pct(this.result?.p_prize, 2)} · printed ${new Date().toLocaleString()}`),
      h('table', {}, h('tr', {}, h('th', {}, '#'), h('th', {}, 'Match'), h('th', {}, 'Play')),
        ...lines.map((l) => h('tr', {}, h('td', {}, String(l.i)), h('td', {}, `${l.home} - ${l.away}`),
          h('td', {}, ...LABELS.map((s) => h('span', { class: `box${l.play.includes(s) ? ' x' : ''}` }, s)))))));
    window.print();
  }

  saveHistory() {
    const r = this.result;
    if (!r) { toast('Nothing to save yet'); return; }
    const note = h('input', { class: 'txt', placeholder: 'e.g. week 34' });
    note.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); note.closest('.modal')?.querySelector('.btn.primary')?.click(); } });
    const lines = this.ticketLines();
    modal('Save this ticket to history', h('div', {},
      h('p', { class: 'small muted' }, `${lines.length} matches · ${r.columns} columns · P(prize) ${pct(r.p_prize, 2)}. After the matches are played, History grades it against the real results.`),
      note), [
      ['Cancel'],
      ['Save', async () => {
        try {
          await api.post('/api/history', {
            game: this.game, note: note.value, budget: this.budget, columns: r.columns,
            p_prize: r.p_prize, kind: 'ticket',
            matches: lines.map((l) => ({ home: l.home, away: l.away, probs: l.probs, pick: l.play, src: l.src })),
          });
          toast('Saved — see 📈 History once the matches are played');
        } catch (e) { toast(`Save failed: ${e.message}`, { error: true }); return false; }
        return true;
      }, 'primary']]);
    setTimeout(() => note.focus(), 30);
  }
}

// Used by the other pages: append a match to a coupon (skipping duplicates).
export async function addToCoupon(game, home, away, odds) {
  const c = await api.get(`/api/coupon/${game}`);
  const key = [fold(home), fold(away)].sort().join('|');
  if (c.rows.some((r) => [fold(r.home), fold(r.away)].sort().join('|') === key)) {
    toast(`${home} – ${away} is already on the ${store.status.games[game].label} coupon`); return;
  }
  c.rows.push({ home, away, odds: odds || null });
  await api.put(`/api/coupon/${game}`, { rows: c.rows, budget: c.budget, tilt: c.tilt, objective: c.objective });
  const info = store.status.games[game];
  toast(`Added to ${info.flag} ${info.label} (${c.rows.length}/${info.n})`);
}

export function fairOdds(p) { return p.map(fair).join(' / '); }
