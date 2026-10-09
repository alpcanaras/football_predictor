// The coupon: one list of matches, shared by every part of the page.
//
// There is no game switch. Both pools pay a prize for at most three wrong, so
// the coupon is just the matches you added; 13 of them plays as the 13er
// Wette, 15 as Spor Toto, and any other size is previewed with the same rule.
// Rows price themselves (instantly when they came from the fixtures list), the
// ticket re-optimises on every change, and everything autosaves.
import { api, store } from './api.js';
import { debounce, norm, toast } from './ui.js';

const UNIFORM = [1 / 3, 1 / 3, 1 / 3];
const MASK = { '1': 1, X: 2, '2': 4, '1X': 3, '12': 5, X2: 6, '1X2': 7 };
const LABELS = ['1', 'X', '2'];
const oddsOk = (o) => Array.isArray(o) && o.length === 3 && o.every((x) => x > 1);

let uid = 0;
export class Row {
  constructor(d = {}) {
    this.id = ++uid;
    this.home = d.home || ''; this.away = d.away || '';
    this.odds = oddsOk(d.odds) ? d.odds : null;
    this.crowd = d.crowd || null; this.lock = d.lock || null;
    this.an = d.an || null; this.akey = d.an ? this.analysisKey() : null;
    this.busy = false;
  }
  get complete() { return !!(this.home.trim() && this.away.trim()); }
  analysisKey() {
    return `${norm(this.home)}|${norm(this.away)}|${this.odds ? this.odds.join(',') : ''}`;
  }
  probs() { return this.an?.probs || UNIFORM; }
  // identity independent of home/away order and of how the name was typed
  get key() {
    const s = this.an?.sides;
    const a = s?.home?.canonical || this.home; const b = s?.away?.canonical || this.away;
    return [norm(a), norm(b)].sort().join('|');
  }
  get homeDisplay() { return this.an?.sides?.home?.display || store.display(this.home); }
  get awayDisplay() { return this.an?.sides?.away?.display || store.display(this.away); }
}

export function pairKey(home, away) { return [norm(home), norm(away)].sort().join('|'); }

// What a coupon of n matches is, in words.
export function describe(n) {
  const [g13, g15] = store.status.games;
  const wrong = store.status.allowed_wrong;
  if (n === 0) return { n, state: 'empty', title: 'Your coupon is empty', hint: `Add matches from the list. 13 plays as the ${g13.label}, 15 as ${g15.label}.` };
  if (n === g13.n || n === g15.n) {
    const g = n === g13.n ? g13 : g15;
    return { n, state: 'ok', game: g, title: g.label, hint: `Prize at ${n - wrong}+ correct of ${n}` };
  }
  if (n < g13.n) return { n, state: 'partial', title: `${n} match${n > 1 ? 'es' : ''}`, hint: `${g13.n - n} more for the ${g13.label} · ${g15.n - n} for ${g15.label}` };
  if (n === 14) return { n, state: 'between', title: '14 matches', hint: `1 more → ${g15.label} · or remove 1 → ${g13.label}` };
  return { n, state: 'over', title: `${n} matches`, hint: `${g15.label} has ${g15.n} — remove ${n - g15.n}` };
}

export class Coupon {
  constructor() {
    this.rows = [];
    this.budget = 1; this.tilt = 0.5; this.objective = 'hit'; this.showCrowd = false;
    this.result = null; this.ready = false;
    this.listeners = new Set();
    this.cache = new Map();
    this.queue = []; this.inflight = 0;
    this.save = debounce(() => this.persist(), 500);
    this.retick = debounce(() => this.computeTicket(), 220);
    this.seq = 0;
  }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(what = 'rows') { for (const fn of this.listeners) fn(what); }

  get complete() { return this.rows.filter((r) => r.complete); }
  get n() { return this.complete.length; }
  get info() { return describe(this.n); }
  has(home, away) { const k = pairKey(home, away); return this.rows.some((r) => r.complete && r.key === k); }
  find(home, away) { const k = pairKey(home, away); return this.rows.find((r) => r.complete && r.key === k); }

  // ------------------------------------------------------------------ load
  async load() {
    const c = await api.get('/api/coupon');
    this.rows = c.rows.map((d) => new Row(d));
    this.budget = c.budget || 1; this.tilt = c.tilt ?? 0.5; this.objective = c.objective || 'hit';
    this.showCrowd = this.rows.some((r) => r.crowd);
    this.ready = true;
    this.rows.forEach((r) => this.analyze(r));
    this.emit('rows'); this.retick();
  }

  async persist() {
    if (!this.ready) return;
    const body = {
      rows: this.rows.filter((r) => r.home.trim() || r.away.trim())
        .map((r) => ({ home: r.home, away: r.away, odds: r.odds, crowd: r.crowd, lock: r.lock })),
      budget: this.budget, tilt: this.tilt, objective: this.objective,
    };
    try { await api.put('/api/coupon', body); this.emit('saved'); }
    catch (e) { toast(`Could not save the coupon: ${e.message}`, { error: true }); }
  }

  // ----------------------------------------------------------------- edit
  add(d, { silent = false } = {}) {
    if (!d.home || !d.away) return 'invalid';
    if (this.has(d.home, d.away)) {
      if (!silent) toast(`${store.display(d.home)} – ${store.display(d.away)} is already on the coupon`);
      return 'dup';
    }
    const row = new Row({ home: d.home, away: d.away, odds: d.odds, an: d.an });
    this.rows.push(row);
    this.emit('rows'); this.save(); this.retick();
    if (!row.an) this.analyze(row);
    if (!silent) {
      const i = this.rows.length;
      toast(`Added ${row.homeDisplay} – ${row.awayDisplay}  (${i})`, { action: 'Undo', key: 'add', onAction: () => this.remove(row, { quiet: true }) });
    }
    return 'added';
  }

  remove(row, { quiet = false } = {}) {
    const i = this.rows.indexOf(row);
    if (i < 0) return;
    this.rows.splice(i, 1);
    this.emit('rows'); this.save(); this.retick();
    if (!quiet) toast(`Removed ${row.homeDisplay} – ${row.awayDisplay}`, { key: 'add', action: 'Undo', onAction: () => { this.rows.splice(Math.min(i, this.rows.length), 0, row); this.emit('rows'); this.save(); this.retick(); } });
  }

  move(row, d) {
    const i = this.rows.indexOf(row); const j = i + d;
    if (i < 0 || j < 0 || j >= this.rows.length) return;
    [this.rows[i], this.rows[j]] = [this.rows[j], this.rows[i]];
    this.emit('rows'); this.save(); this.retick();
  }

  clear() {
    if (!this.rows.length) return;
    const keep = this.rows;
    this.rows = []; this.result = null;
    this.emit('rows'); this.save(); this.retick();
    toast('Coupon cleared', { action: 'Undo', onAction: () => { this.rows = keep; this.emit('rows'); this.save(); this.retick(); } });
  }

  edit(row, patch, { reanalyze = true } = {}) {
    Object.assign(row, patch);
    this.save();
    if (reanalyze) this.analyze(row, true);
    else { this.emit('row'); this.retick(); }
  }

  toggleSymbol(row, s) {
    const i = this.complete.indexOf(row);
    const cur = row.lock || this.result?.rowLabels?.get(row) || '';
    let set = new Set(cur.split('').filter(Boolean));
    if (!row.lock) set = new Set([s]);                // first click: lock exactly this
    else if (set.has(s)) set.delete(s); else set.add(s);
    const lab = LABELS.filter((x) => set.has(x)).join('');
    row.lock = lab && MASK[lab] ? lab : null;
    void i;
    this.emit('row'); this.save(); this.retick();
  }
  unlock(row) { row.lock = null; this.emit('row'); this.save(); this.retick(); }

  setBudget(b) { this.budget = Math.max(1, Math.round(b)); this.emit('budget'); this.save(); this.retick(); }
  setObjective(o) { this.objective = o; this.emit('budget'); this.save(); this.retick(); }
  setTilt(t) { this.tilt = t; this.save(); this.retick(); }

  // -------------------------------------------------------------- analysis
  pump() {
    while (this.inflight < 3 && this.queue.length) {
      const job = this.queue.shift(); this.inflight += 1;
      job().finally(() => { this.inflight -= 1; this.pump(); });
    }
  }

  analyze(row, force = false) {
    if (!row.complete) { row.an = null; row.akey = null; this.emit('row'); this.retick(); return; }
    const key = row.analysisKey();
    if (!force && key === row.akey && row.an) return;
    row.akey = key;
    if (this.cache.has(key)) { this.apply(row, this.cache.get(key)); return; }
    row.busy = true; this.emit('row');
    this.queue.push(async () => {
      try {
        const res = await api.post('/api/analyze', { home: row.home, away: row.away, odds: row.odds });
        this.cache.set(key, res);
        if (row.akey === key) this.apply(row, res);
      } catch (e) {
        toast(`Could not price ${row.home} – ${row.away}: ${e.message}`, { error: true });
      } finally { row.busy = false; this.emit('row'); }
    });
    this.pump();
  }

  apply(row, res) {
    row.an = res; row.busy = false;
    // a name we are sure about is stored as the data's own name, so the same
    // match is the same row everywhere (and the file stays tidy)
    const sure = (s) => s?.canonical && s.how && s.how !== 'fuzzy';
    let changed = false;
    if (sure(res.sides?.home) && res.sides.home.canonical !== row.home) { row.home = res.sides.home.canonical; changed = true; }
    if (sure(res.sides?.away) && res.sides.away.canonical !== row.away) { row.away = res.sides.away.canonical; changed = true; }
    if (changed) { row.akey = row.analysisKey(); this.cache.set(row.akey, res); this.save(); }
    this.emit('row'); this.retick();
  }

  async confirmName(row, side, canonical) {
    const typed = row[side];
    try { await api.post('/api/names/remember', { typed, canonical }); } catch { /* best effort */ }
    this.edit(row, { [side]: canonical });
  }

  // ---------------------------------------------------------------- ticket
  ticketBody() {
    const rows = this.complete;
    return {
      rows,
      body: {
        budget: this.budget, objective: this.objective, tilt: this.tilt,
        probs: rows.map((r) => r.probs()), locks: rows.map((r) => r.lock),
        crowd: rows.length && rows.every((r) => r.crowd) ? rows.map((r) => r.crowd) : null,
      },
    };
  }

  async computeTicket() {
    const { rows, body } = this.ticketBody();
    if (!rows.length) { this.result = null; this.emit('ticket'); return; }
    const seq = ++this.seq;
    this.thinking = true; this.emit('ticket');
    try {
      const res = await api.post('/api/ticket', body);
      if (seq !== this.seq) return;                // a newer request is on its way
      res.rowLabels = new Map(rows.map((r, i) => [r, res.labels[i]]));
      this.result = res;
    } catch (e) {
      if (seq === this.seq) toast(`Ticket: ${e.message}`, { error: true });
    } finally {
      if (seq === this.seq) { this.thinking = false; this.emit('ticket'); }
    }
  }

  labelOf(row) { return row.lock || this.result?.rowLabels?.get(row) || ''; }

  // a fixture from the list -> a row that is priced from the first paint
  static fromFixture(f) {
    const known = f.probs && f.src;
    return {
      home: f.home, away: f.away, odds: f.odds,
      an: known ? {
        home: f.home, away: f.away, probs: f.probs, src: f.src, market: f.market, model: f.model,
        ou25: f.ou25, btts: f.btts, score: f.score, score_p: f.score_p, xg: f.xg, league_name: f.league_name,
        sides: { home: { canonical: f.home, display: f.home_display, kind: 'club', how: 'exact', suggest: [] },
          away: { canonical: f.away, display: f.away_display, kind: 'club', how: 'exact', suggest: [] } },
      } : null,
    };
  }
}
