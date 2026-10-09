// The coupon slip (right-hand side): the matches you picked, a pick chip per
// outcome, and the ticket they add up to. It is always visible while you browse
// fixtures, and edits happen in place — no re-render under your cursor.
import { h, pct, pct3, fair, norm, textField, Autocomplete, badge, toast, modal } from './ui.js';
import { api, store } from './api.js';
import { showDetails } from './details.js';

const LABELS = ['1', 'X', '2'];
const PRESETS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256];
const parseNum = (s) => { const v = parseFloat(String(s).replace(',', '.')); return Number.isFinite(v) ? v : null; };

export class SlipView {
  constructor(root, coupon) {
    this.root = root; this.c = coupon; this.views = new Map(); this.more = false;
    this.unsub = coupon.on((w) => this.onChange(w));
    this.build();
    this.onChange('rows');
  }
  destroy() { this.unsub(); }

  // ----------------------------------------------------------------- layout
  build() {
    this.title = h('div', { class: 'slip-title' });
    this.menuBtn = h('button', { class: 'iconbtn', title: 'More', onclick: (e) => this.menu(e.currentTarget) }, '⋯');
    this.progress = h('div', { class: 'progress' });
    this.hint = h('div', { class: 'small muted hintline' });
    const home = textField({ placeholder: 'Add a match: home team…', cls: 'sm' });
    const away = textField({ placeholder: 'away team…', cls: 'sm' });
    this.qa = { home, away };
    const go = () => this.quickAdd();
    new Autocomplete(home.input, { source: () => store.teams, onPick: () => setTimeout(() => away.input.focus(), 0) });
    new Autocomplete(away.input, { source: () => store.teams, onPick: () => setTimeout(go, 0) });
    home.input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !home.wrap.querySelector('.ac-list')) { e.preventDefault(); away.input.focus(); } });
    away.input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !away.wrap.querySelector('.ac-list')) { e.preventDefault(); go(); } });
    for (const f of [home, away]) {                      // paste "Turkey - France" or a whole coupon
      f.input.addEventListener('paste', (e) => {
        const t = (e.clipboardData.getData('text') || '').trim();
        if (/\n/.test(t) || /\s(-|–|v|vs)\s/i.test(t)) { e.preventDefault(); this.pasteText(t); f.set(''); }
      });
    }
    this.quick = h('div', { class: 'quick' }, home.wrap, away.wrap,
      h('button', { class: 'btn primary sm', onclick: go, title: 'Add this match' }, '＋'));
    this.rowsBox = h('div', { class: 'slip-rows' });
    this.summary = h('div', { class: 'summary' });
    this.root.replaceChildren(
      h('div', { class: 'slip-head' }, this.title, h('span', { class: 'spacer' }), this.menuBtn),
      this.progress, this.hint, this.quick, this.rowsBox, this.summary);
    this.buildSummary();
  }

  menu(anchor) {
    const items = [
      ['📋 Paste a coupon…', () => this.pasteDialog()],
      ['📡 Fill missing odds from the feed', () => this.fillOdds()],
      [this.c.showCrowd ? '👥 Hide crowd %' : '👥 Crowd % column', () => { this.c.showCrowd = !this.c.showCrowd; this.onChange('rows'); }],
      this.c.showCrowd ? ['📋 Paste crowd % for all', () => this.pasteCrowd()] : null,
      ['🗑 Clear the coupon', () => this.c.clear()],
    ].filter(Boolean);
    const m = h('div', { class: 'menu' }, ...items.map(([t, fn]) => h('button', { onmousedown: (e) => { e.preventDefault(); m.remove(); fn(); } }, t)));
    const r = anchor.getBoundingClientRect();
    Object.assign(m.style, { position: 'fixed', top: `${r.bottom + 4}px`, right: `${window.innerWidth - r.right}px` });
    document.body.append(m);
    const off = (e) => { if (!m.contains(e.target)) { m.remove(); document.removeEventListener('mousedown', off); } };
    setTimeout(() => document.addEventListener('mousedown', off), 0);
  }

  // ------------------------------------------------------------------ events
  onChange(what) {
    if (what === 'rows') { this.syncRows(); this.paintHeader(); this.paintSummary(); }
    else if (what === 'row') { this.views.forEach((v, row) => this.paintRow(row, v)); this.paintHeader(); this.paintSummary(); }
    else if (what === 'ticket') { this.views.forEach((v, row) => this.paintPicks(row, v)); this.paintSummary(); }
    else if (what === 'budget') { this.paintSummary(); }
  }

  paintHeader() {
    const info = this.c.info; const n = this.c.n;
    const [g13, g15] = store.status.games;
    this.title.replaceChildren(h('span', { class: `gname s-${info.state}` }, info.title), n ? h('span', { class: 'gcount' }, `${n}`) : '');
    const pctOf = (k) => `${Math.min(100, (k / g15.n) * 100)}%`;
    const ms = (g) => h('span', { class: `mile${n === g.n ? ' now' : n > g.n ? ' past' : ''}`, title: `${g.label}: ${g.n} matches, prize at ${g.prize_from}+` },
      h('b', {}, g.n), ` ${g.flag} ${g.label}`);
    this.progress.replaceChildren(
      h('div', { class: 'ptrack' }, h('div', { class: `pfill s-${info.state}`, style: { width: pctOf(n) } }),
        h('i', { class: `pdot${n >= g13.n ? ' hit' : ''}`, style: { left: pctOf(g13.n) } }),
        h('i', { class: `pdot${n >= g15.n ? ' hit' : ''}`, style: { left: pctOf(g15.n) } })),
      h('div', { class: 'miles' }, ms(g13), ms(g15)));
    this.hint.textContent = info.hint;
    this.hint.className = `small hintline ${info.state === 'over' || info.state === 'between' ? 'warn' : 'muted'}`;
  }

  // -------------------------------------------------------------------- rows
  syncRows() {
    const seen = new Set();
    const els = [];
    this.c.rows.forEach((row, i) => {
      seen.add(row);
      let v = this.views.get(row);
      if (!v) { v = this.buildRow(row); this.views.set(row, v); }
      v.idx.textContent = String(i + 1);
      els.push(v.el);
      this.paintRow(row, v);
    });
    for (const row of [...this.views.keys()]) if (!seen.has(row)) this.views.delete(row);
    if (!els.length) els.push(this.emptyState());
    this.rowsBox.replaceChildren(...els);
    this.views.forEach((v) => v.el.classList.toggle('with-crowd', this.c.showCrowd));
  }

  emptyState() {
    const box = h('div', { class: 'slip-empty' },
      h('div', { class: 'big-emoji' }, '🎟️'),
      h('b', {}, 'Pick matches from the list'),
      h('div', { class: 'small muted' }, 'Hover a fixture and press ＋ — or type teams above, or paste a whole coupon. 13 matches plays as the 13er Wette, 15 as Spor Toto.'));
    api.get('/api/legacy').then((leg) => {
      const bits = Object.entries(leg).filter(([, rows]) => rows.length);
      if (!bits.length) return;
      box.append(h('div', { class: 'small', style: { marginTop: '10px' } }, 'Import from your old coupon: ',
        ...bits.map(([g, rows]) => h('button', { class: 'chip', onclick: () => { rows.forEach((r) => this.c.add(r, { silent: true })); toast(`Imported ${rows.length} matches`); } }, `${g === 'turkish' ? '🇹🇷' : '🇩🇪'} ${rows.length} matches`))));
    }).catch(() => {});
    return box;
  }

  buildRow(row) {
    const mk = (side) => {
      const f = textField({ value: '', placeholder: side === 'home' ? 'Home' : 'Away', cls: 'bare',
        onCommit: (v) => {
          const typed = v.trim();
          const shown = side === 'home' ? row.homeDisplay : row.awayDisplay;
          if (norm(typed) === norm(shown)) { f.set(shown); return; }
          this.c.edit(row, { [side]: typed });
        } });
      new Autocomplete(f.input, { source: () => store.teams, onPick: (t) => this.c.edit(row, { [side]: t.name }) });
      f.input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !f.wrap.querySelector('.ac-list')) f.input.blur(); });
      return f;
    };
    const home = mk('home'); const away = mk('away');
    const picks = LABELS.map((s, i) => h('button', { class: 'pk', 'data-s': s, onclick: () => this.c.toggleSymbol(row, s) }, h('b', {}, s), h('span', {}, '·')));
    const lockBtn = h('button', { class: 'unlock', title: 'Locked by you — click to hand this match back to the optimiser', onclick: () => this.c.unlock(row) }, '🔒');
    const remove = h('button', { class: 'sx', title: 'Remove from the coupon', onclick: () => this.c.remove(row) }, '✕');
    const detail = h('div', { class: 'sdetail' });
    const sug = h('div', { class: 'ssug' });
    const crowd = textField({ placeholder: 'crowd % 1/X/2', cls: 'bare xs', onCommit: (v) => {
      const t = v.trim().split(/[\s/;,]+/).map(parseNum).filter((x) => x != null);
      this.c.edit(row, { crowd: t.length === 3 && t.reduce((a, b) => a + b, 0) > 0 ? t : null }, { reanalyze: false });
      crowd.set(row.crowd ? row.crowd.map((x) => Math.round(x)).join('/') : '');
    } });
    const idx = h('div', { class: 'sidx' });
    const open = h('button', { class: 'iconbtn tiny', title: 'Full breakdown', onclick: () => showDetails(row.home, row.away, this.c, { odds: row.odds }) }, 'ⓘ');
    const el = h('div', { class: 'srow' }, idx,
      h('div', { class: 'steams' }, home.wrap, away.wrap),
      h('div', { class: 'spicks' }, ...picks, lockBtn),
      h('div', { class: 'sside' }, remove, open),
      detail, sug, h('div', { class: 'scrowd' }, crowd.wrap));
    return { el, idx, home, away, picks, lockBtn, detail, sug, crowd, editing: false };
  }

  paintRow(row, v) {
    const a = row.an;
    if (document.activeElement !== v.home.input) v.home.set(row.homeDisplay);
    if (document.activeElement !== v.away.input) v.away.set(row.awayDisplay);
    if (document.activeElement !== v.crowd.input) v.crowd.set(row.crowd ? row.crowd.map((x) => Math.round(x)).join('/') : '');
    const sides = a?.sides || {};
    const bad = (s) => a && s && !s.canonical;
    v.home.input.classList.toggle('bad', !!bad(sides.home));
    v.away.input.classList.toggle('bad', !!bad(sides.away));
    v.el.classList.toggle('unknown', !!(bad(sides.home) || bad(sides.away)));
    v.el.classList.toggle('busy', row.busy);
    v.el.classList.toggle('dup', this.c.rows.some((r) => r !== row && r.complete && r.key === row.key));
    this.paintPicks(row, v);
    // always-on details line
    const bits = [];
    if (a?.src) bits.push(badge(a.src));
    else if (row.busy) bits.push(h('span', { class: 'spin-sm' }));
    bits.push(this.oddsPart(row, v));
    if (a?.v2) bits.push(h('span', { class: 'cmp', title: 'Model v2' }, h('b', {}, 'v2 '), pct3(a.v2)));
    if (a?.v1) bits.push(h('span', { class: 'cmp', title: 'Model v1' }, h('b', {}, 'v1 '), pct3(a.v1)));
    v.detail.replaceChildren(...bits);
    v.el.title = a?.league_name || '';
    // suggestions for names we could not place, and a nudge for guesses
    const chips = [];
    for (const side of ['home', 'away']) {
      const s = sides[side];
      if (!a || !s) continue;
      if (!s.canonical) {
        chips.push(h('span', {}, `“${row[side]}” not recognised`), s.suggest?.length ? ' → ' : ' — add odds for this match',
          ...(s.suggest || []).map((o) => h('button', { class: 'chip', title: o.league_name, onclick: () => this.c.confirmName(row, side, o.name) }, o.display)));
      } else if (s.how === 'fuzzy') {
        chips.push(h('span', {}, `“${row[side]}” looks like `), h('button', { class: 'chip', onclick: () => this.c.confirmName(row, side, s.canonical) }, `${s.display} ✓`));
      }
    }
    v.sug.replaceChildren(...chips);
  }

  oddsPart(row, v) {
    if (v.editing) {
      const ins = [0, 1, 2].map((k) => h('input', { class: 'oi', value: row.odds ? row.odds[k].toFixed(2) : '', placeholder: LABELS[k], inputmode: 'decimal' }));
      const done = () => {
        const vals = ins.map((i) => parseNum(i.value));
        v.editing = false;
        this.c.edit(row, { odds: vals.every((x) => x && x > 1) ? vals : null });
        this.paintRow(row, v);
      };
      ins.forEach((i, k) => {
        i.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); done(); } if (e.key === 'Escape') { v.editing = false; this.paintRow(row, v); } });
        i.addEventListener('paste', (e) => {
          const t = e.clipboardData.getData('text').trim().split(/[\s/;]+/).map(parseNum);
          if (t.length === 3 && t.every((x) => x && x > 1)) { e.preventDefault(); ins.forEach((o, j) => { o.value = t[j].toFixed(2); }); }
        });
      });
      setTimeout(() => ins[0].focus(), 0);
      return h('span', { class: 'oedit' }, ...ins, h('button', { class: 'chip', onclick: done }, 'ok'));
    }
    const edit = () => { v.editing = true; this.paintRow(row, v); };
    return row.odds
      ? h('span', { class: 'odds', title: 'Click to edit the odds' }, h('button', { class: 'linkbtn', onclick: edit }, `odds ${row.odds.map((o) => o.toFixed(2)).join(' / ')} ✎`))
      : h('button', { class: 'linkbtn faint', onclick: edit, title: 'Typing the bookmaker odds makes the probabilities much sharper' }, '＋ odds');
  }

  paintPicks(row, v) {
    const p = row.probs();
    const label = this.c.labelOf(row);
    v.picks.forEach((b, k) => {
      b.classList.toggle('sel', label.includes(LABELS[k]));
      b.classList.toggle('lockd', !!row.lock && label.includes(LABELS[k]));
      b.lastChild.textContent = row.an?.probs ? Math.round(p[k] * 100) : '·';
      b.title = row.an?.probs ? `${LABELS[k]}: ${pct(p[k], 1)} · fair odds ${fair(p[k])} — click to play it yourself` : 'pricing…';
    });
    v.lockBtn.hidden = !row.lock;
  }

  // ------------------------------------------------------------------- quick add
  quickAdd() {
    const h1 = this.qa.home.input.value.trim(); const a1 = this.qa.away.input.value.trim();
    if (!h1 || !a1) { (h1 ? this.qa.away : this.qa.home).input.focus(); return; }
    const resolve = (typed) => store.teams.find((t) => norm(t.display) === norm(typed) || t.name === typed)?.name || typed;
    this.c.add({ home: resolve(h1), away: resolve(a1) });
    this.qa.home.set(''); this.qa.away.set('');
    this.qa.home.input.focus();
  }

  async pasteText(text) {
    try {
      const { rows } = await api.post('/api/parse', { text });
      if (!rows.length) { toast('Nothing recognisable to add'); return; }
      rows.forEach((r) => this.c.add(r, { silent: true }));
      toast(`Added ${rows.length} match${rows.length > 1 ? 'es' : ''} (${this.c.n} on the coupon)`);
    } catch (e) { toast(e.message, { error: true }); }
  }

  pasteDialog() {
    const ta = h('textarea', { class: 'txt', rows: 12, placeholder: 'Galatasaray - Fenerbahçe\nBayern München - Dortmund  1.50 4.60 6.00\nTurkey - Spain\n…' });
    modal('Paste a coupon', h('div', {},
      h('div', { class: 'small muted', style: { margin: '0 0 8px' } }, 'One match per line: “Home - Away”, optionally followed by the three odds. “v”, “vs” and “:” work too. Full names are fine.'), ta), [
      ['Cancel'],
      ['Add to coupon', async () => { await this.pasteText(ta.value); }],
      ['Replace coupon', async () => { this.c.rows = []; await this.pasteText(ta.value); }, 'primary']]);
    setTimeout(() => ta.focus(), 30);
  }

  pasteCrowd() {
    const rows = this.c.complete;
    const ta = h('textarea', { class: 'txt', rows: Math.min(Math.max(rows.length, 6), 16), placeholder: '68/21/11\n45/30/25\n…' });
    modal('Paste crowd percentages', h('div', {},
      h('div', { class: 'small muted', style: { marginBottom: '8px' } }, `One line per match in coupon order (${rows.length} matches): the share of players on 1, X and 2 — “68/21/11”, “68 21 11” or “68;21;11”.`), ta), [
      ['Cancel'],
      ['Apply', () => {
        const lines = ta.value.split(/\n/).map((l) => l.trim().split(/[\s/;,]+/).map(parseNum).filter((x) => x != null)).filter((t) => t.length >= 3);
        if (!lines.length) { toast('No “1/X/2” lines found'); return false; }
        rows.forEach((r, i) => { if (lines[i]) r.crowd = lines[i].slice(-3); });
        this.c.save(); this.c.emit('row'); this.c.retick();
        toast(`Crowd % set for ${Math.min(lines.length, rows.length)} of ${rows.length} matches` + (lines.length !== rows.length ? ` — ${lines.length} lines for ${rows.length} matches, check the order` : ''));
        return true;
      }, 'primary']]);
    setTimeout(() => ta.focus(), 30);
  }

  async fillOdds() {
    const targets = this.c.complete.filter((r) => !r.odds);
    if (!targets.length) { toast('Every match already has odds'); return; }
    toast('Looking the matches up in the odds feed…');
    try {
      const { odds } = await api.post('/api/fill-odds', { rows: targets.map((r) => ({ home: r.home, away: r.away })) });
      let n = 0;
      targets.forEach((r, k) => { if (odds[k]) { this.c.edit(r, { odds: odds[k] }); n += 1; } });
      toast(n ? `Filled odds for ${n} of ${targets.length}` + (n < targets.length ? ' — the rest are not in the feed yet (it covers about a week ahead)' : '')
        : 'None of these are in the feed yet — it covers about a week ahead');
    } catch (e) { toast(`Feed unavailable: ${e.message}`, { error: true }); }
  }

  // ------------------------------------------------------------------ summary
  buildSummary() {
    this.pBig = h('div', { class: 'psum' });
    this.budgetIn = h('input', { class: 'txt num sm', type: 'number', min: 1, max: 100000, value: this.c.budget });
    this.budgetIn.addEventListener('input', () => { const b = parseInt(this.budgetIn.value, 10); if (b >= 1) this.c.setBudget(b); });
    this.presets = h('div', { class: 'presets' }, ...PRESETS.map((p) => h('button', { onclick: () => { this.budgetIn.value = p; this.c.setBudget(p); } }, String(p))));
    this.tiers = h('div', { class: 'tiers-strip' });
    this.notes = h('div', { class: 'notes' });
    this.moreBox = h('div', { class: 'more', hidden: true });
    this.moreBtn = h('button', { class: 'btn sm ghost', onclick: () => this.toggleMore() }, 'More ▾');
    const acts = h('div', { class: 'row sum-actions' },
      h('button', { class: 'btn sm', onclick: () => this.copy() }, '📋 Copy'),
      h('button', { class: 'btn sm', onclick: () => this.print() }, '🖨 Print'),
      this.moreBtn, h('span', { class: 'spacer' }),
      h('button', { class: 'btn sm primary', onclick: () => this.saveHistory() }, '💾 Save'));
    this.summary.replaceChildren(this.pBig,
      h('div', { class: 'brow' }, h('span', { class: 'muted small' }, 'Budget'), this.budgetIn, h('span', { class: 'muted small' }, 'columns'), this.price = h('input', { class: 'txt num xs', type: 'number', min: 0, step: '0.01', placeholder: 'price/col', title: 'Price of one column — shows the total cost', value: localStorage.getItem('fp.price') || '' })),
      this.presets, this.tiers, this.notes, acts, this.moreBox);
    this.price.addEventListener('input', () => { localStorage.setItem('fp.price', this.price.value); this.paintSummary(); });
  }

  paintSummary() {
    const c = this.c; const r = c.result; const n = c.n;
    if (document.activeElement !== this.budgetIn) this.budgetIn.value = c.budget;
    [...this.presets.children].forEach((b) => b.classList.toggle('on', Number(b.textContent) === c.budget));
    this.summary.classList.toggle('busy', !!c.thinking);
    const info = c.info;
    if (!r || !n) {
      this.pBig.replaceChildren(h('div', { class: 'muted small' }, 'Your chance of a prize'), h('div', { class: 'big faint' }, '—'));
      this.tiers.replaceChildren(); this.notes.replaceChildren();
      return;
    }
    const wrong = store.status.allowed_wrong;
    const every = r.p_prize > 0 ? Math.round(1 / r.p_prize) : null;
    const price = parseFloat(this.price.value || '');
    this.pBig.replaceChildren(
      h('div', { class: 'muted small' }, `Chance of ${r.threshold}+ correct of ${n}`, info.state === 'ok' ? '' : h('span', { class: 'preview' }, ' preview')),
      h('div', { class: 'row baseline' },
        h('div', { class: 'big' }, pct(r.p_prize, r.p_prize < 0.1 ? 2 : 1)),
        h('div', { class: 'psub' }, every && every > 1 ? `≈ 1 in ${every}` : '',
          h('span', { class: 'muted' }, ` · ${r.columns} column${r.columns > 1 ? 's' : ''}${price > 0 ? ` = ${(r.columns * price).toFixed(2)}` : ''}`))));
    this.tiers.replaceChildren(...Object.entries(r.tiers).map(([k, p]) => h('div', { class: 'tier-chip', title: `Chance of exactly reaching ${k}+ correct` },
      h('b', {}, Number(k) === n ? `${k}` : `${k}+`), h('span', {}, pct(p, p < 0.01 ? 2 : 1)))));
    const notes = [];
    const unk = c.complete.filter((x) => x.an && x.an.src === 'none').length;
    if (unk) notes.push(h('div', { class: 'warnbox' }, `${unk} match${unk > 1 ? 'es are' : ' is'} counted as ⅓ each — fix the name or add odds.`));
    if (r.over_budget) notes.push(h('div', { class: 'warnbox' }, `Your locked picks alone need ${r.columns} columns — more than the budget of ${r.budget}.`));
    this.notes.replaceChildren(...notes);
    if (this.more) this.paintMore();
  }

  toggleMore() { this.more = !this.more; this.moreBox.hidden = !this.more; this.moreBtn.textContent = this.more ? 'Less ▴' : 'More ▾'; if (this.more) this.paintMore(); }

  paintMore() {
    const c = this.c;
    const obj = h('div', { class: 'seg' },
      h('button', { class: `seg-btn${c.objective === 'hit' ? ' on' : ''}`, onclick: () => c.setObjective('hit'), title: 'Maximise the chance of a prize' }, 'Hit chance'),
      h('button', { class: `seg-btn${c.objective === 'crowd' ? ' on' : ''}`, disabled: !(c.complete.length && c.complete.every((r) => r.crowd)), onclick: () => c.setObjective('crowd'), title: 'Lean toward outcomes the crowd under-plays (needs crowd % on every match)' }, 'Crowd-aware'));
    const tilt = h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: c.tilt, style: { width: '100%' } });
    tilt.addEventListener('input', () => c.setTilt(parseFloat(tilt.value)));
    const ladder = h('div', { class: 'small muted' }, 'Computing…');
    const hasCrowd = c.complete.length && c.complete.every((r) => r.crowd);
    this.moreBox.replaceChildren(
      h('div', { class: 'mrow' }, h('span', { class: 'muted small' }, 'Optimise for'), obj),
      c.objective === 'crowd' ? h('div', { class: 'small' }, h('div', { class: 'muted' }, 'Lean on the crowd (0 = pure hit chance, above ~0.75 the hit rate collapses)'), tilt) : '',
      h('div', { class: 'mrow' }, h('button', { class: 'btn sm', onclick: () => this.showLadder() }, '💸 What does more budget buy?'),
        hasCrowd ? h('button', { class: 'btn sm', onclick: () => this.showSweep() }, '⚖️ Hit chance vs payout') : ''),
      c.result ? h('div', { class: 'small muted' }, `Single column: ${pct(c.result.single.p_prize, 2)} → ${c.result.single.p_prize > 0 ? `×${(c.result.p_prize / c.result.single.p_prize).toFixed(1)}` : '—'} with this system · ${c.result.n_double} doubles, ${c.result.n_triple} triples · expect ${c.result.expected.toFixed(1)} right`) : '',
      h('div', { class: 'tiny faint' }, 'Click 1 / X / 2 on any match to play it yourself; the rest of the ticket is optimised around your picks (🔒).'));
    void ladder;
  }

  async showLadder() {
    const { body } = this.c.ticketBody();
    const out = h('div', {}, h('span', { class: 'spin-sm' }), ' computing…');
    modal('What does more budget buy?', out, [['Close']], { wide: true });
    try {
      const { ladder } = await api.post('/api/ladder', body);
      const base = ladder[0]?.p_prize || 0;
      out.replaceChildren(h('table', { class: 'simple' },
        h('tr', {}, h('th', {}, 'Columns'), h('th', { class: 'r' }, 'P(prize)'), h('th', { class: 'r' }, 'vs single'), h('th', { class: 'r' }, 'doubles · triples'), h('th', {})),
        ...ladder.map((x) => h('tr', {}, h('td', {}, String(x.columns)), h('td', { class: 'r' }, pct(x.p_prize, 2)),
          h('td', { class: 'r' }, base > 0 ? `×${(x.p_prize / base).toFixed(1)}` : '—'), h('td', { class: 'r' }, `${x.doubles} · ${x.triples}`),
          h('td', { class: 'r' }, h('button', { class: 'btn sm ghost', onclick: () => { this.c.setBudget(x.columns); } }, 'use'))))),
      h('div', { class: 'note' }, 'Cost grows with the columns, the chance of a prize much more slowly — pick a budget you are happy to lose. (Ignores locked picks and the crowd.)'));
    } catch (e) { out.textContent = `Failed: ${e.message}`; }
  }

  async showSweep() {
    const { body } = this.c.ticketBody();
    const out = h('div', {}, h('span', { class: 'spin-sm' }), ' simulating (a few seconds)…');
    modal('Hit chance vs crowd payout', out, [['Close']], { wide: true });
    try {
      const { sweep } = await api.post('/api/sweep', body);
      out.replaceChildren(h('table', { class: 'simple' },
        h('tr', {}, h('th', {}, 'Tilt'), h('th', { class: 'r' }, 'P(prize)'), h('th', { class: 'r' }, 'Payout'), h('th', { class: 'r' }, 'Hits every')),
        ...sweep.map((x) => h('tr', {}, h('td', {}, x.tilt.toFixed(2)), h('td', { class: 'r' }, pct(x.p_hit, 2)), h('td', { class: 'r' }, `${x.payout.toFixed(2)}×`), h('td', { class: 'r' }, x.p_hit > 0 ? `${Math.round(1 / x.p_hit)} wk` : '—')))),
      h('div', { class: 'note' }, 'Payout is a crowd-weighted preference score relative to tilt 0, not money. Take the largest tilt whose hit rate you can live with.'));
    } catch (e) { out.textContent = `Failed: ${e.message}`; }
  }

  // ------------------------------------------------------------------ output
  lines() {
    return this.c.complete.map((r, i) => ({ i: i + 1, home: r.homeDisplay, away: r.awayDisplay, play: this.c.labelOf(r), probs: r.probs(), src: r.an?.src, rawHome: r.home, rawAway: r.away }));
  }

  copy() {
    const info = this.c.info; const r = this.c.result;
    const lines = this.lines();
    const w = Math.max(...lines.map((l) => `${l.home} - ${l.away}`.length), 10);
    const txt = [`${info.title.replace(/^[^\w]+/, '')} — ${r?.columns || 1} columns · P(prize) ${pct(r?.p_prize, 2)}`,
      ...lines.map((l) => `${String(l.i).padStart(2)}  ${`${l.home} - ${l.away}`.padEnd(w)}  ${l.play}`)].join('\n');
    navigator.clipboard.writeText(txt).then(() => toast('Ticket copied to the clipboard'), () => toast('Copy failed', { error: true }));
  }

  print() {
    const info = this.c.info; const r = this.c.result;
    const lines = this.lines();
    const area = document.getElementById('print-area');
    area.replaceChildren(
      h('h2', {}, `${info.title} — ${r?.columns || 1} columns`),
      h('p', {}, `P(prize) ${pct(r?.p_prize, 2)} · printed ${new Date().toLocaleString()}`),
      h('table', {}, h('tr', {}, h('th', {}, '#'), h('th', {}, 'Match'), h('th', {}, 'Play')),
        ...lines.map((l) => h('tr', {}, h('td', {}, String(l.i)), h('td', {}, `${l.home} - ${l.away}`),
          h('td', {}, ...LABELS.map((s) => h('span', { class: `box${l.play.includes(s) ? ' x' : ''}` }, s)))))));
    window.print();
  }

  saveHistory() {
    const r = this.c.result;
    if (!r) { toast('Nothing to save yet'); return; }
    const note = h('input', { class: 'txt', placeholder: 'e.g. week 41' });
    const lines = this.lines();
    const save = async () => {
      try {
        await api.post('/api/history', {
          note: note.value, budget: this.c.budget, columns: r.columns, p_prize: r.p_prize, kind: 'ticket',
          matches: lines.map((l) => ({ home: l.rawHome, away: l.rawAway, probs: l.probs, pick: l.play, src: l.src })),
        });
        toast('Saved — see History once the matches are played');
      } catch (e) { toast(`Save failed: ${e.message}`, { error: true }); return false; }
      return true;
    };
    note.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); note.closest('.modal')?.querySelector('.btn.primary')?.click(); } });
    modal('Save this ticket to history', h('div', {},
      h('p', { class: 'small muted' }, `${lines.length} matches · ${r.columns} columns · P(prize) ${pct(r.p_prize, 2)}. After the matches are played, History grades it against the real results.`), note),
    [['Cancel'], ['Save', save, 'primary']]);
    setTimeout(() => note.focus(), 30);
  }
}
