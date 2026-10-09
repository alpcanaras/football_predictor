// The coupon (right-hand side): the matches you picked, the result chips for
// each, and the ticket they add up to. It stays in view while you browse the
// fixtures, and edits happen in place — nothing re-renders under your cursor.
import { h, pct, fair, norm, textField, Autocomplete, toast, modal, SRC_TEXT } from './ui.js';
import { icon } from './icons.js';
import { api, store } from './api.js';
import { showDetails } from './details.js';

const LABELS = ['1', 'X', '2'];
const PRESETS = [1, 2, 4, 8, 16, 32, 64, 128];
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
    this.title = h('span', { class: 'slip-title' });
    this.countEl = h('span', { class: 'slip-count' });
    this.menuBtn = h('button', { class: 'iconbtn', title: 'Coupon actions', 'aria-label': 'Coupon actions', onclick: (e) => this.menu(e.currentTarget) }, icon('more', 18));
    this.progress = h('div', { class: 'progress' });
    this.hint = h('div', { class: 'hintline' });
    const home = textField({ placeholder: 'Add a match: home team', cls: 'sm' });
    const away = textField({ placeholder: 'away team', cls: 'sm' });
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
      h('button', { class: 'btn primary sm', style: { height: '28px', padding: '0', width: '32px', justifyContent: 'center' }, onclick: go, title: 'Add this match', 'aria-label': 'Add this match' }, icon('plus', 16)));
    this.rowsBox = h('div', { class: 'slip-rows' });
    this.summary = h('div', { class: 'summary' });
    this.root.replaceChildren(
      h('div', { class: 'slip-head' }, this.title, this.countEl, h('span', { class: 'spacer' }), this.menuBtn),
      this.progress, this.hint, this.quick, this.rowsBox, this.summary);
    this.buildSummary();
  }

  menu(anchor) {
    const items = [
      ['Paste a coupon…', () => this.pasteDialog()],
      ['Fill missing odds from the feed', () => this.fillOdds()],
      [this.c.showCrowd ? 'Hide crowd percentages' : 'Show crowd percentages', () => { this.c.showCrowd = !this.c.showCrowd; this.onChange('rows'); }],
      this.c.showCrowd ? ['Paste crowd percentages…', () => this.pasteCrowd()] : null,
      'hr',
      ['Clear the coupon', () => this.c.clear()],
    ].filter(Boolean);
    const m = h('div', { class: 'menu' }, ...items.map((it) => (it === 'hr' ? h('hr') : h('button', { onmousedown: (e) => { e.preventDefault(); m.remove(); it[1](); } }, it[0]))));
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
    this.title.textContent = info.state === 'ok' ? info.title : 'Coupon';
    this.title.className = `slip-title ${info.state === 'ok' ? 'ok' : ''}`;
    this.countEl.textContent = info.state === 'ok' ? `${n} matches` : n ? `${n} match${n > 1 ? 'es' : ''}` : '';
    const pctOf = (k) => `${Math.min(100, (k / g15.n) * 100)}%`;
    this.progress.replaceChildren(h('div', { class: 'ptrack', title: `${g13.label}: ${g13.n} matches · ${g15.label}: ${g15.n} matches` },
      h('div', { class: `pfill s-${info.state}`, style: { width: pctOf(n) } }),
      h('i', { class: `pmark${n >= g13.n ? ' hit' : ''}`, style: { left: pctOf(g13.n) } }),
      h('i', { class: `pmark${n >= g15.n ? ' hit' : ''}`, style: { left: pctOf(g15.n) } })));
    this.hint.textContent = info.state === 'empty' ? `${g13.n} matches play as ${g13.label}, ${g15.n} as ${g15.label}.` : info.hint;
    this.hint.className = `hintline ${info.state === 'over' || info.state === 'between' ? 'warn' : ''}`;
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
      h('b', {}, 'No matches yet'),
      h('div', {}, 'Press + on a fixture, type two teams above, or paste a whole coupon.'));
    api.get('/api/legacy').then((leg) => {
      const bits = Object.entries(leg).filter(([, rows]) => rows.length);
      if (!bits.length) return;
      box.append(h('div', { class: 'small', style: { marginTop: '12px' } }, 'Import your earlier coupon: ',
        ...bits.map(([g, rows]) => h('button', { class: 'link', style: { marginRight: '10px' }, onclick: () => { rows.forEach((r) => this.c.add(r, { silent: true })); toast(`Imported ${rows.length} matches`); } },
          `${g === 'turkish' ? 'Spor Toto' : '13er Wette'} (${rows.length})`))));
    }).catch(() => {});
    return box;
  }

  buildRow(row) {
    const mk = (side) => {
      const f = textField({ value: '', placeholder: side === 'home' ? 'Home team' : 'Away team', cls: 'bare',
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
    const picks = LABELS.map((s) => h('button', { class: 'pk', 'data-s': s, onclick: () => this.c.toggleSymbol(row, s) }, h('b', {}, s), h('span', {}, '–')));
    const remove = h('button', { class: 'iconbtn sm', title: 'Remove from the coupon', 'aria-label': 'Remove from the coupon', onclick: () => this.c.remove(row) }, icon('x', 14));
    const detail = h('div', { class: 'sdetail' });
    const sug = h('div', { class: 'ssug' });
    const crowd = textField({ placeholder: 'Crowd % 1/X/2', cls: 'sm', onCommit: (v) => {
      const t = v.trim().split(/[\s/;,]+/).map(parseNum).filter((x) => x != null);
      this.c.edit(row, { crowd: t.length === 3 && t.reduce((a, b) => a + b, 0) > 0 ? t : null }, { reanalyze: false });
      crowd.set(row.crowd ? row.crowd.map((x) => Math.round(x)).join('/') : '');
    } });
    const idx = h('div', { class: 'sidx num' });
    const info = h('button', { class: 'iconbtn sm', title: 'Open the full breakdown', 'aria-label': 'Open the full breakdown', onclick: () => showDetails(row.home, row.away, this.c, { odds: row.odds }) }, icon('info', 15));
    const el = h('div', { class: 'srow' }, idx,
      h('div', { class: 'steams' }, home.wrap, away.wrap),
      h('div', { class: 'spicks' }, ...picks),
      h('div', { class: 'sside' }, info),
      detail, sug, h('div', { class: 'scrowd' }, crowd.wrap));
    const side = el.querySelector('.sside');
    side.append(remove);
    side.style.flexDirection = 'column';
    return { el, idx, home, away, picks, detail, sug, crowd, editing: false };
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
    v.el.classList.toggle('dup', this.c.rows.some((r) => r !== row && r.complete && r.key === row.key));
    this.paintPicks(row, v);
    const bits = [];
    if (a?.src && !(a.src === 'odds' && row.odds)) bits.push(h('span', { class: `src ${a.src}` }, SRC_TEXT[a.src] || a.src));
    else if (row.busy) bits.push(h('span', { class: 'spin-sm' }));
    bits.push(this.oddsPart(row, v));
    if (a?.ou25 != null) bits.push(h('span', {}, `O2.5 ${Math.round(a.ou25 * 100)}%`));
    if (a?.btts != null) bits.push(h('span', {}, `BTTS ${Math.round(a.btts * 100)}%`));
    if (a?.score) bits.push(h('span', { title: 'Most likely score' }, `Score ${a.score}`));
    v.detail.replaceChildren(...bits);
    v.el.title = a?.league_name || '';
    const chips = [];
    for (const side of ['home', 'away']) {
      const s = sides[side];
      if (!a || !s) continue;
      if (!s.canonical) {
        chips.push(h('span', {}, `“${row[side]}” is not recognised`), s.suggest?.length ? h('span', {}, ' Did you mean:') : h('span', {}, ' Add odds for this match.'),
          ...(s.suggest || []).map((o) => h('button', { class: 'sugchip', title: o.league_name, onclick: () => this.c.confirmName(row, side, o.name) }, o.display)));
      } else if (s.how === 'fuzzy') {
        chips.push(h('span', {}, `“${row[side]}” looks like`), h('button', { class: 'sugchip', onclick: () => this.c.confirmName(row, side, s.canonical) }, `${s.display} · confirm`));
      }
    }
    v.sug.replaceChildren(...chips);
  }

  oddsPart(row, v) {
    if (v.editing) {
      const ins = [0, 1, 2].map((k) => h('input', { value: row.odds ? row.odds[k].toFixed(2) : '', placeholder: LABELS[k], inputmode: 'decimal', 'aria-label': `Odds for ${LABELS[k]}` }));
      const done = () => {
        const vals = ins.map((i) => parseNum(i.value));
        v.editing = false;
        this.c.edit(row, { odds: vals.every((x) => x && x > 1) ? vals : null });
        this.paintRow(row, v);
      };
      ins.forEach((i) => {
        i.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); done(); } if (e.key === 'Escape') { v.editing = false; this.paintRow(row, v); } });
        i.addEventListener('paste', (e) => {
          const t = e.clipboardData.getData('text').trim().split(/[\s/;]+/).map(parseNum);
          if (t.length === 3 && t.every((x) => x && x > 1)) { e.preventDefault(); ins.forEach((o, j) => { o.value = t[j].toFixed(2); }); }
        });
      });
      setTimeout(() => ins[0].focus(), 0);
      return h('span', { class: 'oedit' }, ...ins, h('button', { class: 'btn sm', onclick: done }, 'OK'));
    }
    const edit = () => { v.editing = true; this.paintRow(row, v); };
    return row.odds
      ? h('button', { class: 'linkish num', onclick: edit, title: 'Edit the odds' }, row.odds.map((o) => o.toFixed(2)).join(' / '))
      : h('button', { class: 'linkish', onclick: edit, title: 'Typing the bookmaker odds makes the numbers more accurate' }, 'Add odds');
  }

  paintPicks(row, v) {
    const p = row.probs();
    const label = this.c.labelOf(row);
    v.picks.forEach((b, k) => {
      b.classList.toggle('sel', label.includes(LABELS[k]));
      b.classList.toggle('lockd', !!row.lock && label.includes(LABELS[k]));
      b.lastChild.textContent = row.an?.probs ? Math.round(p[k] * 100) : '–';
      b.title = row.an?.probs ? `${LABELS[k]}: ${pct(p[k], 1)} · fair odds ${fair(p[k])}. Click to play it yourself.` : 'Pricing…';
    });
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
      toast(`Added ${rows.length} match${rows.length > 1 ? 'es' : ''}. The coupon has ${this.c.n}.`, { key: 'add' });
    } catch (e) { toast(e.message, { error: true }); }
  }

  pasteDialog() {
    const ta = h('textarea', { class: 'txt', rows: 12, placeholder: 'Galatasaray - Fenerbahçe\nBayern München - Dortmund  1.50 4.60 6.00\nTurkey - Spain' });
    modal('Paste a coupon', h('div', {},
      h('div', { class: 'small muted', style: { margin: '0 0 8px' } }, 'One match on each line: “Home - Away”, then the three odds if you have them. “v”, “vs” and “:” also work. Full names are fine.'), ta), [
      ['Cancel'],
      ['Add to the coupon', async () => { await this.pasteText(ta.value); }],
      ['Replace the coupon', async () => { this.c.rows = []; await this.pasteText(ta.value); }, 'primary']]);
    setTimeout(() => ta.focus(), 30);
  }

  pasteCrowd() {
    const rows = this.c.complete;
    const ta = h('textarea', { class: 'txt', rows: Math.min(Math.max(rows.length, 6), 16), placeholder: '68/21/11\n45/30/25' });
    modal('Paste crowd percentages', h('div', {},
      h('div', { class: 'small muted', style: { marginBottom: '8px' } }, `One line for each match, in coupon order (${rows.length} matches): the share of players on 1, X and 2. Write “68/21/11”, “68 21 11” or “68;21;11”.`), ta), [
      ['Cancel'],
      ['Apply', () => {
        const lines = ta.value.split(/\n/).map((l) => l.trim().split(/[\s/;,]+/).map(parseNum).filter((x) => x != null)).filter((t) => t.length >= 3);
        if (!lines.length) { toast('No “1/X/2” lines found'); return false; }
        rows.forEach((r, i) => { if (lines[i]) r.crowd = lines[i].slice(-3); });
        this.c.save(); this.c.emit('row'); this.c.retick();
        toast(`Crowd percentages set for ${Math.min(lines.length, rows.length)} of ${rows.length} matches` + (lines.length !== rows.length ? `. You gave ${lines.length} lines: check the order.` : ''));
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
      toast(n ? `Found odds for ${n} of ${targets.length}.` + (n < targets.length ? ' The feed covers about a week ahead.' : '')
        : 'No odds yet. The feed covers about a week ahead.');
    } catch (e) { toast(`The odds feed is not available: ${e.message}`, { error: true }); }
  }

  // ------------------------------------------------------------------ summary
  buildSummary() {
    this.pBig = h('div', { class: 'psum' });
    this.budgetIn = h('input', { class: 'txt num sm', type: 'number', min: 1, max: 100000, value: this.c.budget, 'aria-label': 'Budget in columns' });
    this.budgetIn.addEventListener('input', () => { const b = parseInt(this.budgetIn.value, 10); if (b >= 1) this.c.setBudget(b); });
    this.presets = h('div', { class: 'presets' }, ...PRESETS.map((p) => h('button', { onclick: () => { this.budgetIn.value = p; this.c.setBudget(p); } }, String(p))));
    this.price = h('input', { class: 'txt num sm', type: 'number', min: 0, step: '0.01', placeholder: 'Price per column', 'aria-label': 'Price per column', style: { width: '130px', marginLeft: 'auto' }, value: localStorage.getItem('fp.price') || '' });
    this.price.addEventListener('input', () => { localStorage.setItem('fp.price', this.price.value); this.paintSummary(); });
    this.tiers = h('div', { class: 'tiers' });
    this.notes = h('div', { class: 'notes' });
    this.moreBox = h('div', { class: 'more', hidden: true });
    this.moreBtn = h('button', { class: 'btn sm quiet', onclick: () => this.toggleMore() }, 'More options');
    const acts = h('div', { class: 'actions' },
      h('button', { class: 'btn sm', onclick: () => this.copy() }, icon('copy', 14), 'Copy'),
      h('button', { class: 'btn sm', onclick: () => this.print() }, icon('print', 14), 'Print'),
      this.moreBtn, h('span', { class: 'spacer' }),
      h('button', { class: 'btn sm primary', onclick: () => this.saveHistory() }, icon('save', 14), 'Save'));
    this.summary.replaceChildren(this.pBig,
      h('div', { class: 'budget' }, h('span', { class: 'label' }, 'Budget'), this.budgetIn, h('span', { class: 'muted small' }, 'columns'), this.price),
      this.presets, this.tiers, this.notes, acts, this.moreBox);
  }

  paintSummary() {
    const c = this.c; const r = c.result; const n = c.n;
    if (document.activeElement !== this.budgetIn) this.budgetIn.value = c.budget;
    [...this.presets.children].forEach((b) => b.classList.toggle('on', Number(b.textContent) === c.budget));
    this.summary.classList.toggle('busy', !!c.thinking);
    const info = c.info;
    if (!r || !n) {
      this.pBig.replaceChildren(h('div', { class: 'label' }, 'Chance of a prize'), h('div', { class: 'pbig dash' }, '–'));
      this.tiers.replaceChildren(); this.notes.replaceChildren();
      return;
    }
    const every = r.p_prize > 0 ? Math.round(1 / r.p_prize) : null;
    const price = parseFloat(this.price.value || '');
    this.pBig.replaceChildren(
      h('div', { class: 'label' }, `Chance of ${r.threshold} or more correct out of ${n}`, info.state === 'ok' ? '' : h('span', { class: 'warn' }, '  · preview')),
      h('div', { class: 'pbig num' }, pct(r.p_prize, r.p_prize < 0.1 ? 2 : 1)),
      h('div', { class: 'kv num' }, every && every > 1 ? h('span', {}, 'about ', h('b', {}, `1 in ${every}`)) : '',
        h('span', {}, h('b', {}, String(r.columns)), ` column${r.columns > 1 ? 's' : ''}`),
        price > 0 ? h('span', {}, 'cost ', h('b', {}, (r.columns * price).toFixed(2))) : ''));
    this.tiers.replaceChildren(...Object.entries(r.tiers).slice(0, 4).map(([k, p]) => h('div', { class: 'tier', title: `Chance of getting ${k}${Number(k) === n ? '' : ' or more'} correct` },
      h('div', { class: 't' }, Number(k) === n ? `All ${k}` : `${k}+`), h('div', { class: 'v num' }, pct(p, p < 0.01 ? 2 : 1)))));
    const notes = [];
    const unk = c.complete.filter((x) => x.an && x.an.src === 'none').length;
    if (unk) notes.push(h('div', { class: 'box warn' }, `${unk} match${unk > 1 ? 'es are' : ' is'} counted as one third each. Correct the name or add odds.`));
    if (r.over_budget) notes.push(h('div', { class: 'box warn' }, `Your own picks need ${r.columns} columns. That is more than the budget of ${r.budget}.`));
    this.notes.replaceChildren(...notes);
    if (this.more) this.paintMore();
  }

  toggleMore() { this.more = !this.more; this.moreBox.hidden = !this.more; this.moreBtn.textContent = this.more ? 'Fewer options' : 'More options'; if (this.more) this.paintMore(); }

  paintMore() {
    const c = this.c;
    const hasCrowd = c.complete.length && c.complete.every((r) => r.crowd);
    const obj = h('div', { class: 'seg' },
      h('button', { class: c.objective === 'hit' ? 'on' : '', onclick: () => c.setObjective('hit'), title: 'Maximise the chance of a prize' }, 'Most likely prize'),
      h('button', { class: c.objective === 'crowd' ? 'on' : '', disabled: !hasCrowd, onclick: () => c.setObjective('crowd'), title: 'Lean toward results the crowd under-plays. Needs crowd percentages on every match.' }, 'Against the crowd'));
    const tilt = h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: c.tilt, style: { width: '100%' }, 'aria-label': 'How much to lean on the crowd' });
    tilt.addEventListener('input', () => c.setTilt(parseFloat(tilt.value)));
    const single = c.result ? `A single column has ${pct(c.result.single.p_prize, 2)}. This system has ${c.result.single.p_prize > 0 ? `${(c.result.p_prize / c.result.single.p_prize).toFixed(1)} times` : 'more'} that. It uses ${c.result.n_double} doubles and ${c.result.n_triple} triples; you expect ${c.result.expected.toFixed(1)} correct.` : '';
    this.moreBox.replaceChildren(
      h('div', { class: 'row' }, h('span', { class: 'label' }, 'Optimise for'), obj),
      c.objective === 'crowd' ? h('div', { class: 'small' }, h('div', { class: 'muted' }, 'Lean on the crowd. Above about 0.75 the chance of a prize falls fast.'), tilt) : '',
      h('div', { class: 'row' },
        h('button', { class: 'btn sm', onclick: () => this.showLadder() }, 'What does a bigger budget buy?'),
        hasCrowd ? h('button', { class: 'btn sm', onclick: () => this.showSweep() }, 'Prize chance and payout') : ''),
      single ? h('div', { class: 'small muted' }, single) : '',
      h('div', { class: 'small muted' }, 'Click 1, X or 2 on a match to play it yourself. The rest of the ticket is optimised around your picks.'));
  }

  async showLadder() {
    const { body } = this.c.ticketBody();
    const out = h('div', {}, h('span', { class: 'spin-sm' }), ' Calculating…');
    modal('What does a bigger budget buy?', out, [['Close']], { wide: true });
    try {
      const { ladder } = await api.post('/api/ladder', body);
      const base = ladder[0]?.p_prize || 0;
      out.replaceChildren(h('table', { class: 'simple' },
        h('tr', {}, h('th', {}, 'Columns'), h('th', { class: 'r' }, 'Chance of a prize'), h('th', { class: 'r' }, 'Times a single column'), h('th', { class: 'r' }, 'Doubles · triples'), h('th', {})),
        ...ladder.map((x) => h('tr', {}, h('td', {}, String(x.columns)), h('td', { class: 'r' }, pct(x.p_prize, 2)),
          h('td', { class: 'r' }, base > 0 ? (x.p_prize / base).toFixed(1) : '–'), h('td', { class: 'r' }, `${x.doubles} · ${x.triples}`),
          h('td', { class: 'r' }, h('button', { class: 'btn sm quiet', onclick: () => this.c.setBudget(x.columns) }, 'Use'))))),
      h('div', { class: 'note' }, 'The cost grows with the number of columns, but the chance of a prize grows much more slowly. Pick a budget you accept to lose. This table ignores your own picks and the crowd.'));
    } catch (e) { out.textContent = `Failed: ${e.message}`; }
  }

  async showSweep() {
    const { body } = this.c.ticketBody();
    const out = h('div', {}, h('span', { class: 'spin-sm' }), ' Simulating. This takes a few seconds…');
    modal('Prize chance and crowd payout', out, [['Close']], { wide: true });
    try {
      const { sweep } = await api.post('/api/sweep', body);
      out.replaceChildren(h('table', { class: 'simple' },
        h('tr', {}, h('th', {}, 'Lean on crowd'), h('th', { class: 'r' }, 'Chance of a prize'), h('th', { class: 'r' }, 'Payout score'), h('th', { class: 'r' }, 'Wins about every')),
        ...sweep.map((x) => h('tr', {}, h('td', {}, x.tilt.toFixed(2)), h('td', { class: 'r' }, pct(x.p_hit, 2)), h('td', { class: 'r' }, `${x.payout.toFixed(2)}×`), h('td', { class: 'r' }, x.p_hit > 0 ? `${Math.round(1 / x.p_hit)} coupons` : '–')))),
      h('div', { class: 'note' }, 'The payout score is a comparison between tickets, not money. Choose the largest lean whose chance of a prize you accept.'));
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
    const txt = [`${info.state === 'ok' ? info.title : `${lines.length} matches`} · ${r?.columns || 1} columns · chance of a prize ${pct(r?.p_prize, 2)}`,
      ...lines.map((l) => `${String(l.i).padStart(2)}  ${`${l.home} - ${l.away}`.padEnd(w)}  ${l.play}`)].join('\n');
    navigator.clipboard.writeText(txt).then(() => toast('Ticket copied'), () => toast('Copy failed', { error: true }));
  }

  print() {
    const info = this.c.info; const r = this.c.result;
    const lines = this.lines();
    const area = document.getElementById('print-area');
    area.replaceChildren(
      h('h2', {}, `${info.state === 'ok' ? info.title : `${lines.length} matches`}: ${r?.columns || 1} columns`),
      h('p', {}, `Chance of a prize ${pct(r?.p_prize, 2)} · printed ${new Date().toLocaleString()}`),
      h('table', {}, h('tr', {}, h('th', {}, '#'), h('th', {}, 'Match'), h('th', {}, 'Play')),
        ...lines.map((l) => h('tr', {}, h('td', {}, String(l.i)), h('td', {}, `${l.home} - ${l.away}`),
          h('td', {}, ...LABELS.map((s) => h('span', { class: `box${l.play.includes(s) ? ' x' : ''}` }, s)))))));
    window.print();
  }

  saveHistory() {
    const r = this.c.result;
    if (!r) { toast('Nothing to save yet'); return; }
    const note = h('input', { class: 'txt', placeholder: 'Note, for example “week 41”' });
    const lines = this.lines();
    const save = async () => {
      try {
        await api.post('/api/history', {
          note: note.value, budget: this.c.budget, columns: r.columns, p_prize: r.p_prize, kind: 'ticket',
          matches: lines.map((l) => ({ home: l.rawHome, away: l.rawAway, probs: l.probs, pick: l.play, src: l.src })),
        });
        toast('Saved. History grades it when the matches are played.');
      } catch (e) { toast(`Save failed: ${e.message}`, { error: true }); return false; }
      return true;
    };
    note.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); note.closest('.modal')?.querySelector('.btn.primary')?.click(); } });
    modal('Save this ticket', h('div', {},
      h('p', { class: 'small muted', style: { marginBottom: '10px' } }, `${lines.length} matches · ${r.columns} columns · chance of a prize ${pct(r.p_prize, 2)}. History compares it with the real results after the matches.`), note),
    [['Cancel'], ['Save', save, 'primary']]);
    setTimeout(() => note.focus(), 30);
  }
}
