// Small DOM toolkit: no framework, so typing never re-renders an input.

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else if (k in el && typeof v !== 'string') el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);

export function debounce(fn, ms) {
  let t;
  const d = (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  d.flush = (...args) => { clearTimeout(t); fn(...args); };
  d.cancel = () => clearTimeout(t);
  return d;
}

export const pct = (p, d = 0) => (p == null ? '—' : `${(p * 100).toFixed(d)}%`);
export const fair = (p) => (p > 0 ? (1 / p).toFixed(2) : '—');
export const pct3 = (p) => (p ? p.map((x) => Math.round(x * 100)).join('·') : '—');

// Same folding as scripts/names.norm on the server: accents, Turkish dotless i,
// apostrophes, punctuation. The server stores `keys` already folded this way.
const TRANSLIT = { 'ı': 'i', 'İ': 'i', 'ß': 'ss', 'ø': 'o', 'Ø': 'o', 'æ': 'ae', 'Æ': 'ae', 'œ': 'oe', 'đ': 'd', 'ð': 'd', 'þ': 'th', 'ł': 'l', 'Ł': 'l' };
export function norm(s) {
  let t = String(s ?? '').replace(/[ıİßøØæÆœđðþłŁ]/g, (c) => TRANSLIT[c]);
  t = t.normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase();
  t = t.replace(/['’`´]/g, '').replace(/&/g, ' and ').replace(/[^a-z0-9]+/g, ' ');
  return t.trim().replace(/\s+/g, ' ');
}
export const fold = norm;

// ------------------------------------------------------------ kickoff times
// football-data publishes kickoff in UK time. Show it in the viewer's own.
function londonOffsetMinutes(utcMs) {
  const f = new Intl.DateTimeFormat('en-GB', { timeZone: 'Europe/London', hour12: false, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
  const p = Object.fromEntries(f.formatToParts(new Date(utcMs)).map((x) => [x.type, x.value]));
  return (Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour % 24, +p.minute) - utcMs) / 60000;
}
export function kickoff(dateStr, timeStr) {
  if (!dateStr) return null;
  const [y, m, d] = dateStr.split('-').map(Number);
  if (!timeStr) return { date: new Date(y, m - 1, d, 12), hasTime: false };
  const [hh, mm] = timeStr.split(':').map(Number);
  const guess = Date.UTC(y, m - 1, d, hh, mm);
  return { date: new Date(guess - londonOffsetMinutes(guess) * 60000), hasTime: true };
}
export const ymd = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
export const hhmm = (d) => d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
export function dayLabel(d) {
  const today = new Date(); today.setHours(12, 0, 0, 0);
  const x = new Date(d); x.setHours(12, 0, 0, 0);
  const diff = Math.round((x - today) / 86400000);
  const base = x.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' });
  return diff === 0 ? `Today · ${base}` : diff === 1 ? `Tomorrow · ${base}` : base;
}

// ------------------------------------------------------------ probability bits
export function probBar(p, { thin = false, pending = false } = {}) {
  const bar = h('div', { class: `bar${thin ? ' thin' : ''}${pending ? ' pending' : ''}` });
  if (p) setBar(bar, p);
  return bar;
}
export function setBar(bar, p, pending = false) {
  bar.classList.toggle('pending', !!pending);
  if (!p) { bar.replaceChildren(); return; }
  const lbl = ['1', 'X', '2'];
  bar.replaceChildren(...p.map((x, i) => h('span', {
    class: ['s1', 'sx', 's2'][i], style: { width: `${(x * 100).toFixed(2)}%` },
    title: `${lbl[i]}: ${pct(x, 1)} (fair odds ${fair(x)})`,
  }, x >= 0.12 ? `${Math.round(x * 100)}` : '')));
}
export function legend() {
  return h('div', { class: 'legend' },
    h('span', {}, h('i', { style: { background: 'var(--c1)' } }), '1 home'),
    h('span', {}, h('i', { style: { background: 'var(--cx)' } }), 'X draw'),
    h('span', {}, h('i', { style: { background: 'var(--c2)' } }), '2 away'));
}

// Three outcome tiles "1 49 | X 25 | 2 27"; the likeliest is emphasised.
export function outcomeTiles(p, { src } = {}) {
  const top = p ? p.indexOf(Math.max(...p)) : -1;
  return h('div', { class: `tiles3${src ? ` src-${src}` : ''}` },
    ...['1', 'X', '2'].map((s, i) => h('div', {
      class: `t3 t3-${s}${i === top ? ' top' : ''}${p ? '' : ' empty'}`,
      title: p ? `${s}: ${pct(p[i], 1)} · fair odds ${fair(p[i])}` : '',
    }, h('b', {}, s), h('span', {}, p ? Math.round(p[i] * 100) : '·'))));
}

// source badges
const SRC = {
  odds: ['odds', 'Bookmaker odds — the best information there is'],
  v2: ['model', 'Model v2 (no bookmaker odds for this match)'],
  v1: ['model v1', 'Model v1 (v2 had no data for these teams)'],
  euro: ['club elo', 'Cross-league clubs priced from global club ratings'],
  intl: ['nations', 'National-team model'],
  none: ['no data', 'Not recognised — counted as ⅓ each until you fix the name or add odds'],
};
export function badge(src) {
  if (!src) return h('span');
  const [label, tip] = SRC[src] || [src, ''];
  return h('span', { class: `badge ${src}`, title: tip }, label);
}

// ------------------------------------------------------------ toasts / modal
export function toast(msg, { action, onAction, error = false, ms = 3800, key } = {}) {
  const box = document.getElementById('toasts');
  if (key) box.querySelectorAll(`[data-key="${key}"]`).forEach((e) => e.remove());   // newest replaces older
  while (box.children.length >= 3) box.firstChild.remove();
  const t = h('div', { class: `toast${error ? ' err' : ''}`, 'data-key': key || '' }, msg);
  if (action) {
    t.append(h('button', { onclick: () => { onAction?.(); t.remove(); } }, action));
    ms = Math.max(ms, 7000);
  }
  box.append(t);
  setTimeout(() => t.remove(), ms);
}

export function modal(title, body, buttons = [], { wide = false } = {}) {
  const back = h('div', { class: 'modal-back' });
  const close = () => { back.remove(); document.removeEventListener('keydown', esc); };
  function esc(e) { if (e.key === 'Escape') close(); }
  const head = h('div', { class: 'modal-head' }, h('h3', {}, title), h('button', { class: 'iconbtn', title: 'Close', onclick: close }, '✕'));
  const m = h('div', { class: `modal${wide ? ' wide' : ''}`, role: 'dialog', 'aria-modal': 'true' }, head, body,
    buttons.length ? h('div', { class: 'row end', style: { marginTop: '14px' } },
      ...buttons.map(([label, fn, cls]) => h('button', {
        class: `btn ${cls || ''}`, onclick: async () => { if ((await fn?.()) !== false) close(); },
      }, label))) : '');
  back.append(m);
  back.addEventListener('mousedown', (e) => { if (e.target === back) close(); });
  document.addEventListener('keydown', esc);
  document.body.append(back);
  return { close, el: m };
}

// ------------------------------------------------------------ text field
// A text input that is quick to correct: a clear (×) button, and when focus
// arrives by Tab/Enter the whole text is selected so typing replaces it.
export function textField({ value = '', placeholder = '', cls = '', onInput, onCommit, inputmode } = {}) {
  const input = h('input', { class: `txt ${cls}`.trim(), value, placeholder, spellcheck: false, inputmode, autocomplete: 'off' });
  const clear = h('button', {
    class: 'clearbtn', tabindex: -1, title: 'Clear', type: 'button',
    onmousedown: (e) => {
      e.preventDefault();
      input.value = ''; input.focus(); sync();
      onInput?.(''); onCommit?.('');
    },
  }, '×');
  const wrap = h('div', { class: 'field' }, input, clear);
  const sync = () => wrap.classList.toggle('has', input.value.length > 0);
  let byMouse = false;
  input.addEventListener('mousedown', () => { byMouse = true; setTimeout(() => { byMouse = false; }, 250); });
  input.addEventListener('focus', () => { if (!byMouse) input.select(); });
  input.addEventListener('input', () => { sync(); onInput?.(input.value); });
  input.addEventListener('change', () => onCommit?.(input.value));
  sync();
  return { wrap, input, sync, set(v) { input.value = v; sync(); } };
}

// ------------------------------------------------------------ autocomplete
// Matches the typed text against every name a team goes by (full name, data
// name, nicknames) — accent- and punctuation-insensitive. Free text is always
// allowed: an unknown name is flagged, not erased. Enter/Tab picks the
// highlighted item, ↑↓ moves, Esc closes.
export class Autocomplete {
  constructor(input, { source, onPick, filter, limit = 8 } = {}) {
    this.input = input; this.source = source; this.onPick = onPick; this.filter = filter; this.limit = limit;
    this.list = null; this.items = []; this.hi = 0;
    const host = input.closest('.field') || input.parentElement;
    host.classList.add('ac');
    this.host = host;
    input.addEventListener('input', () => this.open());
    input.addEventListener('focus', () => { if (input.value) this.open(); });
    input.addEventListener('blur', () => setTimeout(() => this.close(), 120));
    input.addEventListener('keydown', (e) => this.key(e));
  }
  rank(t, q, words) {
    let best = 9;
    for (const k of t.keys) {
      if (k.startsWith(q)) { best = Math.min(best, 0); break; }
      if (k.includes(` ${q}`)) best = Math.min(best, 1);
      else if (k.includes(q)) best = Math.min(best, 2);
      else if (words.length > 1 && words.every((w) => k.includes(w))) best = Math.min(best, 3);
    }
    return best;
  }
  matches(text) {
    const q = norm(text);
    if (!q) return [];
    const words = q.split(' ');
    const out = [];
    for (const t of this.source()) {
      if (this.filter && !this.filter(t)) continue;
      const r = this.rank(t, q, words);
      if (r < 9) out.push([r, t]);
    }
    out.sort((a, b) => a[0] - b[0] || a[1].display.length - b[1].display.length);
    return out.slice(0, this.limit).map((x) => x[1]);
  }
  open() {
    this.items = this.matches(this.input.value);
    if (!this.items.length) { this.close(); return; }
    this.hi = 0;
    if (!this.list) { this.list = h('div', { class: 'ac-list' }); this.host.append(this.list); }
    this.render();
  }
  render() {
    const q = norm(this.input.value);
    this.list.replaceChildren(...this.items.map((t, i) => {
      const d = norm(t.display); const at = d.indexOf(q);
      // highlight only when the display name itself (not an alias) contains the text
      const label = (at >= 0 && d.length === t.display.length)
        ? [t.display.slice(0, at), h('mark', {}, t.display.slice(at, at + q.length)), t.display.slice(at + q.length)]
        : [t.display];
      const alias = at < 0 ? t.keys.find((k) => k.includes(q)) : null;
      return h('div', {
        class: `ac-item${i === this.hi ? ' hi' : ''}`,
        onmousedown: (e) => { e.preventDefault(); this.pick(t); },
        onmousemove: () => { if (this.hi !== i) { this.hi = i; this.render(); } },
      }, h('span', { class: 'nm' }, ...label, alias ? h('i', { class: 'alias' }, ` · ${alias}`) : ''),
      h('span', { class: 'lg' }, t.league_name || ''));
    }));
  }
  key(e) {
    if (!this.list) return;
    if (e.key === 'ArrowDown') { this.hi = Math.min(this.hi + 1, this.items.length - 1); this.render(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { this.hi = Math.max(this.hi - 1, 0); this.render(); e.preventDefault(); }
    else if ((e.key === 'Enter' || e.key === 'Tab') && this.items[this.hi]) {
      this.pick(this.items[this.hi]);
      if (e.key === 'Enter') e.preventDefault();
    } else if (e.key === 'Escape') { this.close(); e.stopPropagation(); }
  }
  pick(t) {
    this.input.value = t.display;
    this.close();
    this.input.dispatchEvent(new Event('input', { bubbles: true }));
    this.onPick?.(t);
  }
  close() { this.list?.remove(); this.list = null; }
}
