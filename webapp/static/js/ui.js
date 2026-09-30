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
  return d;
}

export const pct = (p, d = 0) => (p == null ? '—' : `${(p * 100).toFixed(d)}%`);
export const fold = (s) => String(s || '').normalize('NFKD').replace(/[̀-ͯ]/g, '')
  .replace(/ı/g, 'i').toLowerCase().trim();
export const fair = (p) => (p > 0 ? (1 / p).toFixed(2) : '—');
export const fmtDate = (s) => {
  const d = new Date(s + (String(s).length === 10 ? 'T12:00:00' : ''));
  return d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' });
};

// ------------------------------------------------------------ probability bar
export function probBar(p, { thin = false, pending = false, title } = {}) {
  const bar = h('div', { class: `bar${thin ? ' thin' : ''}${pending ? ' pending' : ''}`, title: title || '' });
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

// ------------------------------------------------------------ source badges
const SRC = {
  odds: ['odds', 'Bookmaker odds (typed on the coupon) — the best information there is'],
  feed: ['feed odds', 'Bookmaker odds found in the live feed'],
  v2: ['v2', 'Club model v2'], v1: ['v1', 'Club model v1 (production)'],
  euro: ['euro', 'Cross-league clubs via global club Elo (UCL/UEL)'],
  intl: ['nations', 'National-team model'],
  none: ['no data', 'Not recognised — 1/3 each until you fix the name or add odds'],
};
export function badge(src) {
  const [label, tip] = SRC[src] || [src, ''];
  return h('span', { class: `badge ${src}`, title: tip }, label);
}

// ------------------------------------------------------------ toasts / modal
export function toast(msg, { action, onAction, error = false, ms = 3800 } = {}) {
  const box = document.getElementById('toasts');
  const t = h('div', { class: `toast${error ? ' err' : ''}` }, msg);
  if (action) {
    t.append(h('button', { onclick: () => { onAction?.(); t.remove(); } }, action));
    ms = Math.max(ms, 7000);
  }
  box.append(t);
  setTimeout(() => t.remove(), ms);
}

export function modal(title, body, buttons = []) {
  const back = h('div', { class: 'modal-back' });
  const close = () => back.remove();
  const m = h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true' },
    h('h3', {}, title), body,
    h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '14px' } },
      ...buttons.map(([label, fn, cls]) => h('button', {
        class: `btn ${cls || ''}`, onclick: async () => { if ((await fn?.()) !== false) close(); },
      }, label))));
  back.append(m);
  back.addEventListener('mousedown', (e) => { if (e.target === back) close(); });
  document.addEventListener('keydown', function esc(e) {
    if (e.key === 'Escape') { close(); document.removeEventListener('keydown', esc); }
  });
  document.body.append(back);
  return { close, el: m };
}

// ------------------------------------------------------------ autocomplete
// Free text is always allowed (a name we don't know is flagged, not erased);
// the list only helps. Enter/Tab picks the highlighted item, Esc closes.
export class Autocomplete {
  constructor(input, { source, onPick, filter } = {}) {
    this.input = input; this.source = source; this.onPick = onPick; this.filter = filter;
    this.list = null; this.items = []; this.hi = 0;
    const wrap = input.parentElement;
    wrap.classList.add('ac');
    input.setAttribute('autocomplete', 'off');
    input.addEventListener('input', () => this.open());
    input.addEventListener('focus', () => { if (input.value) this.open(); });
    input.addEventListener('blur', () => setTimeout(() => this.close(), 120));
    input.addEventListener('keydown', (e) => this.key(e));
  }
  matches(q) {
    const f = fold(q);
    if (!f) return [];
    const out = [];
    for (const t of this.source()) {
      if (this.filter && !this.filter(t)) continue;
      const i = t._f.indexOf(f);
      if (i < 0) continue;
      const score = (i === 0 ? 0 : t._f[i - 1] === ' ' ? 1 : 2) + t._f.length / 100;
      out.push([score, t]);
    }
    out.sort((a, b) => a[0] - b[0]);
    return out.slice(0, 8).map((x) => x[1]);
  }
  open() {
    this.items = this.matches(this.input.value);
    if (!this.items.length || (this.items.length === 1 && this.items[0].name === this.input.value)) {
      this.close(); return;
    }
    this.hi = 0;
    if (!this.list) {
      this.list = h('div', { class: 'ac-list' });
      this.input.parentElement.append(this.list);
    }
    this.render();
  }
  render() {
    const f = fold(this.input.value);
    this.list.replaceChildren(...this.items.map((t, i) => {
      const at = t._f.indexOf(f);
      const name = at >= 0
        ? [t.name.slice(0, at), h('mark', {}, t.name.slice(at, at + f.length)), t.name.slice(at + f.length)]
        : [t.name];
      return h('div', {
        class: `ac-item${i === this.hi ? ' hi' : ''}`,
        onmousedown: (e) => { e.preventDefault(); this.pick(t); },
      }, h('span', {}, ...name), h('span', { class: 'lg' }, t.league_name || ''));
    }));
  }
  key(e) {
    if (!this.list) return;
    if (e.key === 'ArrowDown') { this.hi = Math.min(this.hi + 1, this.items.length - 1); this.render(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { this.hi = Math.max(this.hi - 1, 0); this.render(); e.preventDefault(); }
    else if ((e.key === 'Enter' || e.key === 'Tab') && this.items[this.hi]) {
      if (fold(this.input.value) !== this.items[this.hi]._f) this.pick(this.items[this.hi]);
      else this.close();
      if (e.key === 'Enter') e.preventDefault();
    } else if (e.key === 'Escape') this.close();
  }
  pick(t) {
    this.input.value = t.name;
    this.close();
    this.onPick?.(t);
    this.input.dispatchEvent(new Event('change', { bubbles: true }));
  }
  close() { this.list?.remove(); this.list = null; }
}
