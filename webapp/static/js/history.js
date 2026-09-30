// History: saved tickets graded against real results.
import { h, pct, modal, toast } from './ui.js';
import { api, store } from './api.js';

export class HistoryPage {
  constructor(root) { this.root = root; this.load(); }
  destroy() {}

  async load() {
    this.root.replaceChildren(h('div', { class: 'page-head' }, h('h1', {}, 'History')), h('div', { class: 'empty' }, h('span', { class: 'spin-sm' }), ' Grading against results…'));
    let d;
    try { d = await api.get('/api/history'); } catch (e) { this.root.append(h('div', { class: 'empty' }, `Failed: ${e.message}`)); return; }
    const s = d.summary;
    const tiles = h('div', { class: 'tiles' },
      h('div', { class: 'tile' }, h('div', { class: 't' }, 'Saved tickets'), h('div', { class: 'v' }, String(d.entries.length))),
      h('div', { class: 'tile' }, h('div', { class: 't' }, 'Completed'), h('div', { class: 'v' }, String(s.completed))),
      h('div', { class: 'tile' }, h('div', { class: 't' }, 'Reached a prize'), h('div', { class: 'v' }, `${s.cleared}/${s.completed}`)),
      h('div', { class: 'tile' }, h('div', { class: 't' }, 'Correct vs expected'), h('div', { class: 'v' }, s.completed ? `${s.correct} / ${s.expected.toFixed(1)}` : '—')));
    const body = d.entries.length ? d.entries.map((e) => this.card(e)) : [h('div', { class: 'card empty' },
      'No saved tickets yet. Build one on 🎟️ Toto and hit “Save to history” — once the matches are played it grades itself, so you can see your real hit rate against what the model promised.')];
    this.root.replaceChildren(
      h('div', { class: 'page-head' }, h('h1', {}, 'History'), h('span', { class: 'small muted' }, '“Expected” is the honest benchmark: landing around it means the probabilities are calibrated and the rest is variance.')),
      tiles, ...body);
  }

  card(e) {
    const info = store.status.games[e.game] || { flag: '', label: e.game };
    const pending = e.graded < e.n;
    const won = e.threshold != null && e.correct >= e.threshold;
    const prize = pending ? '⏳' : won ? '✅' : '—';
    const del = h('button', { class: 'iconbtn', title: 'Delete', onclick: (ev) => { ev.preventDefault(); this.remove(e); } }, '🗑');
    const rows = h('table', { class: 'simple' },
      h('tr', {}, h('th', {}, '#'), h('th', {}, 'Match'), h('th', {}, 'Played'), h('th', {}, 'Result'), h('th', { class: 'r' }, '1 · X · 2')),
      ...e.rows.map((r, i) => h('tr', {},
        h('td', {}, String(i + 1)), h('td', {}, `${r.home} – ${r.away}`), h('td', {}, h('b', {}, r.pick || '')),
        h('td', {}, r.actual ? h('span', { class: r.ok ? 'okmark' : 'badmark' }, `${r.actual} ${r.ok ? '✓' : '✗'}`) : h('span', { class: 'faint' }, 'pending')),
        h('td', { class: 'r' }, r.p.map((x) => (x == null ? '—' : Math.round(x * 100))).join(' · ')))));
    return h('details', { class: 'card hcard' },
      h('summary', {},
        h('span', {}, (e.saved_at || '').replace('T', ' ').slice(0, 16)),
        h('span', {}, `${info.flag} ${info.label}`),
        h('span', { class: 'muted small' }, e.note || e.kind || ''),
        h('span', {}, `${e.columns} col · said ${pct(e.p_threshold, 1)}`),
        h('span', {}, `${e.correct}/${e.graded}${pending ? ` of ${e.n}` : ''} · exp ${e.expected.toFixed(1)}`),
        h('span', {}, prize), del),
      h('div', { class: 'card-pad' }, rows));
  }

  remove(e) {
    modal('Delete this saved ticket?', h('p', { class: 'small muted' }, `${e.saved_at} · ${e.n} matches. This cannot be undone.`), [
      ['Cancel'],
      ['Delete', async () => { await api.del(`/api/history/${encodeURIComponent(e.id)}`); toast('Deleted'); this.load(); }, 'danger']]);
  }
}
