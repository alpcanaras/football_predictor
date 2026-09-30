// App shell: boot (wait for the engine), router, model switch, data menu.
import { h, $, toast, fold } from './ui.js';
import { api, store } from './api.js';
import { TotoPage } from './toto.js';
import { FixturesPage } from './fixtures.js';
import { MatchPage } from './match.js';
import { IntlPage } from './intl.js';
import { HistoryPage } from './history.js';

const PAGES = { toto: TotoPage, fixtures: FixturesPage, match: MatchPage, internationals: IntlPage, history: HistoryPage };
let current = null;

async function boot() {
  for (;;) {
    try {
      const s = await api.get('/api/status');
      store.status = s;
      if (s.state === 'ready') break;
      if (s.state === 'error') { $('#boot-step').textContent = `Could not start: ${s.error}`; return; }
      $('#boot-step').textContent = `${s.step || 'Starting'}…`;
    } catch { $('#boot-step').textContent = 'Waiting for the server…'; }
    await new Promise((r) => setTimeout(r, 700));
  }
  const { teams } = await api.get('/api/teams');
  store.teams = teams.map((t) => ({ ...t, _f: fold(t.name) }));
  paintHeader();
  window.addEventListener('hashchange', route);
  route();
}

function route() {
  const name = (location.hash.replace(/^#\/?/, '') || 'toto').split('?')[0];
  const Page = PAGES[name] || TotoPage;
  document.querySelectorAll('#tabs a').forEach((a) => a.classList.toggle('active', a.dataset.page === (PAGES[name] ? name : 'toto')));
  current?.destroy?.();
  const root = $('#app');
  root.replaceChildren();
  current = new Page(root);
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ header
function paintHeader() {
  const s = store.status;
  document.querySelectorAll('#model-seg button').forEach((b) => {
    b.classList.toggle('on', b.dataset.mode === s.model_mode);
    b.disabled = b.dataset.mode !== 'v1' && !s.v2?.ok;
    b.onclick = () => setMode(b.dataset.mode);
  });
  const pill = $('#data-pill');
  const stale = (s.stale || []).length;
  pill.textContent = s.job?.running ? '⟳ Updating…' : `📡 ${s.data_through || '—'}${stale ? ` · ⚠ ${stale}` : ''}`;
  pill.classList.toggle('warn', stale > 0);
  pill.onclick = togglePop;
}

async function setMode(mode) {
  if (mode === store.status.model_mode) return;
  try {
    store.status = await api.post('/api/settings', { model_mode: mode });
    paintHeader();
    store.emit('mode');
    const msg = { auto: 'Auto — v2 for the Süper Lig, v1 elsewhere', v1: 'v1 everywhere', v2: 'v2 everywhere' }[mode];
    toast(`Model: ${msg}. Matches with odds still use the odds.`);
  } catch (e) { toast(e.message, { error: true }); }
}

function togglePop() {
  const pop = $('#data-pop');
  if (!pop.hidden) { pop.hidden = true; return; }
  paintPop(); pop.hidden = false;
  const off = (e) => { if (!pop.contains(e.target) && e.target !== $('#data-pill')) { pop.hidden = true; document.removeEventListener('mousedown', off); } };
  setTimeout(() => document.addEventListener('mousedown', off), 0);
}

function paintPop() {
  const s = store.status; const pop = $('#data-pop');
  const compare = h('input', { type: 'checkbox', checked: store.compare });
  compare.addEventListener('change', () => { store.compare = compare.checked; localStorage.setItem('fp.compare', compare.checked ? '1' : '0'); store.emit('compare'); });
  const log = h('pre', { hidden: !s.job?.kind });
  const jobBtn = (kind, label) => h('button', { class: 'btn sm', disabled: s.job?.running, onclick: () => runJob(kind, log) }, label);
  pop.replaceChildren(
    h('h4', {}, 'Data'),
    h('div', { class: 'small' }, `Club results through `, h('b', {}, s.data_through || '—'), ` · ${s.n_leagues || 0} leagues · ${s.n_teams || 0} clubs`),
    s.intl_updated_h != null ? h('div', { class: 'small muted' }, `Internationals refreshed ${s.intl_updated_h < 48 ? `${Math.round(s.intl_updated_h)} h` : `${Math.round(s.intl_updated_h / 24)} days`} ago`) : '',
    (s.stale || []).length ? h('div', { class: 'warnbox', style: { margin: '8px 0' } }, 'No recent results (off-season, or not published yet): ',
      s.stale.map((x) => `${x.name} (${x.days}d)`).join(', ')) : h('div', { class: 'small', style: { color: 'var(--ok)', margin: '6px 0' } }, '✓ every league current'),
    h('div', { class: 'row', style: { margin: '8px 0' } }, jobBtn('leagues', '🔄 Update league data'), jobBtn('internationals', '🔄 Update internationals')),
    log,
    h('h4', { style: { marginTop: '14px' } }, 'Models'),
    h('div', { class: 'small' }, s.v2?.ok ? `v2 1X2 ${s.v2.tag}` : `v2 unavailable: ${s.v2?.error || '?'}`),
    s.v2?.goals_tag ? h('div', { class: 'small' }, `v2 goals ${s.v2.goals_tag}`) : '',
    s.v2?.trained_through ? h('div', { class: 'small muted' }, `trained through ${s.v2.trained_through}`) : '',
    h('label', { class: 'row small', style: { marginTop: '10px' } }, compare, 'Show v1 / v2 side by side'),
    h('div', { class: 'tiny faint', style: { marginTop: '8px' } }, 'Auto serves v2 for the Turkish Süper Lig (v1’s Turkish model fails on unseen matches) and v1 elsewhere. With bookmaker odds the market is used whichever you pick.'));
}

async function runJob(kind, log) {
  try {
    const r = await api.post(`/api/jobs/${kind}`);
    if (!r.ok) { toast(r.error, { error: true }); return; }
    log.hidden = false;
    toast(kind === 'leagues' ? 'Updating league data — about 2–3 minutes' : 'Updating internationals…');
    for (;;) {
      await new Promise((res) => setTimeout(res, 1500));
      const j = await api.get('/api/jobs');
      log.textContent = (j.log || '').split('\n').slice(-40).join('\n');
      log.scrollTop = log.scrollHeight;
      store.status = await api.get('/api/status'); paintHeader();
      if (!j.running) {
        toast(j.ok ? 'Data updated ✓' : 'Update finished with errors — see the log', { error: !j.ok });
        if (j.ok) {
          // wait for the engine to reload the new data, then refresh the page
          for (let i = 0; i < 60 && store.status.state !== 'ready'; i += 1) { await new Promise((res) => setTimeout(res, 1000)); store.status = await api.get('/api/status'); }
          const { teams } = await api.get('/api/teams');
          store.teams = teams.map((t) => ({ ...t, _f: fold(t.name) }));
          paintHeader(); route();
        }
        break;
      }
    }
  } catch (e) { toast(e.message, { error: true }); }
}

boot();
