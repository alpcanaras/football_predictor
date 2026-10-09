// App shell: boot (wait for the engine), router, data menu.
import { h, $, toast } from './ui.js';
import { api, store } from './api.js';
import { Coupon } from './coupon.js';
import { BuildPage } from './build.js';
import { HistoryPage } from './history.js';

const PAGES = { build: BuildPage, history: HistoryPage };
let current = null;

async function loadTeams() {
  const { teams } = await api.get('/api/teams');
  store.setTeams(teams.map((t) => ({ ...t })));
}

async function boot() {
  for (;;) {
    try {
      const s = await api.get('/api/status');
      store.status = s;
      if (s.state === 'ready') break;
      if (s.state === 'error') { $('#boot-step').textContent = `Could not start: ${s.error}`; return; }
      $('#boot-step').textContent = `${s.step || 'Starting'}…`;
    } catch { $('#boot-step').textContent = 'Waiting for the server…'; }
    await new Promise((r) => setTimeout(r, 600));
  }
  await loadTeams();
  store.coupon = new Coupon();
  await store.coupon.load();
  paintHeader();
  window.addEventListener('hashchange', route);
  route();
}

function route() {
  const name = (location.hash.replace(/^#\/?/, '') || 'build').split('?')[0];
  const key = PAGES[name] ? name : 'build';
  document.querySelectorAll('#tabs a').forEach((a) => a.classList.toggle('active', a.dataset.page === key));
  current?.destroy?.();
  const root = $('#app');
  root.replaceChildren();
  current = new PAGES[key](root, store.coupon);
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ header
function paintHeader() {
  const s = store.status;
  const pill = $('#data-pill');
  const stale = (s.stale || []).length;
  pill.textContent = s.job?.running ? '⟳ Updating…' : `📡 ${s.data_through || '—'}${stale ? ` · ⚠ ${stale}` : ''}`;
  pill.classList.toggle('warn', stale > 0);
  pill.onclick = togglePop;
}

function togglePop() {
  const pop = $('#data-pop');
  if (!pop.hidden) { pop.hidden = true; return; }
  paintPop(); pop.hidden = false;
  const off = (e) => { if (!pop.contains(e.target) && e.target !== $('#data-pill')) { pop.hidden = true; document.removeEventListener('mousedown', off); } };
  setTimeout(() => document.addEventListener('mousedown', off), 0);
}

async function setMode(mode) {
  if (mode === store.status.model_mode) return;
  try {
    store.status = await api.post('/api/settings', { model_mode: mode });
    paintHeader(); paintPop();
    store.coupon.reanalyzeModelRows();
    store.emit('mode');
    toast(`Model for matches without odds: ${mode}`);
  } catch (e) { toast(e.message, { error: true }); }
}

function paintPop() {
  const s = store.status; const pop = $('#data-pop');
  const log = h('pre', { hidden: !s.job?.kind });
  const jobBtn = (kind, label) => h('button', { class: 'btn sm', disabled: s.job?.running, onclick: () => runJob(kind, log) }, label);
  const seg = h('div', { class: 'seg' }, ...[['v2', 'v2 (recommended)'], ['auto', 'Auto'], ['v1', 'v1']].map(([m, label]) =>
    h('button', { class: `seg-btn${s.model_mode === m ? ' on' : ''}`, disabled: m !== 'v1' && !s.v2?.ok, onclick: () => setMode(m) }, label)));
  pop.replaceChildren(
    h('h4', {}, 'Data'),
    h('div', { class: 'small' }, 'Club results through ', h('b', {}, s.data_through || '—'), ` · ${s.n_leagues || 0} leagues · ${s.n_teams || 0} clubs`),
    s.intl_updated_h != null ? h('div', { class: 'small muted' }, `Internationals refreshed ${s.intl_updated_h < 48 ? `${Math.round(s.intl_updated_h)} h` : `${Math.round(s.intl_updated_h / 24)} days`} ago`) : '',
    (s.stale || []).length ? h('div', { class: 'warnbox', style: { margin: '8px 0' } }, 'No recent results (off-season, or not published yet): ',
      s.stale.map((x) => `${x.name} (${x.days}d)`).join(', ')) : h('div', { class: 'small', style: { color: 'var(--ok)', margin: '6px 0' } }, '✓ every league current'),
    h('div', { class: 'row', style: { margin: '8px 0' } }, jobBtn('leagues', '🔄 Update league data'), jobBtn('internationals', '🔄 Update internationals')),
    log,
    h('h4', { style: { marginTop: '14px' } }, 'Models'),
    h('div', { class: 'small' }, s.v2?.ok ? `v2: ${s.v2.tag}` : `v2 unavailable: ${s.v2?.error || '?'}`),
    s.v2?.goals_tag ? h('div', { class: 'small' }, `goals: ${s.v2.goals_tag}`) : '',
    s.v2?.trained_through ? h('div', { class: 'small muted' }, `trained through ${s.v2.trained_through}`) : '',
    h('div', { class: 'small muted', style: { margin: '10px 0 4px' } }, 'Model for matches without bookmaker odds'),
    seg,
    h('div', { class: 'tiny faint', style: { marginTop: '8px' } }, 'With bookmaker odds the odds are always used: on unseen matches they beat both models. v1 stays available for comparison and is shown next to v2 on every match.'));
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
          for (let i = 0; i < 90 && store.status.state !== 'ready'; i += 1) { await new Promise((res) => setTimeout(res, 1000)); store.status = await api.get('/api/status'); }
          await loadTeams();
          store.coupon.cache.clear(); store.coupon.rows.forEach((r) => store.coupon.analyze(r, true));
          paintHeader(); route();
        }
        break;
      }
    }
  } catch (e) { toast(e.message, { error: true }); }
}

boot();
