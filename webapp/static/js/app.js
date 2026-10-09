// App shell: start-up (wait for the engine), pages, header, the data menu, theme.
import { h, $, toast } from './ui.js';
import { icon } from './icons.js';
import { api, store } from './api.js';
import { Coupon } from './coupon.js';
import { BuildPage } from './build.js';
import { HistoryPage } from './history.js';

const PAGES = { build: BuildPage, history: HistoryPage };
let current = null;

// ------------------------------------------------------------------- theme
const THEMES = { dark: 'Dark', light: 'Light', auto: 'System' };
const stored = () => { try { return localStorage.getItem('fp.theme') || 'dark'; } catch { return 'dark'; } };
function applyTheme() {
  const t = stored();
  const dark = t === 'dark' || (t === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  const btn = $('#theme-btn');
  btn.replaceChildren(icon(dark ? 'moon' : 'sun', 17));
  btn.title = `Theme: ${THEMES[t]}. Click for ${dark ? 'light' : 'dark'}.`;
}
function setTheme(t) { try { localStorage.setItem('fp.theme', t); } catch { /* private mode */ } applyTheme(); }
function initTheme() {
  applyTheme();
  $('#theme-btn').onclick = () => {
    const dark = document.documentElement.dataset.theme === 'dark';
    setTheme(dark ? 'light' : 'dark');
  };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (stored() === 'auto') applyTheme(); });
}

// ------------------------------------------------------------------- start
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
      $('#boot-step').textContent = s.step || 'Starting';
    } catch { $('#boot-step').textContent = 'Waiting for the program'; }
    await new Promise((r) => setTimeout(r, 500));
  }
  await loadTeams();
  store.coupon = new Coupon();
  await store.coupon.load();
  paintHeader();
  window.addEventListener('hashchange', route);
  route();
  setInterval(pollStatus, 8000);
  setInterval(() => api.post('/api/ping').catch(() => {}), 15000);    // lets a packaged app know the page is open
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

async function pollStatus() {
  try {
    const was = store.status?.job?.running;
    store.status = await api.get('/api/status');
    paintHeader();
    if (was && !store.status.job.running) {            // an update just finished: show the new data
      await loadTeams(); store.coupon.cache.clear();
      store.coupon.rows.forEach((r) => store.coupon.analyze(r, true));
      toast('Data updated');
      route();
    }
  } catch { /* the program is restarting or closed */ }
}

// ------------------------------------------------------------------ header
function ago(ts) {
  if (!ts) return 'never';
  const m = Math.round((Date.now() / 1000 - ts) / 60);
  return m < 2 ? 'just now' : m < 90 ? `${m} minutes ago` : m < 2880 ? `${Math.round(m / 60)} hours ago` : `${Math.round(m / 1440)} days ago`;
}

function paintHeader() {
  const s = store.status;
  const btn = $('#data-btn');
  const stale = (s.stale || []).length;
  const d = s.data_through ? new Date(`${s.data_through}T12:00:00`).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) : '';
  btn.replaceChildren(s.job?.running ? h('span', { class: 'spin-sm' }) : '', s.job?.running ? ' Updating…' : `Data to ${d}`,
    stale && !s.job?.running ? h('span', { class: 'warn' }, ` · ${stale} late`) : '');
  btn.onclick = togglePop;
}

function togglePop() {
  const pop = $('#data-pop');
  if (!pop.hidden) { pop.hidden = true; return; }
  paintPop(); pop.hidden = false;
  const off = (e) => { if (!pop.contains(e.target) && !$('#data-btn').contains(e.target)) { pop.hidden = true; document.removeEventListener('mousedown', off); } };
  setTimeout(() => document.addEventListener('mousedown', off), 0);
}

function paintPop() {
  const s = store.status; const pop = $('#data-pop');
  const log = h('pre', { hidden: !s.job?.kind });
  const upd = h('button', { class: 'btn sm', disabled: !!s.job?.running, onclick: () => runUpdate(log) }, icon('refresh', 14), s.job?.running ? 'Updating…' : 'Update now');
  const auto = h('input', { type: 'checkbox', checked: !!s.auto_update });
  auto.addEventListener('change', async () => { store.status = await api.post('/api/settings', { auto_update: auto.checked }); });
  const themeSel = h('select', { class: 'txt sm', style: { width: '130px' }, 'aria-label': 'Theme' },
    ...Object.entries(THEMES).map(([k, v]) => h('option', { value: k, selected: k === stored() }, v)));
  themeSel.addEventListener('change', () => setTheme(themeSel.value));
  const m = s.models || {};
  pop.replaceChildren(
    h('h4', {}, 'Data'),
    h('div', { class: 'small' }, 'Results to ', h('b', {}, s.data_through || '—'), ` · ${s.n_leagues || 0} leagues · ${s.n_teams || 0} clubs`),
    h('div', { class: 'small muted' }, `Last updated ${ago(s.last_update)}`),
    (s.stale || []).length
      ? h('div', { class: 'box warn', style: { margin: '8px 0' } }, 'No recent results from: ', s.stale.map((x) => `${x.name} (${x.days} days)`).join(', '), '. The league is between seasons or the new season is not published yet.')
      : h('div', { class: 'small pos', style: { margin: '6px 0' } }, 'Every league is up to date.'),
    h('div', { class: 'row', style: { margin: '8px 0' } }, upd, h('label', { class: 'check' }, auto, 'Update automatically')),
    log,
    h('h4', {}, 'How the numbers are made'),
    h('div', { class: 'tasklist' },
      h('div', { class: 'task' }, h('b', {}, 'Result 1 · X · 2'), h('span', {}, 'The bookmaker odds when the match has them. If not, the model, which uses team form and ratings built from past odds.')),
      h('div', { class: 'task' }, h('b', {}, 'Over / under'), h('span', {}, 'The scoreline model, combined with the over/under odds when they exist.')),
      h('div', { class: 'task' }, h('b', {}, 'Both teams score'), h('span', {}, 'The scoreline model.')),
      h('div', { class: 'task' }, h('b', {}, 'Correct score'), h('span', {}, 'The scoreline model, adjusted so it agrees with the result and over/under numbers above.'))),
    m.ok ? h('div', { class: 'small muted', style: { marginTop: '8px' } }, `Models trained to ${m.trained_through}.`) : '',
    h('h4', {}, 'Appearance'),
    h('div', { class: 'row' }, h('span', { class: 'small' }, 'Theme'), themeSel),
    h('div', { class: 'row', style: { marginTop: '14px' } },
      h('span', { class: 'small faint' }, `Version ${s.version}`), h('span', { class: 'spacer' }),
      h('button', { class: 'btn sm', onclick: quitApp }, 'Quit the program')));
}

async function quitApp() {
  try { await api.post('/api/quit'); } catch { /* the program closes the connection */ }
  document.body.replaceChildren(h('div', { class: 'boot' }, h('div', { id: 'boot-step' }, 'The program has stopped.'),
    h('div', { class: 'muted small' }, 'You can close this tab. Open the app again to continue.')));
}

async function runUpdate(log) {
  try {
    const r = await api.post('/api/jobs/all');
    if (!r.ok) { toast(r.error, { error: true }); return; }
    log.hidden = false;
    toast('Updating the data. This takes one or two minutes.');
    for (;;) {
      await new Promise((res) => setTimeout(res, 1500));
      const j = await api.get('/api/jobs');
      log.textContent = (j.log || '').split('\n').slice(-30).join('\n');
      log.scrollTop = log.scrollHeight;
      store.status = await api.get('/api/status'); paintHeader();
      if (!j.running) {
        toast(j.ok ? 'Data updated' : 'The update had errors. See the log.', { error: !j.ok });
        if (j.ok) {
          for (let i = 0; i < 90 && store.status.state !== 'ready'; i += 1) { await new Promise((res) => setTimeout(res, 1000)); store.status = await api.get('/api/status'); }
          await loadTeams();
          store.coupon.cache.clear(); store.coupon.rows.forEach((r) => store.coupon.analyze(r, true));
          paintHeader(); paintPop(); route();
        }
        break;
      }
    }
  } catch (e) { toast(e.message, { error: true }); }
}

initTheme();
boot();
