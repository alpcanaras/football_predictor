// JSON over fetch; server errors arrive as {error} and become exceptions.
async function call(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty */ }
  if (!res.ok) throw new Error(data?.error || `${res.status} ${res.statusText}`);
  return data;
}

export const api = {
  get: (p) => call('GET', p),
  post: (p, b) => call('POST', p, b || {}),
  put: (p, b) => call('PUT', p, b || {}),
  del: (p) => call('DELETE', p),
};

// Shared app state.
export const store = {
  status: null,
  teams: [],                 // [{name, display, kind, league_name, keys}]
  byName: new Map(),         // canonical name -> team
  coupon: null,              // Coupon (set by app.js)
  listeners: new Set(),
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
  emit(what) { for (const fn of this.listeners) fn(what); },
  display(name) { return this.byName.get(name)?.display || name; },
  setTeams(list) {
    this.teams = list;
    this.byName = new Map(list.map((t) => [t.name, t]));
  },
};
