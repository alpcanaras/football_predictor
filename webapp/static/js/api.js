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

// Shared app state (status, team index, preferences).
export const store = {
  status: null,
  teams: [],
  teamIndex: new Map(),
  compare: localStorage.getItem('fp.compare') !== '0',
  listeners: new Set(),
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
  emit(what) { for (const fn of this.listeners) fn(what); },
};
