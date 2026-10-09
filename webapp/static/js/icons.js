// Small stroke icons (24x24 grid) so the interface carries no emoji.
const P = {
  plus: 'M12 5v14M5 12h14',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  x: 'M6 6l12 12M18 6L6 18',
  search: 'M11 4a7 7 0 100 14 7 7 0 000-14zM20 20l-4-4',
  sun: 'M12 8a4 4 0 100 8 4 4 0 000-8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4',
  moon: 'M20 14.5A8 8 0 0111.5 4a8 8 0 108.5 10.5z',
  refresh: 'M20 11a8 8 0 10-2.3 5.7M20 4v7h-7',
  chevron: 'M6 9l6 6 6-6',
  lock: 'M7 11V8a5 5 0 0110 0v3M6 11h12v9H6z',
  info: 'M12 3a9 9 0 100 18 9 9 0 000-18zM12 11v6M12 7.5v.01',
  more: 'M5 12h.01M12 12h.01M19 12h.01',
  download: 'M12 4v11M7 11l5 5 5-5M5 20h14',
  sliders: 'M4 7h10M18 7h2M4 17h2M10 17h10M14 4v6M6 14v6',
  list: 'M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01',
  history: 'M4 12a8 8 0 108-8 8 8 0 00-5.7 2.4L4 8M4 4v4h4M12 8v4l3 2',
  copy: 'M9 9h10v11H9zM5 15V4h10',
  print: 'M7 9V4h10v5M6 18H4v-8h16v8h-2M7 14h10v6H7z',
  save: 'M5 4h11l3 3v13H5zM8 4v5h7M8 20v-6h8v6',
  trash: 'M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13',
  filter: 'M4 5h16l-6 8v6l-4-2v-4z',
};

export function icon(name, size = 16, cls = '') {
  const NS = 'http://www.w3.org/2000/svg';
  const s = document.createElementNS(NS, 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('width', size); s.setAttribute('height', size);
  s.setAttribute('fill', 'none'); s.setAttribute('stroke', 'currentColor');
  s.setAttribute('stroke-width', '2'); s.setAttribute('stroke-linecap', 'round'); s.setAttribute('stroke-linejoin', 'round');
  s.setAttribute('aria-hidden', 'true');
  if (cls) s.setAttribute('class', cls);
  const p = document.createElementNS(NS, 'path');
  p.setAttribute('d', P[name] || P.info);
  s.append(p);
  return s;
}
