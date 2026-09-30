"""
Football Predictor — web app server
===================================
A small JSON API over webapp/engine.py plus the static single-page app in
webapp/static/. Runs locally only (127.0.0.1).

    python -m webapp                 # opens http://127.0.0.1:8765
    python -m webapp --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import webbrowser

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from webapp.engine import ENGINE

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


class FreshStatic(StaticFiles):
    """Static files the browser must revalidate (ETag -> 304 when unchanged),
    so an updated app never runs yesterday's cached scripts."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers['Cache-Control'] = 'no-cache'
        return resp


def _clean(obj):
    """JSON-safe: numpy scalars -> Python, NaN -> None."""
    import math
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def api(fn):
    """Wrap an endpoint: JSON body in, JSON out, errors as {error} with 400."""
    async def endpoint(request: Request):
        from starlette.concurrency import run_in_threadpool
        try:
            body = await request.json() if request.method in ('POST', 'PUT') else {}
        except Exception:
            body = {}
        try:
            out = await run_in_threadpool(fn, request, body)
            return JSONResponse(_clean(out))
        except Exception as e:
            status = 503 if 'warming up' in str(e) else 400
            return JSONResponse({'error': f'{e}'}, status_code=status)
    return endpoint


# ------------------------------------------------------------------ handlers
def status(_r, _b):
    return ENGINE.status()


def set_settings(_r, b):
    if 'model_mode' in b:
        return ENGINE.set_model_mode(b['model_mode'])
    return ENGINE.status()


def teams(_r, _b):
    ENGINE.need_ready()
    return {'teams': ENGINE.teams}


def get_coupon(r, _b):
    return ENGINE.load_coupon(r.path_params['game'])


def put_coupon(r, b):
    return ENGINE.save_coupon(r.path_params['game'], b)


def parse(_r, b):
    return {'rows': ENGINE.parse_text(b.get('text', ''))}


def analyze(_r, b):
    return ENGINE.analyze_row(b.get('home'), b.get('away'), b.get('odds'))


def analyze_many(_r, b):
    return {'rows': [ENGINE.analyze_row(x.get('home'), x.get('away'), x.get('odds'))
                     for x in b.get('rows', [])]}


def ticket(_r, b):
    return ENGINE.ticket(b)


def ladder(_r, b):
    return {'ladder': ENGINE.ladder(b)}


def sweep(_r, b):
    return {'sweep': ENGINE.sweep(b)}


def fill_odds(_r, b):
    return {'odds': ENGINE.fill_odds(b.get('rows', []))}


def fixtures(r, _b):
    return ENGINE.fixtures(int(r.query_params.get('days', 3)))


def match(r, _b):
    return ENGINE.match(r.query_params.get('home', ''), r.query_params.get('away', ''))


def internationals(r, _b):
    return ENGINE.internationals(int(r.query_params.get('days', 14)))


def history(_r, _b):
    return ENGINE.history()


def save_history(_r, b):
    return ENGINE.save_history(b)


def delete_history(r, _b):
    return ENGINE.delete_history(r.path_params['id'])


def start_job(r, _b):
    return ENGINE.start_job(r.path_params['kind'])


def job(_r, _b):
    return ENGINE.job()


async def index(_request):
    return FileResponse(os.path.join(STATIC, 'index.html'),
                        headers={'Cache-Control': 'no-store'})


routes = [
    Route('/', index),
    Route('/api/status', api(status)),
    Route('/api/settings', api(set_settings), methods=['POST']),
    Route('/api/teams', api(teams)),
    Route('/api/coupon/{game}', api(get_coupon)),
    Route('/api/coupon/{game}', api(put_coupon), methods=['PUT']),
    Route('/api/parse', api(parse), methods=['POST']),
    Route('/api/analyze', api(analyze), methods=['POST']),
    Route('/api/analyze-many', api(analyze_many), methods=['POST']),
    Route('/api/ticket', api(ticket), methods=['POST']),
    Route('/api/ladder', api(ladder), methods=['POST']),
    Route('/api/sweep', api(sweep), methods=['POST']),
    Route('/api/fill-odds', api(fill_odds), methods=['POST']),
    Route('/api/fixtures', api(fixtures)),
    Route('/api/match', api(match)),
    Route('/api/internationals', api(internationals)),
    Route('/api/history', api(history)),
    Route('/api/history', api(save_history), methods=['POST']),
    Route('/api/history/{id}', api(delete_history), methods=['DELETE']),
    Route('/api/jobs/{kind}', api(start_job), methods=['POST']),
    Route('/api/jobs', api(job)),
    Mount('/static', FreshStatic(directory=STATIC), name='static'),
]


def create_app(start_engine: bool = True) -> Starlette:
    app = Starlette(routes=routes)
    if start_engine:
        ENGINE.start()
    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description='Football Predictor web app')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()
    url = f'http://127.0.0.1:{a.port}/'
    if not a.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(url)),
                         daemon=True).start()
    print(f'\n  Football Predictor running at {url}  (Ctrl+C to stop)\n')
    uvicorn.run(create_app(), host='127.0.0.1', port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
