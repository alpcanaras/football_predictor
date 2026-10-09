"""
Football Predictor — entry point of the packaged app
====================================================
Double-clicking the app runs this. It

  1. keeps everything the app writes in ~/Library/Application Support/
     Football Predictor (the app bundle itself is read-only and replaced on
     every update),
  2. copies the shipped data and models there on the first start, and the
     shipped models again whenever the app version changes,
  3. starts the web app on 127.0.0.1 and opens the browser,
  4. exits by itself ten minutes after the browser page was closed.

Starting the app a second time just opens the page of the running one.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

APP_NAME = 'Football Predictor'
PORT = int(os.environ.get('FOOTBALL_PREDICTOR_PORT', '8765'))
IDLE_EXIT_S = 600          # no page open for this long -> quit
NEVER_OPENED_S = 900       # the page never connected -> quit


def bundle_root() -> str:
    """Where the shipped files are: the PyInstaller folder, or the checkout."""
    return getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def user_home() -> str:
    return (os.environ.get('FOOTBALL_PREDICTOR_APP_HOME')
            or os.path.expanduser(f'~/Library/Application Support/{APP_NAME}'))


def _read(path: str) -> str:
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return ''


def seed(home: str, say=print) -> str:
    """First-run copy of the shipped data; models follow the app version."""
    src = os.path.join(bundle_root(), 'seed')
    version = _read(os.path.join(src, 'VERSION')) or 'dev'
    data_src, data_dst = os.path.join(src, 'data'), os.path.join(home, 'data')
    if os.path.isdir(data_src):
        os.makedirs(data_dst, exist_ok=True)
        for name in sorted(os.listdir(data_src)):                 # only what is missing
            if not os.path.exists(os.path.join(data_dst, name)):
                say(f'seeding data/{name}')
                shutil.copytree(os.path.join(data_src, name), os.path.join(data_dst, name))
    marker = os.path.join(home, '.seed_version')
    if _read(marker) != version:
        models_src = os.path.join(src, 'models')
        if os.path.isdir(models_src):
            say(f'installing models for version {version}')
            shutil.rmtree(os.path.join(home, 'models'), ignore_errors=True)
            shutil.copytree(models_src, os.path.join(home, 'models'))
            shutil.rmtree(os.path.join(home, 'reports'), ignore_errors=True)   # cached states
        with open(marker, 'w', encoding='utf-8') as f:
            f.write(version)
    return version


def _running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/status', timeout=1.5) as r:
            return r.status == 200 and b'"state"' in r.read(400)
    except Exception:
        return False


def _free_port(start: int) -> int:
    for p in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return start


def _open_when_ready(port: int) -> None:
    for _ in range(300):
        if _running(port):
            webbrowser.open(f'http://127.0.0.1:{port}/')
            return
        time.sleep(0.2)


def _idle_watch(started: float) -> None:
    from webapp import server
    while True:
        time.sleep(15)
        ping = server._PING
        now = time.time()
        if ping['seen'] and now - ping['t'] > IDLE_EXIT_S:
            server.shutdown()
        if not ping['seen'] and now - started > NEVER_OPENED_S:
            server.shutdown()


def main() -> None:
    multiprocessing.freeze_support()
    home = user_home()
    os.makedirs(home, exist_ok=True)
    if getattr(sys, 'frozen', False):                      # no terminal: log to a file
        log = open(os.path.join(home, 'app.log'), 'a', buffering=1, encoding='utf-8')
        sys.stdout = sys.stderr = log
        print(f'\n--- {time.strftime("%Y-%m-%d %H:%M:%S")} start')
    if _running(PORT):                                     # already open: just show it
        webbrowser.open(f'http://127.0.0.1:{PORT}/')
        return

    os.environ['FOOTBALL_PREDICTOR_HOME'] = home
    os.environ.setdefault('FOOTBALL_PREDICTOR_AUTOUPDATE', '1')
    version = seed(home)
    os.environ['FOOTBALL_PREDICTOR_VERSION'] = version

    port = _free_port(PORT)
    import uvicorn
    from webapp.server import create_app
    app = create_app()
    threading.Thread(target=_open_when_ready, args=(port,), daemon=True).start()
    threading.Thread(target=_idle_watch, args=(time.time(),), daemon=True).start()
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning', log_config=None)


if __name__ == '__main__':
    main()
