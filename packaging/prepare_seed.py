"""Collect exactly what the app ships: league data and the two model bundles.

Not shipped on purpose: coupons and history (data/_toto), staging and retired
folders, the downloaded fixture feed, the v1 models, caches and experiments.
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'build', 'seed')
SKIP_DATA = {'_toto', '_incoming', '_retired', '_fixtures', '_backup'}


def main() -> str:
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    data_src = os.path.join(ROOT, 'data')
    n = size = 0
    for name in sorted(os.listdir(data_src)):
        p = os.path.join(data_src, name)
        if name in SKIP_DATA or not os.path.isdir(p):
            continue
        shutil.copytree(p, os.path.join(OUT, 'data', name),
                        ignore=shutil.ignore_patterns('.DS_Store', '*.bak', '*.tmp', '*.orig'))
    for dp, _, fs in os.walk(os.path.join(OUT, 'data')):
        for f in fs:
            n += 1
            size += os.path.getsize(os.path.join(dp, f))
    mdir = os.path.join(ROOT, 'models', 'v2')
    out_m = os.path.join(OUT, 'models', 'v2')
    os.makedirs(out_m)
    keep = []
    for marker in ('LATEST', 'LATEST_GOALS'):
        with open(os.path.join(mdir, marker)) as f:
            keep.append(f.read().strip())
        shutil.copy(os.path.join(mdir, marker), os.path.join(out_m, marker))
    for tag in keep:
        shutil.copytree(os.path.join(mdir, tag), os.path.join(out_m, tag))
    try:
        head = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        head = 'nogit'
    version = f"{time.strftime('%Y.%m.%d')}-{head}"
    with open(os.path.join(OUT, 'VERSION'), 'w') as f:
        f.write(version)
    print(f'seed: {n} data files ({size / 1e6:.0f} MB), models {keep}, version {version}')
    return version


if __name__ == '__main__':
    main()
