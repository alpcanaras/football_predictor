# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for the macOS app. Build with packaging/build.sh.
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))
BUILD = os.path.join(ROOT, 'build')
VERSION = open(os.path.join(BUILD, 'seed', 'VERSION')).read().strip()

hidden = (collect_submodules('scripts') + collect_submodules('webapp')
          + collect_submodules('uvicorn') + collect_submodules('starlette') + collect_submodules('anyio')
          + ['xgboost', 'xgboost.sklearn', 'lightgbm', 'lightgbm.sklearn', 'sklearn.ensemble._forest',
             'sklearn.utils._typedefs', 'sklearn.neighbors._partition_nodes', 'scipy.special._cdflib',
             'h11', 'certifi', 'multipart', 'lxml', 'bs4'])
datas = [(os.path.join(ROOT, 'webapp', 'static'), 'webapp/static'),
         (os.path.join(BUILD, 'seed'), 'seed')]
datas += collect_data_files('xgboost') + collect_data_files('lightgbm') + collect_data_files('certifi')
binaries = collect_dynamic_libs('xgboost') + collect_dynamic_libs('lightgbm')

a = Analysis(
    [os.path.join(ROOT, 'packaging', 'launcher.py')],
    pathex=[ROOT], binaries=binaries, datas=datas, hiddenimports=hidden, hookspath=[], runtime_hooks=[],
    excludes=['matplotlib', 'seaborn', 'jupyterlab', 'jupyter', 'notebook', 'IPython', 'tkinter', 'optuna',
              'skimage', 'sympy', 'pytest', 'sphinx', 'docutils', 'numba', 'torch', 'tensorflow', 'streamlit',
              'altair', 'pydeck', 'plotly', 'bokeh', 'dask', 'pyarrow', 'cv2', 'PyQt5', 'PyQt6', 'PySide6'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Football Predictor', debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          disable_windowed_traceback=False, argv_emulation=False, target_arch=None,
          codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='Football Predictor')
app = BUNDLE(coll, name='Football Predictor.app', icon=os.path.join(BUILD, 'icon', 'icon.icns'),
             bundle_identifier='app.footballpredictor.local', version=VERSION,
             info_plist={'CFBundleDisplayName': 'Football Predictor', 'CFBundleShortVersionString': VERSION,
                         'LSMinimumSystemVersion': '14.0', 'NSHighResolutionCapable': True,
                         'LSApplicationCategoryType': 'public.app-category.sports'})
