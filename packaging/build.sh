#!/bin/bash
# Build "Football Predictor.app" and a shareable .dmg (Apple Silicon, macOS 14+).
#
#   packaging/build.sh          # -> dist/Football Predictor.app and dist/Football-Predictor-<version>.dmg
#
# Needs: the project's venv (with pyinstaller), Xcode command line tools.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./venv/bin/python
BUILD=build
rm -rf "$BUILD/pyi" dist
mkdir -p "$BUILD/icon"

echo "==> 1/6  seed (data + the two model bundles)"
$PY packaging/prepare_seed.py
VERSION=$(cat "$BUILD/seed/VERSION")

echo "==> 2/6  icon"
$PY packaging/make_icon.py "$BUILD/icon"

echo "==> 3/6  PyInstaller"
$PY -m PyInstaller packaging/FootballPredictor.spec --noconfirm --clean \
    --distpath dist --workpath "$BUILD/pyi" 2>&1 | grep -v "^[0-9]* INFO\|WARNING: Hidden\|^$" | tail -25
APP="dist/Football Predictor.app"
[ -d "$APP" ] || { echo "build failed: no app"; exit 1; }

echo "==> 4/6  one libomp for everyone (scikit-learn's, which runs on macOS 11+)"
OMP=$($PY -c "import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), '.dylibs', 'libomp.dylib'))")
find "$APP" -name "libomp*.dylib" | while read -r f; do
    cp "$OMP" "$f"
    echo "    replaced ${f#$APP/}"
done

echo "==> 5/6  sign (ad hoc; there is no Apple developer account behind this build)"
codesign --force --deep --sign - "$APP" 2>&1 | tail -3
codesign --verify --deep --strict "$APP" && echo "    signature ok"

echo "==> 6/6  disk image"
STAGE="$BUILD/dmg"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp packaging/READ_ME_FIRST.txt "$STAGE/READ ME FIRST.txt"
DMG="dist/Football-Predictor-$VERSION.dmg"
hdiutil create -volname "Football Predictor" -srcfolder "$STAGE" -ov -format UDZO -fs HFS+ "$DMG" >/dev/null
echo
echo "built: $APP  ($(du -sh "$APP" | cut -f1))"
echo "built: $DMG  ($(du -sh "$DMG" | cut -f1))"
