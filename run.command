#!/bin/bash
# Double-click this file to launch the Football Predictor in your browser.
# (macOS: first time, right-click → Open to get past Gatekeeper.)
# The classic Streamlit version is still here: run_classic.command.
cd "$(dirname "$0")" || exit 1

if [ ! -d venv ]; then
  echo "First run: creating environment (one-time, ~1 min)…"
  python3 -m venv venv
  ./venv/bin/pip install -q --upgrade pip
  ./venv/bin/pip install -q -r requirements.txt
fi

# The web app needs starlette + uvicorn (they come with the requirements)
./venv/bin/python -c "import starlette, uvicorn" 2>/dev/null \
  || ./venv/bin/pip install -q starlette uvicorn

echo "Launching Football Predictor… (close this window to stop)"
exec ./venv/bin/python -m webapp
