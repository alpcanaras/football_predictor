#!/bin/bash
# Double-click this file to launch the Football Predictor app in your browser.
# (macOS: first time, right-click → Open to get past Gatekeeper.)
cd "$(dirname "$0")" || exit 1

if [ ! -d venv ]; then
  echo "First run: creating environment (one-time, ~1 min)…"
  python3 -m venv venv
  ./venv/bin/pip install -q --upgrade pip
  ./venv/bin/pip install -q -r requirements.txt
fi

# Make sure streamlit is present even if requirements changed
./venv/bin/python -c "import streamlit" 2>/dev/null || ./venv/bin/pip install -q streamlit

echo "Launching Football Predictor… (close this window to stop)"
exec ./venv/bin/streamlit run app.py
