#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  PerfoSim Pro v3 — Startup Script
#  Niger Delta Perforation & HF Optimization Framework
#  Usage: bash run.sh   (or: make run)
# ══════════════════════════════════════════════════════════════════════════════
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "══════════════════════════════════════════════════════════════════"
echo "  PerfoSim Pro v3 — Niger Delta Perf & HF Optimization Framework"
echo "  Real-Data Upload · Coupled Simulation · PINNs · NSGA-II"
echo "  Statistical Validation · 2D-9D Visualization"
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo "Installing / verifying dependencies..."

pip install -r requirements.txt --quiet 2>/dev/null || \
pip install -r requirements.txt --quiet --break-system-packages 2>/dev/null || true

echo ""
echo "Starting Streamlit on http://localhost:8501 ..."
echo "(theme, upload size, and other server settings come from .streamlit/config.toml)"
echo ""

streamlit run app.py \
  --server.port=8501 \
  --server.address=localhost \
  --server.headless=true
