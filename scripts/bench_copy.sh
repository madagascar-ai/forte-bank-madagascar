#!/bin/bash
# Run a private copy of the dev service for benchmarks, so autodeploy restarts of :8000 cannot
# interrupt a run. Same settings, weights, labels and videos as dev; separate job data; code from
# this checkout. Usage: scripts/bench_copy.sh [port]   (default 8001, loopback only)
set -euo pipefail
CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$HOME/Library/Application Support/ForteGuard"
PORT="${1:-8001}"

export FORTE_DATA_DIR="$HOME/forte-bench-data"
export FORTE_CONFIG_DIR="$CHECKOUT/config"
export FORTE_HOST=127.0.0.1
export FORTE_PORT="$PORT"
export FORTE_BENCHMARK_ENABLED=true
# Record the code version in each run, like the deployed service does.
export FORTE_RELEASE_SHA="$(git -C "$CHECKOUT" rev-parse HEAD)"
mkdir -p "$FORTE_DATA_DIR"

# Relative paths in the dev .env (models/, videos/) resolve against the dev workspace.
cd "$WORKSPACE"
echo "Benchmark copy: http://127.0.0.1:$PORT  code $(git -C "$CHECKOUT" rev-parse --short HEAD)  data $FORTE_DATA_DIR"
exec "$CHECKOUT/.venv/bin/forte-guard" serve --host 127.0.0.1 --port "$PORT"
