#!/usr/bin/env bash
# Activate an uploaded release on the GPU machine: drain jobs, switch, verify, roll back on failure.
# Layout under $BASE: releases/<sha>/ (code), app -> releases/<sha>, venv/, shared/{.env,models,data}.
set -euo pipefail
# A dropped SSH session must not kill a half-done switch: finish the switch or the rollback regardless.
trap '' HUP

SHA="$1"
BASE="${FORTE_BASE:-$HOME/forte-guard}"
NEW="$BASE/releases/$SHA"
VENV="$BASE/venv"
UNIT="forte-guard.service"

log() { printf '[activate %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

env_value() { grep -E "^$1=" "$BASE/shared/.env" | tail -1 | cut -d= -f2- | tr -d "\"'"; }
URL="http://$(env_value FORTE_HOST):$(env_value FORTE_PORT)"
KEY="$(env_value FORTE_API_KEY)"

[ -d "$NEW" ] || { log "release $SHA not uploaded"; exit 1; }
echo "$SHA" > "$NEW/REVISION"
ln -sfn "$BASE/shared/.env" "$NEW/.env"

# First run after the manual install: keep the old checkout as a release to roll back to.
if [ -d "$BASE/app" ] && [ ! -L "$BASE/app" ]; then
  old="$(cat "$BASE/app/REVISION" 2>/dev/null || echo initial)"
  mv "$BASE/app" "$BASE/releases/$old"
  ln -s "$BASE/releases/$old" "$BASE/app"
fi
PREV="$(readlink -f "$BASE/app" 2>/dev/null || true)"

install() {  # Point the editable install at a release; reinstall dependencies only when they changed.
  local dir="$1"
  if [ -n "$PREV" ] && cmp -s "$dir/pyproject.toml" "$PREV/pyproject.toml"; then
    "$VENV/bin/pip" install -q --no-deps -e "$dir"
  else
    "$VENV/bin/pip" install -q -e "$dir[vision]"
  fi
}

switch_to() {
  ln -sfn "$1" "$BASE/app.next" && mv -T "$BASE/app.next" "$BASE/app"
  install "$1"
  mkdir -p "$HOME/.config/systemd/user"
  if ! cmp -s "$1/deploy/gpu/$UNIT" "$HOME/.config/systemd/user/$UNIT"; then
    cp "$1/deploy/gpu/$UNIT" "$HOME/.config/systemd/user/$UNIT"
    systemctl --user daemon-reload
  fi
  systemctl --user enable "$UNIT" >/dev/null 2>&1 || true
  systemctl --user restart "$UNIT"
}

healthy() {  # The new code answers with its own SHA and every readiness check passes.
  local expected="$1"
  URL="http://$(env_value FORTE_HOST):$(env_value FORTE_PORT)"  # settings may have just changed
  KEY="$(env_value FORTE_API_KEY)"
  for _ in $(seq 1 45); do
    if curl -fsS -m 3 "$URL/health" 2>/dev/null | grep -q "\"release\":\"$expected\""; then
      curl -fsS -m 20 -H "X-API-Key: $KEY" "$URL/api/v1/readiness" >/dev/null && return 0
    fi
    sleep 2
  done
  return 1
}

# Do not cut a job in the middle: wait (bounded) until nothing is processing.
for _ in $(seq 1 48); do
  busy="$(curl -fsS -m 5 -H "X-API-Key: $KEY" "$URL/api/v1/jobs" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); jobs=d.get("jobs", d) if isinstance(d, dict) else d; print(sum(j.get("status") in ("queued","processing") for j in jobs))' 2>/dev/null || echo 0)"
  [ "${busy:-0}" = "0" ] && break
  log "waiting for $busy job(s) to finish"
  sleep 5
done

# Settings managed in GitHub (FORTE_* secrets and variables) arrive as shared/.env.github.
ENV="$BASE/shared/.env"
OVERRIDES="$BASE/shared/.env.github"
ENV_CHANGED=""
if [ -f "$OVERRIDES" ]; then
  cp -p "$ENV" "$ENV.previous"
  log "$(python3 "$NEW/deploy/gpu/merge_env.py" "$ENV" "$OVERRIDES")"
  rm -f "$OVERRIDES"
  ENV_CHANGED=1
fi

log "switching to $SHA (previous: $(basename "${PREV:-none}"))"
switch_to "$NEW"
if healthy "$SHA"; then
  log "release $SHA is live"
  ls -1dt "$BASE"/releases/*/ | tail -n +6 | xargs -r rm -rf  # keep the last five releases
  exit 0
fi

log "release $SHA failed health/readiness; rolling back"
if [ -n "$ENV_CHANGED" ]; then
  cp -p "$ENV.previous" "$ENV"
  log "restored the previous server settings"
fi
if [ -n "$PREV" ] && [ "$PREV" != "$NEW" ]; then
  switch_to "$PREV"
  healthy "$(cat "$PREV/REVISION" 2>/dev/null || basename "$PREV")" && log "rolled back to $(basename "$PREV")"
fi
exit 1
