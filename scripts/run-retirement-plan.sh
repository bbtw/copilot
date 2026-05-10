#!/usr/bin/env bash
set -euo pipefail

HOST="${LOCAL_LLM_GATEWAY_HOST:-127.0.0.1}"
PORT="${LOCAL_LLM_GATEWAY_PORT:-8001}"
HEALTH_URL="http://${HOST}:${PORT}/healthz"
PID_FILE="${LOCAL_LLM_GATEWAY_PID_FILE:-.local-llm-gateway.pid}"
LOG_FILE="${LOCAL_LLM_GATEWAY_LOG_FILE:-}"

is_gateway_up() {
  curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null
}

gateway_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true
  elif [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      printf '%s\n' "${pid}"
    fi
  fi
}

stop_gateway() {
  local pids
  pids="$(gateway_pids)"
  if [[ -z "${pids}" ]]; then
    return
  fi

  printf 'Stopping existing local LLM gateway on %s...\n' "${HEALTH_URL}"
  # shellcheck disable=SC2086
  kill ${pids}

  for _ in {1..20}; do
    if ! is_gateway_up; then
      return
    fi
    sleep 0.25
  done

  printf 'Gateway did not stop gracefully; forcing shutdown...\n'
  # shellcheck disable=SC2086
  kill -9 ${pids} 2>/dev/null || true
}

start_gateway() {
  printf 'Starting local LLM gateway on %s...\n' "${HEALTH_URL}"
  if [[ -n "${LOG_FILE}" ]]; then
    uv run uvicorn "lang_graph_state.llm_gateway.server:create_app" --factory --host "${HOST}" --port "${PORT}" 2>&1 | tee "${LOG_FILE}" &
  else
    uv run uvicorn "lang_graph_state.llm_gateway.server:create_app" --factory --host "${HOST}" --port "${PORT}" &
  fi
  local log_pid=$!
  local pid="${log_pid}"
  sleep 0.2
  local detected_pids
  detected_pids="$(gateway_pids)"
  if [[ -n "${detected_pids}" ]]; then
    pid="$(printf '%s\n' "${detected_pids}" | head -n 1)"
  fi
  printf '%s\n' "${pid}" >"${PID_FILE}"

  for _ in {1..60}; do
    if is_gateway_up; then
      if [[ -n "${LOG_FILE}" ]]; then
        printf 'Local LLM gateway is ready. Logs: %s\n' "${LOG_FILE}"
      else
        printf 'Local LLM gateway is ready.\n'
      fi
      return
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      printf 'Local LLM gateway exited while starting.\n'
      if [[ -n "${LOG_FILE}" ]]; then
        printf 'Logs:\n'
        tail -n 40 "${LOG_FILE}" || true
      fi
      exit 1
    fi
    sleep 0.5
  done

  printf 'Timed out waiting for local LLM gateway.\n'
  if [[ -n "${LOG_FILE}" ]]; then
    printf 'Logs:\n'
    tail -n 40 "${LOG_FILE}" || true
  fi
  kill "${log_pid}" 2>/dev/null || true
  exit 1
}

if is_gateway_up; then
  stop_gateway
fi

start_gateway
uv run python -m lang_graph_state.main
