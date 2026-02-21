#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$BASE_DIR/.run"
LOG_DIR="$BASE_DIR/.logs"
ORCH_API_PORT="${ORCH_API_PORT:-8001}"

PID_FILE="$RUN_DIR/orchestrator-api.pid"
LOG_FILE="$LOG_DIR/orchestrator-api.log"

mkdir -p "$RUN_DIR" "$LOG_DIR"

is_pid_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

is_orch_healthy() {
  curl -fsS --max-time 2 "http://127.0.0.1:${ORCH_API_PORT}/health" >/dev/null 2>&1
}

kill_port_if_busy() {
  local port="$1"
  local pids
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "⚠️  Puerto $port ocupado. Cerrando PID(s): $pids"
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

start_api() {
  if [ -f "$PID_FILE" ]; then
    local old_pid
    old_pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$old_pid" ] && is_pid_alive "$old_pid"; then
      echo "ℹ️  Orchestrator API ya está corriendo (PID $old_pid)"
      return 0
    fi
  fi

  kill_port_if_busy "$ORCH_API_PORT"
  : > "$LOG_FILE"

  echo "▶️  Iniciando Orchestrator API..."
  (
    cd "$BASE_DIR"
    nohup bash -c "./start_api.sh" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
  )
  sleep 1

  local new_pid
  new_pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [ -n "$new_pid" ] && is_pid_alive "$new_pid" && is_orch_healthy; then
    echo "✅ Orchestrator API iniciado (PID $new_pid)"
    return 0
  else
    echo "❌ No se pudo iniciar Orchestrator API"
    echo "   Revisa logs: $LOG_FILE"
    tail -n 40 "$LOG_FILE" || true
    return 1
  fi
}

stop_api() {
  if [ ! -f "$PID_FILE" ]; then
    echo "ℹ️  Orchestrator API no tiene PID registrado"
    # PID file can go stale (manual runs / crashes). Ensure the port is freed.
    kill_port_if_busy "$ORCH_API_PORT"
    return
  fi

  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [ -n "$pid" ] && is_pid_alive "$pid"; then
    echo "⏹️  Deteniendo Orchestrator API (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 1
    if is_pid_alive "$pid"; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi

  rm -f "$PID_FILE"
  echo "✅ Orchestrator API detenido"
}

show_status() {
  echo "📊 Estado de servicios"
  if [ ! -f "$PID_FILE" ]; then
    echo "- Orchestrator API: detenido"
    return
  fi

  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [ -n "$pid" ] && is_pid_alive "$pid"; then
    echo "- Orchestrator API: activo (PID $pid)"
  else
    echo "- Orchestrator API: detenido (PID stale)"
  fi
}

show_logs() {
  tail -n 120 "$LOG_FILE"
}

case "${1:-}" in
  up|start)
    if ! start_api; then
      show_status
      exit 1
    fi
    show_status
    ;;
  down|stop)
    stop_api
    show_status
    ;;
  restart)
    stop_api
    if ! start_api; then
      show_status
      exit 1
    fi
    show_status
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs
    ;;
  *)
    echo "Uso:"
    echo "  ./stack.sh up|down|restart|status|logs"
    exit 1
    ;;
esac
