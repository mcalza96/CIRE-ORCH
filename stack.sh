#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$BASE_DIR/.run"
LOG_DIR="$BASE_DIR/.logs"

PID_FILE="$RUN_DIR/orchestrator-api.pid"
LOG_FILE="$LOG_DIR/orchestrator-api.log"

mkdir -p "$RUN_DIR" "$LOG_DIR"

is_pid_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
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
      return
    fi
  fi

  kill_port_if_busy 8001
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
  if [ -n "$new_pid" ] && is_pid_alive "$new_pid"; then
    echo "✅ Orchestrator API iniciado (PID $new_pid)"
  else
    echo "❌ No se pudo iniciar Orchestrator API"
    echo "   Revisa logs: $LOG_FILE"
  fi
}

stop_api() {
  if [ ! -f "$PID_FILE" ]; then
    echo "ℹ️  Orchestrator API no tiene PID registrado"
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
    start_api
    show_status
    ;;
  down|stop)
    stop_api
    show_status
    ;;
  restart)
    stop_api
    start_api
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
