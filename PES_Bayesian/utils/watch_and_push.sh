#!/usr/bin/env bash
# ------------------------------------------------------------------
#  Vigila las optimizaciones bayesianas activas.
#  Cuando cada una termina, hace git add + commit + push a rl_optim.
#
#  Todas las rutas se resuelven de forma relativa a la ubicación
#  de este script (utils/ -> PES_QLv2/ -> mPES/).
#
#  Lógica:
#    - Recibe uno o más PIDs como argumentos.
#    - Cada 30 segundos comprueba si cada PID sigue vivo (kill -0).
#    - Cuando un PID termina, ejecuta git add -A, commit y push
#      a la rama configurada (por defecto: rl_optim).
#    - Sale cuando todos los PIDs han terminado.
#
#  Uso:
#    nohup ./watch_and_push.sh <pid1> [pid2] ... &
# ------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$PKG_DIR/.." && pwd)"
BRANCH="rl_optim"

# PIDs a vigilar (recibidos como argumentos)
if [[ $# -lt 1 ]]; then
    echo "Uso: $0 <pid1> [pid2] ..."
    exit 1
fi

PIDS=("$@")
echo "[watch_and_push] Vigilando PIDs: ${PIDS[*]}"
echo "[watch_and_push] Proyecto: $PROJECT_DIR"
echo "[watch_and_push] Rama: $BRANCH"

do_commit_push() {
    local pid="$1"
    local label="$2"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M')"

    echo "[watch_and_push] [$ts] PID $pid ($label) terminó. Realizando commit + push..."

    cd "$PROJECT_DIR" || return 1

    # Asegurar que estamos en la rama correcta
    current_branch="$(git branch --show-current)"
    if [[ "$current_branch" != "$BRANCH" ]]; then
        echo "[watch_and_push] WARN: Rama actual es '$current_branch', cambiando a '$BRANCH'"
        git checkout "$BRANCH" 2>/dev/null || {
            echo "[watch_and_push] ERROR: No se pudo cambiar a $BRANCH"
            return 1
        }
    fi

    git add -A
    git commit -m "auto: Optimización $label completada ($ts)" || {
        echo "[watch_and_push] Sin cambios para commit (PID $pid)"
        return 0
    }
    git push origin "$BRANCH" 2>&1
    echo "[watch_and_push] [$ts] Push completado para $label"
}

# Obtener el comando de cada PID para usarlo como label
declare -A PID_LABELS
for pid in "${PIDS[@]}"; do
    label="$(ps -o args= -p "$pid" 2>/dev/null | sed 's/python3 -m //' || echo "PID_$pid")"
    PID_LABELS[$pid]="$label"
    echo "[watch_and_push] PID $pid -> $label"
done

# Vigilar cada PID
declare -A DONE
for pid in "${PIDS[@]}"; do
    DONE[$pid]=0
done

while true; do
    all_done=1
    for pid in "${PIDS[@]}"; do
        [[ ${DONE[$pid]} -eq 1 ]] && continue

        if ! kill -0 "$pid" 2>/dev/null; then
            # El proceso terminó
            do_commit_push "$pid" "${PID_LABELS[$pid]}"
            DONE[$pid]=1
        else
            all_done=0
        fi
    done

    if [[ $all_done -eq 1 ]]; then
        echo "[watch_and_push] Todas las optimizaciones terminaron. Saliendo."
        break
    fi

    sleep 30
done
