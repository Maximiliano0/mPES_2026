#!/usr/bin/env bash
# ------------------------------------------------------------------
#  Vigila las optimizaciones bayesianas activas.
#  Cuando cada una termina, hace git add + commit + push a la rama actual.
#
#  Script compartido para pes_ql, pes_dql, pes_dqn,
#  pes_ac y pes_trf.  Todas las rutas se resuelven de forma
#  relativa a la ubicación de este script (utils/ → mPES/).
#
#  Lógica:
#    - Recibe el nombre del paquete y uno o más PIDs como argumentos.
#    - Cada 30 segundos comprueba si cada PID sigue vivo (kill -0).
#    - Cuando un PID termina, ejecuta git add -A, commit y push
#      a la rama actual de Git.
#    - Envía una notificación push tras el push (o si hay error).
#    - Sale cuando todos los PIDs han terminado.
#
#  Uso:
#    nohup ./watch_and_push.sh bayesian <pid1> [pid2] ... &
#    nohup ./watch_and_push.sh dql <pid1> [pid2] ... &
# ------------------------------------------------------------------
set -uo pipefail

# ── Rutas base ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"       # mPES/ (raíz del workspace)
NOTIFY="$SCRIPT_DIR/notify.py"                     # notify.py vive en utils/
VENV="$PROJECT_DIR/linux_mpes_env/bin/activate"

source "$VENV"

# ── Resolver paquete desde primer argumento ──────────────────────
resolve_package() {
    case "${1:-}" in
        pes_ql|bayesian|Bayesian|BAYESIAN|bay|1)  echo "pes_ql"        ;;
        pes_dql|dql|DQL|ql|2)                     echo "pes_dql"       ;;
        pes_dqn|dqn|DQN|3)                        echo "pes_dqn"             ;;
        pes_ac|ac|a2c|actor-critic|4)             echo "pes_ac"    ;;
        pes_trf|transformer|tr|5)                 echo "pes_trf"     ;;
        *) return 1 ;;
    esac
}

# ── Verificar argumentos ────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Uso: $0 <paquete> <pid1> [pid2] ..."
    echo ""
    echo "  Paquetes: bayesian (pes_ql), dql (pes_dql), dqn (pes_dqn), ac (pes_ac), transformer (pes_trf)"
    echo ""
    echo "Ejemplo: nohup $0 bayesian 12345 &"
    exit 1
fi

PKG_NAME="$(resolve_package "$1")" || {
    echo "Error: Paquete desconocido: '$1'"
    echo "  Opciones válidas: bayesian, dql, dqn, ac, transformer (o nombre completo del paquete)"
    exit 1
}
shift  # Quitar el primer argumento (paquete); quedan solo PIDs

# Rama actual de Git (se detecta automáticamente)
BRANCH="$(cd "$PROJECT_DIR" && git branch --show-current)"

PIDS=("$@")
echo "[watch_and_push] Paquete: $PKG_NAME"
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

    git add -A
    git commit -m "auto: Optimización $label completada ($ts)" || {
        echo "[watch_and_push] Sin cambios para commit (PID $pid)"
        return 0
    }
    if git push origin "$BRANCH" 2>&1; then
        echo "[watch_and_push] [$ts] Push completado para $label"
        BODY=$(printf "La optimización '%s' (PID %s) terminó correctamente.\nCommit y push a rama '%s' completados (%s).\nProyecto: %s" \
            "$label" "$pid" "$BRANCH" "$ts" "$PROJECT_DIR")
        python3 "$NOTIFY" \
            "[$PKG_NAME] Optimizacion completada - push realizado" \
            "$BODY" || true
    else
        echo "[watch_and_push] [$ts] ERROR en push para $label"
        BODY=$(printf "Error al hacer push de la optimización '%s' (PID %s).\nRama: %s\nTimestamp: %s\nProyecto: %s" \
            "$label" "$pid" "$BRANCH" "$ts" "$PROJECT_DIR")
        python3 "$NOTIFY" \
            "[$PKG_NAME] ERROR en git push" \
            "$BODY" || true
    fi
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
