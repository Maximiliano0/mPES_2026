#!/usr/bin/env bash
# ------------------------------------------------------------------
#  Lanzar optimización Bayesiana de Q-Learning
#
#  Script compartido para pes_base_line y pes_qlv2.
#  Todas las rutas se resuelven de forma relativa a la ubicación
#  de este script (utils/ → mPES/).
#
#  Funcionalidades:
#    - Lanza la optimización en segundo plano con nohup.
#    - Inhibe la suspensión del sistema (systemd-inhibit) mientras
#      el proceso esté vivo.
#    - Configura GNOME para ignorar el cierre de tapa.
#
#  Uso:
#    chmod +x run_bayesian_opt.sh
#    ./run_bayesian_opt.sh bayesian 100              # pes_base_line, corrida nueva
#    ./run_bayesian_opt.sh qlv2 100                  # pes_qlv2, corrida nueva
#    ./run_bayesian_opt.sh bayesian 100 2026-02-12   # reanudar desde fecha
# ------------------------------------------------------------------
set -euo pipefail

# ── Rutas base ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"       # mPES/ (raíz del workspace)
VENV="$PROJECT_DIR/linux_mpes_env/bin/activate"

# ── Resolver paquete desde primer argumento ──────────────────────
resolve_package() {
    case "${1:-}" in
        bayesian|Bayesian|BAYESIAN|bay|1) echo "pes_base_line" ;;
        qlv2|QLv2|QLVAL2|ql|2)           echo "pes_qlv2"     ;;
        *) return 1 ;;
    esac
}

# ── Verificar argumentos ────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Error: Debe especificar el paquete y el número de trials"
    echo ""
    echo "Uso: $0 <paquete> <n_trials> [fecha_resume]"
    echo ""
    echo "  Paquetes: bayesian (pes_base_line), qlv2 (pes_qlv2)"
    echo ""
    echo "Ejemplos:"
    echo "  $0 bayesian 100"
    echo "  $0 qlv2 100"
    echo "  $0 bayesian 100 2026-02-12"
    exit 1
fi

PKG_NAME="$(resolve_package "$1")" || {
    echo "Error: Paquete desconocido: '$1'"
    echo "  Opciones válidas: bayesian, qlv2"
    exit 1
}
N_TRIALS="$2"

LOG_DIR="$PROJECT_DIR/$PKG_NAME/inputs"

cd "$PROJECT_DIR"
source "$VENV"

# Construir argumentos (--resume si se pasa una fecha)
ARGS="$N_TRIALS"
LOG_SUFFIX=""
if [[ ${3:-} != "" ]]; then
    ARGS="$N_TRIALS --resume $3"
    LOG_SUFFIX="_resume_$3"
fi

LOGFILE="$LOG_DIR/bayesian_opt${LOG_SUFFIX}.log"

# Evitar suspensión por GNOME al cerrar la tapa
gsettings set org.gnome.settings-daemon.plugins.power lid-close-ac-action 'nothing'

# Lanzar optimización en segundo plano
nohup python3 -m "${PKG_NAME}.ext.optimize_rl" $ARGS > "$LOGFILE" 2>&1 &
OPT_PID=$!
echo "Optimización lanzada  PID=$OPT_PID  trials=$N_TRIALS"
echo "Log: $LOGFILE"

# Inhibir suspensión mientras el proceso esté vivo
nohup systemd-inhibit \
    --what=idle:sleep:handle-lid-switch \
    --who="mPES Bayesian Optimization ($PKG_NAME)" \
    --why="Running $N_TRIALS-trial Bayesian optimization" \
    --mode=block \
    tail --pid=$OPT_PID -f /dev/null > /dev/null 2>&1 &
echo "Inhibidor de suspensión activo (se desactiva al terminar)"

echo ""
echo "Comandos útiles:"
echo "  Progreso:    grep 'Trial' $LOGFILE | tail -10"
echo "  Tiempo real: tail -f $LOGFILE"
echo "  Vivo?:       pgrep -f '${PKG_NAME}.ext.optimize_rl' -a"