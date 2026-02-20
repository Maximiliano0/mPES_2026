#!/usr/bin/env bash
# ------------------------------------------------------------------
#  Lanzar optimización Bayesiana de Q-Learning (PES_QLv2)
#
#  Todas las rutas se resuelven de forma relativa a la ubicación
#  de este script (utils/ -> PES_QLv2/ -> mPES/).  El nombre del
#  paquete (PKG_NAME) se deriva automáticamente del directorio padre.
#
#  Funcionalidades:
#    - Lanza la optimización en segundo plano con nohup.
#    - Inhibe la suspensión del sistema (systemd-inhibit) mientras
#      el proceso esté vivo.
#    - Configura GNOME para ignorar el cierre de tapa.
#
#  Uso:
#    chmod +x run_bayesian_opt.sh
#    ./run_bayesian_opt.sh <n_trials>              # corrida nueva
#    ./run_bayesian_opt.sh <n_trials> 2026-02-12   # reanudar desde fecha indicada
# ------------------------------------------------------------------
set -euo pipefail

# Determinar rutas relativas al script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # PES_QLv2/
PROJECT_DIR="$(cd "$PKG_DIR/.." && pwd)"        # mPES/
PKG_NAME="$(basename "$PKG_DIR")"               # PES_QLv2

VENV="$PROJECT_DIR/linux_mpes_env/bin/activate"
LOG_DIR="$PKG_DIR/inputs"

# Verificar que se pasó el número de trials
if [[ $# -lt 1 ]]; then
    echo "Error: Debe especificar el número de trials"
    echo "Uso: $0 <n_trials> [fecha_resume]"
    echo "Ejemplo: $0 100"
    echo "Ejemplo: $0 100 2026-02-12"
    exit 1
fi

N_TRIALS="$1"

cd "$PROJECT_DIR"
source "$VENV"

# Construir argumentos (--resume si se pasa una fecha)
ARGS="$N_TRIALS"
LOG_SUFFIX=""
if [[ ${2:-} != "" ]]; then
    ARGS="$N_TRIALS --resume $2"
    LOG_SUFFIX="_resume_$2"
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