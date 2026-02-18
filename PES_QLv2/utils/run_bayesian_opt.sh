#!/usr/bin/env bash
# ------------------------------------------------------------------
#  Lanzar optimización Bayesiana de Q-Learning 
#
#  Uso:
#    chmod +x run_bayesian_opt.sh
#    ./run_bayesian_opt.sh <n_trials>              # corrida nueva
#    ./run_bayesian_opt.sh <n_trials> 2026-02-12   # reanudar desde fecha indicada
# ------------------------------------------------------------------
set -euo pipefail

# Verificar que se pasó el número de trials
if [[ $# -lt 1 ]]; then
    echo "Error: Debe especificar el número de trials"
    echo "Uso: $0 <n_trials> [fecha_resume]"
    echo "Ejemplo: $0 100"
    echo "Ejemplo: $0 100 2026-02-12"
    exit 1
fi

N_TRIALS="$1"
PROJECT_DIR="/home/mecatronica/Documentos/maximiliano/mPES"
VENV="$PROJECT_DIR/linux_mpes_env/bin/activate"
LOG_DIR="$PROJECT_DIR/PES_Bayesian/inputs"

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
nohup python3 -m PES_Bayesian.ext.optimize_rl $ARGS > "$LOGFILE" 2>&1 &
OPT_PID=$!
echo "Optimización lanzada  PID=$OPT_PID  trials=$N_TRIALS"
echo "Log: $LOGFILE"

# Inhibir suspensión mientras el proceso esté vivo
nohup systemd-inhibit \
    --what=idle:sleep:handle-lid-switch \
    --who="mPES Bayesian Optimization" \
    --why="Running $N_TRIALS-trial Bayesian optimization" \
    --mode=block \
    tail --pid=$OPT_PID -f /dev/null > /dev/null 2>&1 &
echo "Inhibidor de suspensión activo (se desactiva al terminar)"

echo ""
echo "Comandos útiles:"
echo "  Progreso:    grep 'Trial' $LOGFILE | tail -10"
echo "  Tiempo real: tail -f $LOGFILE"
echo "  Vivo?:       pgrep -f 'PES_Bayesian.ext.optimize_rl' -a"