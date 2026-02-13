#!/usr/bin/env bash
# ------------------------------------------------------------------
#  Lanzar optimización Bayesiana de Q-Learning (200 trials)
#
#  Uso:
#    chmod +x run_bayesian_opt.sh
#    ./run_bayesian_opt.sh              # corrida nueva
#    ./run_bayesian_opt.sh 2026-02-12   # reanudar desde fecha indicada
# ------------------------------------------------------------------
set -euo pipefail

N_TRIALS=200
PROJECT_DIR="/home/mecatronica/Documentos/maximiliano/mPES"
VENV="$PROJECT_DIR/linux_mpes_env/bin/activate"
LOG_DIR="$PROJECT_DIR/PES_Bayesian/inputs"

cd "$PROJECT_DIR"
source "$VENV"

# Construir argumentos (--resume si se pasa una fecha)
ARGS="$N_TRIALS"
LOG_SUFFIX=""
if [[ ${1:-} != "" ]]; then
    ARGS="$N_TRIALS --resume $1"
    LOG_SUFFIX="_resume_$1"
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
