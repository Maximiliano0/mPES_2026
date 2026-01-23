#!/usr/bin/env bash
# ------------------------------------------------------------------
#  Muestra el estado de las optimizaciones Bayesianas en curso.
#
#  Busca procesos de optimización activos (optimize_rl, optimize_dqn,
#  optimize_ac, optimize_tr) y sus watchers asociados.  Para cada
#  paquete detecta:
#    - PID del proceso de optimización y del watcher.
#    - Último trial completado, mejor valor y tiempo transcurrido.
#    - Últimas líneas del log de errores (si las hay).
#
#  Uso:
#    ./utils/optimization_status.sh              # Todos los paquetes
#    ./utils/optimization_status.sh pes_dqn      # Solo un paquete
#    ./utils/optimization_status.sh ac           # Alias aceptados
# ------------------------------------------------------------------
set -uo pipefail

# ── Rutas base ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colores ──────────────────────────────────────────────────────
BOLD='\033[1m'
RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
CYAN='\033[96m'
GRAY='\033[90m'
RESET='\033[0m'

# ── Resolver paquete (opcional) ──────────────────────────────────
resolve_package() {
    case "${1:-}" in
        pes_ql|bayesian|bay|1)              echo "pes_ql"  ;;
        pes_dql|dql|ql|2)                   echo "pes_dql" ;;
        pes_dqn|dqn|3)                      echo "pes_dqn" ;;
        pes_ac|ac|a2c|actor-critic|4)       echo "pes_ac"  ;;
        pes_trf|transformer|tr|5)           echo "pes_trf" ;;
        all|"")                             echo "all"     ;;
        *) echo ""; return 1 ;;
    esac
}

FILTER_PKG="$(resolve_package "${1:-all}")" || {
    echo "Error: Paquete desconocido: '${1:-}'"
    echo "  Opciones: pes_ql, pes_dql, pes_dqn, pes_ac, pes_trf, all"
    exit 1
}

ALL_PACKAGES=(pes_ql pes_dql pes_dqn pes_ac pes_trf)
if [[ "$FILTER_PKG" != "all" ]]; then
    ALL_PACKAGES=("$FILTER_PKG")
fi

# ── Mapeo paquete → módulo de optimización ───────────────────────
get_opt_module() {
    case "$1" in
        pes_ql)  echo "optimize_rl"  ;;
        pes_dql) echo "optimize_rl"  ;;
        pes_dqn) echo "optimize_dqn" ;;
        pes_ac)  echo "optimize_ac"  ;;
        pes_trf) echo "optimize_tr"  ;;
    esac
}

# ── Cabecera ─────────────────────────────────────────────────────
echo ""
echo -e "  ${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "  ${CYAN}║           ESTADO DE OPTIMIZACIONES BAYESIANAS               ║${RESET}"
echo -e "  ${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${GRAY}Fecha: $(date '+%Y-%m-%d %H:%M:%S')    Directorio: $PROJECT_DIR${RESET}"
echo ""

FOUND_ANY=false

for pkg in "${ALL_PACKAGES[@]}"; do
    opt_module="$(get_opt_module "$pkg")"
    log_dir="$PROJECT_DIR/$pkg/inputs"

    # ── Detectar proceso de optimización ─────────────────────────
    opt_pid=""
    watcher_pid=""

    # Buscar PID de optimización (ps aux con grep)
    opt_line=$(ps aux 2>/dev/null | grep "$opt_module" | grep -v grep | head -n1)
    if [[ -n "$opt_line" ]]; then
        opt_pid=$(echo "$opt_line" | awk '{print $2}')
    fi

    # Buscar PID del watcher
    watch_line=$(ps aux 2>/dev/null | grep "watch_and_push.*$pkg" | grep -v grep | head -n1)
    if [[ -n "$watch_line" ]]; then
        watcher_pid=$(echo "$watch_line" | awk '{print $2}')
    fi

    # ── Buscar log files ─────────────────────────────────────────
    log_file=""
    err_file=""
    if [[ -d "$log_dir" ]]; then
        # Más reciente primero, excluir _err
        log_file=$(find "$log_dir" -maxdepth 1 -name 'bayesian_opt*.log' ! -name '*_err*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2-)
        err_file=$(find "$log_dir" -maxdepth 1 -name 'bayesian_opt*_err.log' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n1 | cut -d' ' -f2-)
    fi

    has_process=false
    has_log=false
    [[ -n "$opt_pid" ]] && has_process=true
    [[ -n "$log_file" && -f "$log_file" ]] && has_log=true

    if ! $has_process && ! $has_log; then
        continue
    fi
    FOUND_ANY=true

    # ── Zombie detection helper ──────────────────────────────────
    opt_status="ACTIVO"
    opt_is_zombie=false
    if $has_process; then
        mem_kb=$(ps -p "$opt_pid" -o rss= 2>/dev/null | tr -d ' ')
        cpu_total=$(ps -p "$opt_pid" -o time= 2>/dev/null | tr -d ' ')
        # Parse CPU time (HH:MM:SS or MM:SS) to seconds
        cpu_secs=0
        if [[ -n "$cpu_total" ]]; then
            IFS=: read -ra parts <<< "$cpu_total"
            if [[ ${#parts[@]} -eq 3 ]]; then
                cpu_secs=$(( 10#${parts[0]}*3600 + 10#${parts[1]}*60 + 10#${parts[2]} ))
            elif [[ ${#parts[@]} -eq 2 ]]; then
                cpu_secs=$(( 10#${parts[0]}*60 + 10#${parts[1]} ))
            fi
        fi
        # Zombie heuristic: < 20MB RAM and < 1s CPU after 60s alive
        start_epoch=$(stat -c %Y /proc/"$opt_pid" 2>/dev/null || echo 0)
        now_epoch=$(date +%s)
        age_s=$(( now_epoch - start_epoch ))
        mem_mb_val=0
        [[ -n "$mem_kb" ]] && mem_mb_val=$(awk "BEGIN {printf \"%d\", $mem_kb/1024}")
        if [[ "$mem_mb_val" -lt 20 && "$cpu_secs" -lt 1 && "$age_s" -gt 60 ]]; then
            opt_status="ZOMBIE"
            opt_is_zombie=true
        fi
    fi

    # ── Encabezado del paquete ───────────────────────────────────
    if $has_process && $opt_is_zombie; then
        echo -e "  ${RED}⚠${RESET} ${YELLOW}$pkg${RESET}  ${RED}[$opt_status]${RESET}"
    elif $has_process; then
        echo -e "  ${GREEN}●${RESET} ${YELLOW}$pkg${RESET}  ${GREEN}[EN CURSO]${RESET}"
    else
        echo -e "  ${GRAY}○${RESET} ${YELLOW}$pkg${RESET}  ${GRAY}[FINALIZADO]${RESET}"
    fi
    echo -e "  ${GRAY}$(printf '─%.0s' {1..60})${RESET}"

    # ── PID info ─────────────────────────────────────────────────
    if $has_process; then
        # Obtener CPU time y memoria
        cpu_time=$(ps -p "$opt_pid" -o time= 2>/dev/null | tr -d ' ')
        mem_kb=$(ps -p "$opt_pid" -o rss= 2>/dev/null | tr -d ' ')
        mem_mb=""
        if [[ -n "$mem_kb" ]]; then
            mem_mb=$(awk "BEGIN {printf \"%.1f\", $mem_kb/1024}")
        fi
        if $opt_is_zombie; then
            echo -e "    ${GRAY}PID optimización:${RESET} $opt_pid  ${GRAY}(CPU: ${cpu_time:-?}, Mem: ${mem_mb:-?} MB, Estado: ${RED}$opt_status${RESET}${GRAY})${RESET}"
            echo -e "    ${RED}*** PROCESO ZOMBIE DETECTADO — debe relanzarse ***${RESET}"
        else
            echo -e "    ${GRAY}PID optimización:${RESET} $opt_pid  ${GRAY}(CPU: ${cpu_time:-?}, Mem: ${mem_mb:-?} MB, Estado: ${GREEN}$opt_status${RESET}${GRAY})${RESET}"
        fi
    fi
    if [[ -n "$watcher_pid" ]]; then
        # Watcher zombie check
        w_mem_kb=$(ps -p "$watcher_pid" -o rss= 2>/dev/null | tr -d ' ')
        w_state="ACTIVO"
        if [[ -z "$w_mem_kb" ]]; then
            w_state="NOT FOUND"
            echo -e "    ${GRAY}PID watcher:     ${RESET} $watcher_pid  ${GRAY}(${RED}$w_state${RESET}${GRAY})${RESET}"
        else
            echo -e "    ${GRAY}PID watcher:     ${RESET} $watcher_pid  ${GRAY}(${GREEN}$w_state${RESET}${GRAY})${RESET}"
        fi
    elif $has_process; then
        echo -e "    ${GRAY}PID watcher:      ${YELLOW}no detectado${RESET}"
    fi

    # ── Parsear progreso del log ─────────────────────────────────
    if $has_log; then
        # Target trials
        n_target=$(grep -oP 'Target number of trials:\s*\K\d+' "$log_file" | tail -n1)
        n_target="${n_target:-?}"

        # Último trial completado
        last_trial=$(grep -oP 'Trial\s+\K\d+/\d+\s+\|\s+value=[\d.]+\s+\|\s+best=[\d.]+\s+\|\s+elapsed=\d+s' "$log_file" | tail -n1)

        if [[ -n "$last_trial" ]]; then
            done_total=$(echo "$last_trial" | grep -oP '^\d+/\d+')
            done_n=$(echo "$done_total" | cut -d'/' -f1)
            total_n=$(echo "$done_total" | cut -d'/' -f2)
            value=$(echo "$last_trial" | grep -oP 'value=\K[\d.]+')
            best=$(echo "$last_trial" | grep -oP 'best=\K[\d.]+')
            elapsed=$(echo "$last_trial" | grep -oP 'elapsed=\K\d+')

            pct=$(awk "BEGIN {printf \"%.1f\", ($done_n/$total_n)*100}")

            hrs=$((elapsed / 3600))
            mins=$(( (elapsed % 3600) / 60 ))
            if [[ $hrs -gt 0 ]]; then
                el_str="${hrs}h ${mins}m"
            else
                el_str="${mins}m"
            fi

            # Barra de progreso
            bar_len=30
            filled=$(awk "BEGIN {printf \"%d\", $bar_len * $pct / 100}")
            empty=$((bar_len - filled))
            bar=$(printf '█%.0s' $(seq 1 "$filled" 2>/dev/null) 2>/dev/null)
            bar_empty=$(printf '░%.0s' $(seq 1 "$empty" 2>/dev/null) 2>/dev/null)

            echo ""
            echo -e "    ${GRAY}Progreso: ${RESET} ${GREEN}${bar}${bar_empty}${RESET}  $done_n/$total_n ($pct%)"
            echo -e "    ${GRAY}Mejor valor:     ${RESET} ${CYAN}$best${RESET}"
            echo -e "    ${GRAY}Último valor:    ${RESET} $value"
            echo -e "    ${GRAY}Tiempo:          ${RESET} $el_str (${elapsed}s)"

            # ETA
            if [[ "$done_n" -gt 0 ]]; then
                avg=$(awk "BEGIN {printf \"%.0f\", $elapsed/$done_n}")
                remaining=$(( (total_n - done_n) * avg ))
                rem_hrs=$((remaining / 3600))
                rem_mins=$(( (remaining % 3600) / 60 ))
                if [[ $rem_hrs -gt 0 ]]; then
                    eta_str="~${rem_hrs}h ${rem_mins}m"
                else
                    eta_str="~${rem_mins}m"
                fi
                echo -e "    ${GRAY}ETA restante:    ${RESET} ${YELLOW}$eta_str (~${avg}s/trial)${RESET}"
            fi
        else
            # Sin trials aún
            resume_line=$(grep -E 'Starting fresh|Resuming' "$log_file" | tail -n1)
            echo ""
            if [[ -n "$resume_line" ]]; then
                echo -e "    ${GRAY}Progreso: ${RESET} $(echo "$resume_line" | sed 's/^[[:space:]]*//')"
            else
                echo -e "    ${GRAY}Progreso: ${RESET} ${YELLOW}Inicializando... (trial #1 en curso)${RESET}"
            fi
            echo -e "    ${GRAY}Target trials:   ${RESET} $n_target"
        fi

        # ── Log path ────────────────────────────────────────────
        rel_log="${log_file#$PROJECT_DIR/}"
        echo -e "    ${GRAY}Log:              $rel_log${RESET}"
    fi

    # ── Errores ──────────────────────────────────────────────────
    if [[ -n "$err_file" && -f "$err_file" ]]; then
        err_count=$(grep -c '\S' "$err_file" 2>/dev/null)
        if [[ "$err_count" -gt 0 ]]; then
            echo -e "    ${GRAY}Errores/Warnings:${RESET} ${YELLOW}$err_count línea(s) en stderr${RESET}"
            # Últimas 3 líneas no vacías
            grep '\S' "$err_file" | tail -n3 | while IFS= read -r line; do
                trimmed="${line:0:80}"
                [[ ${#line} -gt 80 ]] && trimmed="${trimmed}..."
                echo -e "      ${YELLOW}$trimmed${RESET}"
            done
        fi
    fi

    echo ""
done

if ! $FOUND_ANY; then
    echo -e "  ${YELLOW}No se detectaron optimizaciones activas ni logs recientes.${RESET}"
    echo ""
fi

# ── Comandos útiles ──────────────────────────────────────────────
echo -e "  ${GRAY}────────────────────────────────────────────────────────────${RESET}"
echo -e "  ${GRAY}Comandos útiles:${RESET}"
echo -e "  ${GRAY}  Tiempo real:  tail -f <pkg>/inputs/bayesian_opt.log${RESET}"
echo -e "  ${GRAY}  Vivo?:        ps -p <PID> -o pid,comm,time,rss${RESET}"
echo -e "  ${GRAY}  Errores:      tail -20 <pkg>/inputs/bayesian_opt_err.log${RESET}"
echo ""
