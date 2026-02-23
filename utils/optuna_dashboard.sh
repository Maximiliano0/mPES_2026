#!/usr/bin/env bash
# ==============================================================================
# optuna_dashboard.sh — Lanza Optuna Dashboard para PES_Bayesian o PES_QLv2
#
# Uso:
#   ./optuna_dashboard.sh                   # Menú interactivo
#   ./optuna_dashboard.sh bayesian          # Directo a PES_Bayesian
#   ./optuna_dashboard.sh qlv2              # Directo a PES_QLv2
#   ./optuna_dashboard.sh bayesian 9090     # PES_Bayesian en puerto 9090
#
# Requisitos:
#   - Entorno virtual activado (linux_mpes_env)
#   - optuna-dashboard instalado (pip install optuna-dashboard)
# ==============================================================================

set -euo pipefail

# ── Rutas base ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"      # mPES/ (raíz del workspace)

# ── Puerto por defecto ───────────────────────────────────────────────────────
DEFAULT_PORT=8080

# ── Rutas relativas a los directorios de inputs de cada proyecto ─────────────
BAYESIAN_INPUTS="PES_Bayesian/inputs"
QLV2_INPUTS="PES_QLv2/inputs"

# ── Colores para la salida en terminal ───────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

# ==============================================================================
# Funciones auxiliares
# ==============================================================================

# Imprime un mensaje de error y sale
die() {
    echo -e "${RED}❌ Error: $1${RESET}" >&2
    exit 1
}

# Busca el archivo .db más reciente en un directorio de inputs.
# Los estudios se guardan como: <date>_BAYESIAN_OPT/optuna_study_<date>.db
# Esta función encuentra el más reciente por orden alfabético (YYYY-MM-DD).
find_latest_db() {
    local inputs_dir="$1"
    local db_path

    # Buscar archivos .db dentro de subdirectorios *_BAYESIAN_OPT/
    # Ordenar por nombre (las fechas YYYY-MM-DD se ordenan alfabéticamente)
    # y tomar el último (más reciente)
    db_path=$(find "${inputs_dir}" -path "*_BAYESIAN_OPT/optuna_study_*.db" -type f 2>/dev/null \
              | sort | tail -n 1)

    echo "${db_path}"
}

# Verifica que optuna-dashboard esté disponible
check_dependencies() {
    if ! command -v optuna-dashboard &>/dev/null; then
        die "optuna-dashboard no encontrado.\n   Instálalo con: pip install optuna-dashboard"
    fi
}

# Lanza el dashboard para un archivo .db dado
launch_dashboard() {
    local db_path="$1"
    local port="${2:-${DEFAULT_PORT}}"

    # Verificar que el archivo existe
    [[ -f "${db_path}" ]] || die "No se encontró la base de datos: ${db_path}"

    # Construir la URI de SQLite (ruta absoluta con triple /)
    local abs_path
    abs_path="$(cd "$(dirname "${db_path}")" && pwd)/$(basename "${db_path}")"
    local sqlite_uri="sqlite:///${abs_path}"

    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
    echo -e "  ${GREEN}Optuna Dashboard${RESET}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
    echo -e "  ${BLUE}Base de datos:${RESET}  ${db_path}"
    echo -e "  ${BLUE}Puerto:${RESET}         ${port}"
    echo -e "  ${BLUE}URL:${RESET}            ${GREEN}http://localhost:${port}${RESET}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
    echo -e "  Presiona ${YELLOW}Ctrl+C${RESET} para detener el servidor."
    echo ""

    # Lanzar optuna-dashboard
    optuna-dashboard "${sqlite_uri}" --port "${port}"
}

# ==============================================================================
# Menú interactivo
# ==============================================================================
show_menu() {
    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
    echo -e "  ${GREEN}Optuna Dashboard Launcher${RESET}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
    echo ""

    # Detectar estudios disponibles
    local bayesian_db qlv2_db
    bayesian_db=$(find_latest_db "${PROJECT_DIR}/${BAYESIAN_INPUTS}")
    qlv2_db=$(find_latest_db "${PROJECT_DIR}/${QLV2_INPUTS}")

    # Mostrar opciones con estado
    if [[ -n "${bayesian_db}" ]]; then
        echo -e "  ${GREEN}1)${RESET} PES_Bayesian   ${BLUE}→${RESET} $(basename "${bayesian_db}")"
    else
        echo -e "  ${RED}1)${RESET} PES_Bayesian   ${RED}(sin estudios)${RESET}"
    fi

    if [[ -n "${qlv2_db}" ]]; then
        echo -e "  ${GREEN}2)${RESET} PES_QLv2       ${BLUE}→${RESET} $(basename "${qlv2_db}")"
    else
        echo -e "  ${RED}2)${RESET} PES_QLv2       ${RED}(sin estudios)${RESET}"
    fi

    echo -e "  ${YELLOW}q)${RESET} Salir"
    echo ""
    read -rp "  Selección [1/2/q]: " choice

    case "${choice}" in
        1)
            [[ -n "${bayesian_db}" ]] || die "No se encontró ningún estudio en ${BAYESIAN_INPUTS}"
            launch_dashboard "${bayesian_db}"
            ;;
        2)
            [[ -n "${qlv2_db}" ]] || die "No se encontró ningún estudio en ${QLV2_INPUTS}"
            launch_dashboard "${qlv2_db}"
            ;;
        q|Q)
            echo -e "  ${BLUE}Hasta luego.${RESET}"
            exit 0
            ;;
        *)
            die "Opción inválida: ${choice}"
            ;;
    esac
}

# ==============================================================================
# Punto de entrada
# ==============================================================================

# Verificar dependencias
check_dependencies

# Procesar argumentos
PROJECT="${1:-}"
PORT="${2:-${DEFAULT_PORT}}"

case "${PROJECT}" in
    # Argumento directo: bayesian
    bayesian|Bayesian|BAYESIAN|bay|1)
        db=$(find_latest_db "${PROJECT_DIR}/${BAYESIAN_INPUTS}")
        [[ -n "${db}" ]] || die "No se encontró ningún estudio en ${BAYESIAN_INPUTS}"
        launch_dashboard "${db}" "${PORT}"
        ;;
    # Argumento directo: qlv2
    qlv2|QLv2|QLVAL2|ql|2)
        db=$(find_latest_db "${PROJECT_DIR}/${QLV2_INPUTS}")
        [[ -n "${db}" ]] || die "No se encontró ningún estudio en ${QLV2_INPUTS}"
        launch_dashboard "${db}" "${PORT}"
        ;;
    # Sin argumentos: menú interactivo
    "")
        show_menu
        ;;
    *)
        die "Proyecto desconocido: '${PROJECT}'\n   Uso: $0 [bayesian|qlv2] [puerto]"
        ;;
esac
