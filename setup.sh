#!/usr/bin/env bash
# =============================================================================
# setup.sh — Comprobación e instalación de dependencias EXTERNAS a Python
#            para el proyecto Overcooked-AI (ver PLAN.md)
#
# Uso:
#   ./setup.sh --check     Solo diagnostica: reporta qué falta, no instala nada
#   ./setup.sh             Comprueba e instala lo que falte (micromamba + entorno)
#   ./setup.sh --cpu       Igual, pero fuerza torch solo-CPU en el entorno
#
# Qué cubre (y qué no):
#   ✔ Herramientas de sistema: git, curl, unzip, bzip2, tar
#   ✔ micromamba (instalación en espacio de usuario, sin sudo)
#   ✔ Creación del entorno desde environment.yml
#   ✔ Detección de GPU NVIDIA (informativa; el proyecto funciona en CPU)
#   ✔ Detección de Colab/Kaggle → redirige a scripts/setup_colab.sh (allí NO se usa micromamba)
#   ✔ Librerías de sistema para el rendering con pygame (solo aviso; opcional)
#   ✘ NO instala drivers NVIDIA/CUDA del sistema (torch de pip trae sus propias libs CUDA)
#   ✘ NO usa sudo salvo para paquetes apt opcionales de rendering (pregunta antes)
#
# Códigos de salida: 0 = todo listo | 1 = falta algo (en --check) o falló instalación
# =============================================================================
set -euo pipefail

# --------------------------- utilidades de salida ---------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
warn() { echo -e "${YELLOW}[AVISO]${NC} $1"; }
fail() { echo -e "${RED}[FALTA]${NC} $1"; MISSING=1; }

MISSING=0
CHECK_ONLY=0
FORCE_CPU=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --cpu)   FORCE_CPU=1 ;;
    *) echo "Argumento desconocido: $arg (usa --check y/o --cpu)"; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="overcooked-agent"
ENV_YML="${REPO_ROOT}/environment.yml"

echo "======================================================================"
echo " Overcooked-AI — setup de dependencias externas ($( [ $CHECK_ONLY -eq 1 ] && echo 'modo CHECK' || echo 'modo INSTALL'))"
echo "======================================================================"

# --------------------------- 0. Detección de plataforma ---------------------
OS="$(uname -s || echo desconocido)"
ARCH="$(uname -m || echo desconocido)"
IS_WSL=0; grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1

if [ -d /content ] && [ -n "${COLAB_RELEASE_TAG:-}" ] || python3 -c "import google.colab" 2>/dev/null; then
  warn "Entorno Google Colab detectado."
  warn "En Colab NO se usa micromamba: ejecuta scripts/setup_colab.sh y REINICIA el runtime."
  exit 0
fi
if [ -d /kaggle ]; then
  warn "Entorno Kaggle detectado: usa scripts/setup_colab.sh (misma vertiente que Colab)."
  exit 0
fi

case "$OS" in
  Linux)  ok "SO: Linux ($ARCH)$( [ $IS_WSL -eq 1 ] && echo ' [WSL]')" ;;
  Darwin) ok "SO: macOS ($ARCH)" ;;
  *)      fail "SO no soportado por este script: $OS. En Windows nativo usa WSL2 (el timeout del profesor usa SIGALRM, solo-Unix)."; exit 1 ;;
esac
[ $IS_WSL -eq 1 ] && warn "WSL: entrenar aquí está bien; el autotest de timeouts (Etapa 7) también funciona (WSL es Unix)."

# --------------------------- 1. Herramientas de sistema ---------------------
echo "--- [1/5] Herramientas de sistema ---"
NEED_TOOLS=()
for tool in git curl tar bzip2 unzip; do
  if command -v "$tool" >/dev/null 2>&1; then
    ok "$tool $(command -v "$tool")"
  else
    fail "$tool no encontrado"
    NEED_TOOLS+=("$tool")
  fi
done

if [ ${#NEED_TOOLS[@]} -gt 0 ] && [ $CHECK_ONLY -eq 0 ]; then
  if [ "$OS" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
    echo ">> Instalando con apt: ${NEED_TOOLS[*]} (requiere sudo)"
    sudo apt-get update -qq && sudo apt-get install -y -qq "${NEED_TOOLS[@]}"
  elif [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    echo ">> Instalando con brew: ${NEED_TOOLS[*]}"
    brew install "${NEED_TOOLS[@]}"
  else
    fail "Instala manualmente: ${NEED_TOOLS[*]} (no se detectó apt/brew)"
    exit 1
  fi
fi

# --------------------------- 2. GPU (informativo) ----------------------------
echo "--- [2/5] GPU ---"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  ok "GPU NVIDIA detectada: ${GPU_NAME}. torch de pip usará CUDA sin instalar nada más."
else
  warn "Sin GPU NVIDIA visible. El proyecto FUNCIONA en CPU (entrenamiento más lento, ~2-4x)."
  warn "El entregable SIEMPRE infiere en CPU, así que esto no afecta la entrega."
  [ $FORCE_CPU -eq 0 ] && warn "Sugerencia: relanza con --cpu para un torch más liviano (sin libs CUDA)."
fi

# --------------------------- 3. micromamba -----------------------------------
echo "--- [3/5] micromamba ---"
MAMBA_BIN=""
if command -v micromamba >/dev/null 2>&1; then
  MAMBA_BIN="$(command -v micromamba)"
  ok "micromamba ya instalado: $MAMBA_BIN ($(micromamba --version 2>/dev/null))"
elif [ -x "${HOME}/.local/bin/micromamba" ]; then
  MAMBA_BIN="${HOME}/.local/bin/micromamba"
  ok "micromamba encontrado en ~/.local/bin"
else
  if [ $CHECK_ONLY -eq 1 ]; then
    fail "micromamba no instalado (el modo install lo instalará sin sudo en ~/.local/bin)"
  else
    echo ">> Instalando micromamba en espacio de usuario (sin sudo)..."
    # Instalador oficial; coloca el binario en ~/.local/bin y NO toca el sistema
    "${SHELL:-bash}" <(curl -Ls https://micro.mamba.pm/install.sh) < /dev/null
    export PATH="${HOME}/.local/bin:${PATH}"
    MAMBA_BIN="$(command -v micromamba || echo "${HOME}/.local/bin/micromamba")"
    [ -x "$MAMBA_BIN" ] && ok "micromamba instalado: $MAMBA_BIN" || { fail "instalación de micromamba falló"; exit 1; }
  fi
fi

# --------------------------- 4. Entorno del proyecto -------------------------
echo "--- [4/5] Entorno '${ENV_NAME}' desde environment.yml ---"
if [ ! -f "$ENV_YML" ]; then
  fail "No existe ${ENV_YML}. Este script debe correr desde la raíz del repo (junto a environment.yml)."
  exit 1
fi
ok "environment.yml presente"

if [ -n "$MAMBA_BIN" ] && "$MAMBA_BIN" env list 2>/dev/null | grep -q "$ENV_NAME"; then
  ok "El entorno '${ENV_NAME}' ya existe"
elif [ $CHECK_ONLY -eq 1 ]; then
  [ -n "$MAMBA_BIN" ] && fail "El entorno '${ENV_NAME}' no existe aún (el modo install lo creará)"
else
  echo ">> Creando entorno '${ENV_NAME}' (esto tarda unos minutos)..."
  if [ $FORCE_CPU -eq 1 ]; then
    # Variante solo-CPU: crea el env sin torch y lo instala del índice CPU después
    "$MAMBA_BIN" create -f "$ENV_YML" -y
    "$MAMBA_BIN" run -n "$ENV_NAME" pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
  else
    "$MAMBA_BIN" create -f "$ENV_YML" -y
  fi
  ok "Entorno creado"
fi

# --------------------------- 5. Rendering opcional (pygame ventana) ----------
echo "--- [5/5] Rendering opcional (solo para VER partidas con play.yaml) ---"
# Los wheels de pygame traen SDL2 embebido; en Linux de escritorio suele bastar.
# En servidores/WSL sin X11 el rendering 'window' no funcionará: usar rendering: none o save_gif.
if [ "$OS" = "Linux" ]; then
  if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    ok "Sesión gráfica detectada; el rendering en ventana debería funcionar."
  else
    warn "Sin DISPLAY: entorno headless. Entrenar/evaluar funciona igual (rendering: none)."
    warn "Para GIFs de debug usa save_gif: true en los configs de la plantilla."
  fi
fi

# --------------------------- Veredicto --------------------------------------
echo "======================================================================"
if [ $MISSING -eq 1 ]; then
  echo -e "${RED}Resultado: faltan dependencias (ver [FALTA] arriba).${NC}"
  [ $CHECK_ONLY -eq 1 ] && echo "Corre ./setup.sh (sin --check) para instalarlas."
  exit 1
fi
echo -e "${GREEN}Resultado: dependencias externas listas.${NC}"
echo "Siguientes pasos:"
echo "  micromamba activate ${ENV_NAME}"
echo "  pytest tests/test_env_smoke.py     # GATE 0 del PLAN.md"
exit 0
