#!/usr/bin/env bash
# =============================================================================
# setup.sh — Descarga todo lo necesario para empezar el proyecto (ver PLANEACION.md)
#
#   1. Papers de referencia            -> docs/*.pdf
#   2. Repos externos (layouts, refs)  -> external/
#   3. Entorno micromamba "overcooked" (environment.yml) + extras de RL
#   4. Verificación final de imports y datos
#
# Uso:
#   ./setup.sh                 # todo
#   SKIP_ENV=1 ./setup.sh      # solo papers + repos (sin tocar el entorno)
#   TORCH_FLAVOR=cu121 ./setup.sh   # en la máquina con GPU NVIDIA (default: auto)
#
# Idempotente: lo ya descargado/instalado se omite.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS="$ROOT/docs"
EXT="$ROOT/external"
ENV_NAME="overcooked"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*"; }

# -----------------------------------------------------------------------------
# 1. Papers -> docs/
# -----------------------------------------------------------------------------
log "1/4 Papers de referencia -> docs/"
mkdir -p "$DOCS"

download_pdf() { # nombre_destino url
    local dest="$DOCS/$1" url="$2"
    if [ -s "$dest" ] && file "$dest" | grep -q "PDF"; then
        log "  ya existe: $1"
        return 0
    fi
    log "  bajando:   $1"
    if curl -fsSL -A "Mozilla/5.0" -o "$dest" "$url" && file "$dest" | grep -q "PDF"; then
        return 0
    fi
    warn "  FALLO: $1 ($url) — bórralo y reintenta, o descárgalo a mano"
    rm -f "$dest"
}

download_pdf 2019_carroll_utility_of_learning_about_humans.pdf "https://arxiv.org/pdf/1910.05789"
download_pdf 2021_knott_robustness_collaborative_agents.pdf    "https://arxiv.org/pdf/2101.05507"
download_pdf 2021_strouse_fcp.pdf                              "https://arxiv.org/pdf/2110.08176"
download_pdf 2022_zhao_mep.pdf                                 "https://arxiv.org/pdf/2112.11701"
download_pdf 2023_li_cole.pdf                                  "https://arxiv.org/pdf/2302.04831"
download_pdf 2023_yan_e3t.pdf "https://papers.nips.cc/paper_files/paper/2023/file/07a363fd2263091c2063998e0034999c-Paper-Conference.pdf"
download_pdf 2024_wang_zsc_eval.pdf                            "https://arxiv.org/pdf/2310.05208"
download_pdf 2025_ruhdorfer_ogc.pdf                            "https://arxiv.org/pdf/2406.17949"

# -----------------------------------------------------------------------------
# 2. Repos externos -> external/   (shallow, sin LFS: el zoo de agentes es opcional)
# -----------------------------------------------------------------------------
log "2/4 Repos externos -> external/"
mkdir -p "$EXT"

clone_repo() { # carpeta url
    local dir="$EXT/$1" url="$2"
    if [ -d "$dir/.git" ]; then
        log "  ya existe: external/$1"
    else
        log "  clonando:  $url"
        GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$url" "$dir"
    fi
}

clone_repo ZSC-Eval           https://github.com/sjtu-marl/ZSC-Eval.git
clone_repo overcooked_env_gen https://github.com/icaros-usc/overcooked_env_gen.git

# -----------------------------------------------------------------------------
# 3. Entorno python (micromamba/conda) + extras de RL
# -----------------------------------------------------------------------------
if [ "${SKIP_ENV:-0}" = "1" ]; then
    log "3/4 SKIP_ENV=1 -> se omite el entorno"
else
    log "3/4 Entorno '$ENV_NAME'"
    MAMBA="$(command -v micromamba || command -v mamba || command -v conda || true)"
    if [ -z "$MAMBA" ]; then
        warn "  no hay micromamba/conda; instala micromamba o corre con SKIP_ENV=1 dentro de un venv 3.10"
        exit 1
    fi
    if ! "$MAMBA" env list | grep -qE "(^|/| )$ENV_NAME( |$)"; then
        log "  creando entorno desde environment.yml"
        "$MAMBA" env create -y -f "$ROOT/environment.yml"
    else
        log "  ya existe el entorno"
    fi

    PYBIN="$("$MAMBA" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
    log "  python: $PYBIN"

    # torch: cpu por defecto; en máquina NVIDIA usa TORCH_FLAVOR=cu121 (o auto-detección)
    FLAVOR="${TORCH_FLAVOR:-auto}"
    if [ "$FLAVOR" = "auto" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then FLAVOR=cu121; else FLAVOR=cpu; fi
    fi
    if ! "$PYBIN" -c 'import torch' 2>/dev/null; then
        log "  instalando torch ($FLAVOR)"
        if [ "$FLAVOR" = "cpu" ]; then
            "$PYBIN" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
        else
            "$PYBIN" -m pip install -q torch --index-url "https://download.pytorch.org/whl/$FLAVOR"
        fi
    else
        log "  torch ya instalado"
    fi

    for spec in "stable_baselines3:stable-baselines3>=2.3" "gymnasium:gymnasium>=0.29" "pandas:pandas"; do
        mod="${spec%%:*}"; pkg="${spec#*:}"
        if ! "$PYBIN" -c "import $mod" 2>/dev/null; then
            log "  instalando $pkg"
            "$PYBIN" -m pip install -q "$pkg" "numpy<2"
        else
            log "  $mod ya instalado"
        fi
    done

    # guardia: overcooked_ai 1.1.0 requiere numpy<2 (np.Inf fue removido en numpy 2)
    "$PYBIN" -m pip install -q "numpy<2" >/dev/null
fi

# -----------------------------------------------------------------------------
# 4. Verificación
# -----------------------------------------------------------------------------
if [ "${SKIP_ENV:-0}" = "1" ]; then
    log "4/4 verificación omitida (SKIP_ENV=1)"
else
    log "4/4 Verificación final"
    "$PYBIN" - <<'PY'
import numpy, os
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__}: overcooked_ai necesita <2"
import torch, gymnasium, stable_baselines3, pandas
import overcooked_ai_py
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
mdp = OvercookedGridworld.from_layout_name("cramped_room")
data = os.path.join(os.path.dirname(overcooked_ai_py.__file__), "data", "human_data")
pickles = [f for f in os.listdir(data) if f.endswith(".pickle")]
print("  numpy", numpy.__version__, "| torch", torch.__version__,
      "| sb3", stable_baselines3.__version__, "| gymnasium", gymnasium.__version__)
print("  overcooked OK | data humana:", ", ".join(sorted(pickles)))
PY
fi

log "Listo. Ver PLANEACION.md para el plan completo."
