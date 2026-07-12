#!/usr/bin/env bash
# =============================================================================
# setup_local.sh — Vertiente B (local), camino ALTERNATIVO venv+pip — ver PLAN.md, sección 2B
#
# El camino PREFERIDO en local es micromamba vía ./setup.sh (entorno reproducible
# con Python 3.10 y numpy 1.26.*). Usa ESTE script solo si no hay micromamba/conda.
#
# Nota Windows: el timeout del profesor usa SIGALRM (solo Unix). Entrenar en Windows
# funciona, pero el autotest de timeouts (Etapa 7) debe correrse en Linux/WSL/Colab.
#
# Uso:  bash scripts/setup_local.sh   (crea .venv en la raíz del repo)
# =============================================================================
set -euo pipefail

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
# numpy<2 SIEMPRE primero (overcooked-ai usa np.Inf, removido en NumPy 2.0)
pip install "numpy<2" overcooked-ai stable-baselines3 gymnasium pyyaml tqdm

echo ""
echo "[OK] Entorno .venv listo. Actívalo con:  source .venv/bin/activate"
echo "torch: por defecto se instala la variante que traiga overcooked-ai/sb3 (CPU en la mayoría"
echo "       de casos). Si tienes GPU NVIDIA y quieres CUDA, instala la variante cu-* de torch."
echo "GATE 0:  pytest tests/test_env_smoke.py"
