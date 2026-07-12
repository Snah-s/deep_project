#!/usr/bin/env bash
# =============================================================================
# setup_colab.sh — Vertiente A (Colab / Kaggle / Lightning) — ver PLAN.md, sección 2A
#
# En estas plataformas NO se usa micromamba: se instala con pip y se REINICIA el
# runtime, porque Colab precarga numpy 2 y el downgrade a numpy<2 solo aplica
# tras reiniciar el intérprete.
#
# Uso dentro de un notebook:  !bash scripts/setup_colab.sh
# =============================================================================
set -euo pipefail

pip install -q "numpy<2" overcooked-ai stable-baselines3 gymnasium pyyaml tqdm

echo ""
echo "**REINICIA EL RUNTIME AHORA (Entorno de ejecución -> Reiniciar) y vuelve a correr desde la celda 2**"
echo ""
echo "Motivo: Colab precarga numpy 2; el downgrade a numpy<2 solo aplica tras reiniciar el runtime."
