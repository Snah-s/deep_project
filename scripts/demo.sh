#!/usr/bin/env bash
# =============================================================================
# demo.sh — Muestra el ENTREGABLE (deliverable/) jugando con ventana pygame.
#
# Uso (desde cualquier carpeta):
#   scripts/demo.sh [1|2|3|all]     ventana pygame en vivo (default: all)
#   scripts/demo.sh 3 --gif          sin ventana: genera outputs/demo_esc3.gif (headless)
#
# Escenarios:  1=asymmetric_advantages  2=coordination_ring  3=counter_circuit
# agent_0 (AZUL) = nuestro modelo; agent_1 (verde) = greedy_full_task.
# Requiere el env 'overcooked' (micromamba) y un display para la ventana.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

WHICH="${1:-all}"
GIF=""
[ "${2:-}" = "--gif" ] && GIF="--gif"

case "$WHICH" in
  1|2|3) SCENS="$WHICH" ;;
  all)   SCENS="1 2 3" ;;
  *) echo "Uso: scripts/demo.sh [1|2|3|all] [--gif]"; exit 1 ;;
esac

RUN="micromamba run -n overcooked"
command -v micromamba >/dev/null || RUN=""   # si no hay micromamba, usa el python activo

for n in $SCENS; do
  echo ">> Escenario $n  (azul = entregable)"
  $RUN python tools/play.py --config "tools/play_esc${n}.yaml" $GIF
done
