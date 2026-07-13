"""Runner de visualización: corre un config de la plantilla del profesor.

Uso (desde la raíz del repo):
    micromamba run -n overcooked python tools/play.py --config tools/play_coord.yaml         # ventana pygame en vivo
    micromamba run -n overcooked python tools/play.py --config tools/play_coord.yaml --gif    # genera un GIF (headless)

Evita el problema de cwd de `python -m src.run_game`: añade overcooked/ al sys.path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gif", action="store_true", help="render headless a GIF (sin ventana)")
    args = ap.parse_args()

    if args.gif:
        # Driver de video dummy: renderiza superficies sin necesidad de display.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    sys.path.insert(0, str(REPO / "overcooked"))
    from src.config import load_yaml
    from src.runner import run_from_config

    cfg = load_yaml(args.config)
    if args.gif:
        cfg["rendering"]["mode"] = "gif"
        cfg["rendering"]["save_gif"] = True

    print(json.dumps(run_from_config(cfg), indent=2))


if __name__ == "__main__":
    main()
