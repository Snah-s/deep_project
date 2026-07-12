"""Paquete `envs` — entorno de entrenamiento ego/alt y compañeros.

Bootstrap de path: el template intocable del profesor vive en `<repo>/overcooked/`
y usa imports de nivel superior (`src.*`, `policies.*`). Añadimos esa carpeta al
sys.path al importar el paquete, para que `envs.ego_env` / `envs.partners` funcionen
igual desde pytest, desde los scripts de training y desde Colab.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE_DIR = _REPO_ROOT / "overcooked"

if _TEMPLATE_DIR.is_dir() and str(_TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(_TEMPLATE_DIR))
