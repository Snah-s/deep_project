"""Paquete `evaluation` — harness de score oficial y autotest del entregable.

Bootstrap de path: expone el template del profesor (`<repo>/overcooked/`, imports
`src.*` y `policies.*`) al sys.path, igual que `envs/__init__.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE_DIR = _REPO_ROOT / "overcooked"

if _TEMPLATE_DIR.is_dir() and str(_TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(_TEMPLATE_DIR))
