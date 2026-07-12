"""Pytest config: expone el template del profesor a los tests.

El template intocable vive en `<repo>/overcooked/` (copiado tal cual del zip del
profesor) y usa imports de nivel superior `src.*` y `policies.*`. Para importarlo
desde los tests, añadimos esa carpeta al sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "overcooked"

if str(TEMPLATE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_DIR))
