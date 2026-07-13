"""StudentAgent que carga un checkpoint SB3 (MLP+featurized) — para render/eval.

Usado por la plantilla del profesor vía `type: python_class`. Recibe la obs featurized
del ObservationBuilder (dict `{"obs": (96,), "agent_index": int}`) y devuelve un int 0-5.

NO es el entregable final de la Etapa 7 (ese vive en deliverable/ con weights/ y default.pt);
esto es una utilidad para VISUALIZAR nuestros checkpoints con el runner del profesor.

Config esperado: {"checkpoint": "<ruta al .zip SB3>"}.
"""

from __future__ import annotations

import numpy as np


class StudentAgent:
    def __init__(self, config=None):
        config = config or {}
        self.checkpoint = config["checkpoint"]
        self.model = None
        try:
            from stable_baselines3 import PPO

            self.model = PPO.load(self.checkpoint, device="cpu")
            self.reset()  # calienta el forward (evita latencia en la 1a acción)
        except Exception as exc:
            print("StudentAgent(sb3): no se pudo cargar el modelo, uso fallback stay:", repr(exc))

    def reset(self):
        if self.model is not None:
            try:
                self.model.predict(np.zeros(96, np.float32), deterministic=True)
            except Exception:
                pass

    def act(self, obs) -> int:
        try:
            x = obs["obs"] if isinstance(obs, dict) else obs
            action, _ = self.model.predict(np.asarray(x, np.float32), deterministic=True)
            return int(np.asarray(action).item())
        except Exception:
            return 4  # stay: nunca romper el runner
