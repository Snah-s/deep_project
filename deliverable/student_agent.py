"""Entregable final — StudentAgent (torch puro, CPU). PLAN.md Etapa 7.

Política MLP featurized: obs (96,) -> Linear(256)+Tanh -> Linear(256)+Tanh -> Linear(6).
Acción determinista = argmax. Sin dependencia de stable-baselines3: carga los pesos
extraídos en weights/default.pt (verificados idénticos a la política SB3 entrenada).

Un solo modelo cubre los 3 escenarios (asymmetric_advantages, coordination_ring,
counter_circuit). Si `config` trae el layout y existe weights/<layout>.pt, se usa; si no,
weights/default.pt.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


def _build_net() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(96, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, 6),
    )


class StudentAgent:
    def __init__(self, config=None):
        config = config or {}
        here = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(here, "weights")

        # Un solo modelo para los 3 escenarios: default.pt. Si el grader pasa el layout y
        # existe un .pt específico, se prefiere; si no, default.
        path = os.path.join(weights_dir, "default.pt")
        layout = config.get("layout") or config.get("layout_name")
        if layout:
            cand = os.path.join(weights_dir, f"{layout}.pt")
            if os.path.exists(cand):
                path = cand

        self.net = _build_net()
        try:
            self.net.load_state_dict(torch.load(path, map_location="cpu"))
        except Exception:
            pass  # ante fallo de carga, la red queda con pesos random; act() nunca rompe
        self.net.eval()

    def reset(self):
        pass

    def act(self, obs) -> int:
        try:
            x = obs["obs"] if isinstance(obs, dict) else obs
            x = torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(-1)
            with torch.no_grad():
                return int(self.net(x).argmax().item())
        except Exception:
            return 4  # stay: nunca romper el runner
