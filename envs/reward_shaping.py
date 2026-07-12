"""Shaping denso por eventos + annealing (receta human_aware_rl) — PLAN.md Etapa 1.3.

El shaping por EVENTOS ya lo calcula el entorno del profesor: `overcooked_ai_py`
usa `reward_shaping_params = {PLACEMENT_IN_POT_REW: 3, DISH_PICKUP_REWARD: 3,
SOUP_PICKUP_REWARD: 5}`, que coincide EXACTAMENTE con la receta del PLAN
(+3 ingrediente en olla, +3 recoger plato, +5 recoger sopa). Ese valor llega por
`info["shaped_r_by_agent"]` en cada `env.step`.

Este módulo aporta lo que falta: el **coeficiente de annealing** que multiplica al
shaped reward y decae linealmente de 1.0 a 0.0 entre el paso 0 y `anneal_end_step`
(default del PLAN: 60% del total de pasos de entrenamiento). Así el agente aprende
primero con la guía densa y luego se apoya solo en la recompensa sparse (sopas).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShapingSchedule:
    """Coeficiente lineal 1.0 -> 0.0 sobre el shaped reward.

    Args:
        anneal_end_step: paso (global de entrenamiento) en el que el coeficiente
            llega a `end_coef`. A partir de ahí se mantiene en `end_coef`.
        start_coef: coeficiente en el paso 0 (default 1.0).
        end_coef: coeficiente final (default 0.0).
    """

    anneal_end_step: float
    start_coef: float = 1.0
    end_coef: float = 0.0

    def coef(self, step: float) -> float:
        """Coeficiente de shaping en el `step` global dado (interpolación lineal)."""
        if self.anneal_end_step <= 0:
            return self.end_coef
        frac = step / float(self.anneal_end_step)
        frac = min(max(frac, 0.0), 1.0)
        return self.start_coef + (self.end_coef - self.start_coef) * frac

    @classmethod
    def from_total_steps(
        cls,
        total_steps: float,
        anneal_fraction: float = 0.6,
        start_coef: float = 1.0,
        end_coef: float = 0.0,
    ) -> "ShapingSchedule":
        """Construye el schedule anealando hasta `anneal_fraction` del total (default 60%)."""
        return cls(
            anneal_end_step=float(total_steps) * float(anneal_fraction),
            start_coef=start_coef,
            end_coef=end_coef,
        )
