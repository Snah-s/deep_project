"""Entorno de entrenamiento ego/alt de un solo agente — PLAN.md Etapa 1.1.

Un `gymnasium.Env` donde el compañero ("alt") vive DENTRO del `step()`: el agente
que entrena SB3 es el "ego" y solo controla una acción; la del compañero se resuelve
internamente. Patrón ego/alt reimplementado desde cero sobre el env del profesor
(no se clona PantheonRL — regla 2 del PLAN).

Puntos clave (parámetros fijos de la competencia):
  * Observación: Box(shape=(96,), float32) = featurize_state_mdp(state)[ego_index]
  * Acción: Discrete(6) con el mapeo 0=N,1=S,2=E,3=O,4=stay,5=interact
  * horizon=250, old_dynamics=True
  * randomize_index cubre el role-swap: el ego juega como índice 0 o 1 por episodio
  * reward = sparse_ego + coef(step) * shaped_ego   (shaping anealado, ver reward_shaping)
"""

from __future__ import annotations

import os
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from overcooked_ai_py.agents.agent import Agent

from src.constants import action_index_to_overcooked_action
from src.environment import build_env

from envs.reward_shaping import ShapingSchedule

# Recompensa sparse por sopa entregada (convención del env; el PLAN la fija en +20).
DELIVERY_REWARD = 20.0

OBS_DIM = 96
NUM_ACTIONS = 6


class OvercookedEgoEnv(gym.Env):
    """Entorno de un agente con el compañero embebido en step()."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        layout_name_or_file: str,
        partner_factory: Callable[[], Agent],
        horizon: int = 250,
        shaping_schedule: ShapingSchedule | None = None,
        randomize_index: bool = True,
        old_dynamics: bool = True,
        nav_shaping_coef: float = 0.0,
    ):
        super().__init__()
        self.partner_factory = partner_factory
        self.horizon = int(horizon)
        self.shaping_schedule = shaping_schedule
        self.randomize_index = bool(randomize_index)
        self.nav_shaping_coef = float(nav_shaping_coef)

        self._env_config = self._build_env_config(layout_name_or_file, horizon, old_dynamics)
        # El MDP se construye una vez; cada episodio hace reset(regen_mdp=False).
        self.env = build_env(self._env_config)

        # Shaping de navegación opcional (fix counter_circuit). Se anexa al shaped reward
        # y se anela con el mismo coef. Ver envs/nav_shaping.py.
        self._nav = None
        if self.nav_shaping_coef > 0.0:
            from envs.nav_shaping import NavPotential
            self._nav = NavPotential(self.env.mdp)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Estado de episodio
        self.ego_index: int = 0
        self.partner_index: int = 1
        self.partner: Agent | None = None
        self.soups_delivered: int = 0
        # Paso global de entrenamiento para el annealing del shaping. Auto-incrementa,
        # pero un callback de training puede fijarlo con set_progress_step (vectorizado).
        self._progress_step: int = 0

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _build_env_config(layout_name_or_file: str, horizon: int, old_dynamics: bool) -> dict[str, Any]:
        s = str(layout_name_or_file)
        is_file = s.endswith(".layout") or os.path.sep in s or os.path.exists(s)
        cfg: dict[str, Any] = {"horizon": int(horizon), "old_dynamics": bool(old_dynamics)}
        if is_file:
            cfg["layout_file"] = s
        else:
            cfg["layout_name"] = s
        return cfg

    def _featurize_ego(self) -> np.ndarray:
        obs_pair = self.env.featurize_state_mdp(self.env.state)
        return obs_pair[self.ego_index].astype(np.float32)

    def _current_coef(self) -> float:
        if self.shaping_schedule is None:
            return 1.0
        return float(self.shaping_schedule.coef(self._progress_step))

    def set_progress_step(self, step: int) -> None:
        """Fija el paso global de entrenamiento (para el annealing en modo vectorizado)."""
        self._progress_step = int(step)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.env.reset(regen_mdp=False)

        if self.randomize_index:
            self.ego_index = int(self.np_random.integers(0, 2))
        else:
            self.ego_index = 0
        self.partner_index = 1 - self.ego_index

        if self._nav is not None:
            self._nav.reset(self.ego_index)

        self.partner = self.partner_factory()
        # OJO: Agent.reset() de overcooked limpia agent_index y mdp, así que va PRIMERO.
        if hasattr(self.partner, "reset"):
            self.partner.reset()
        self.partner.set_mdp(self.env.mdp)
        self.partner.set_agent_index(self.partner_index)
        # Permite a compañeros aprendidos (checkpoints) featurizar el estado.
        if hasattr(self.partner, "bind_env"):
            self.partner.bind_env(self.env)

        self.soups_delivered = 0

        obs = self._featurize_ego()
        info = {"ego_index": self.ego_index}
        return obs, info

    def step(self, action):
        ego_action = action_index_to_overcooked_action(int(action))
        alt_action, _ = self.partner.action(self.env.state)

        # Ordenar la acción conjunta por índice de agente.
        joint = [None, None]
        joint[self.ego_index] = ego_action
        joint[self.partner_index] = alt_action

        _, sparse_total, done, info = self.env.step(tuple(joint))

        sparse_by_agent = info["sparse_r_by_agent"]
        shaped_by_agent = info["shaped_r_by_agent"]
        ego_sparse = float(sparse_by_agent[self.ego_index])
        ego_shaped = float(shaped_by_agent[self.ego_index])

        coef = self._current_coef()
        # Navegación: recompensa densa de acercarse al subobjetivo (anelada con el shaping).
        nav_reward = self._nav.step_reward(self.env.state) if self._nav is not None else 0.0
        reward = ego_sparse + coef * (ego_shaped + self.nav_shaping_coef * nav_reward)

        # Contar sopas del episodio a partir de la sparse total (+20 por sopa, PLAN).
        soups_this_step = int(round(float(sparse_total) / DELIVERY_REWARD))
        self.soups_delivered += soups_this_step

        self._progress_step += 1

        # Overcooked termina solo por horizonte -> es truncamiento, no terminación.
        terminated = False
        truncated = bool(done)

        obs = self._featurize_ego()
        step_info = {
            "sparse_reward": ego_sparse,
            "shaped_reward": ego_shaped,
            "shaping_coef": coef,
            "soups_delivered": int(self.soups_delivered),
            "ego_index": self.ego_index,
        }
        return obs, float(reward), terminated, truncated, step_info
