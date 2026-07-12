"""Registro de compañeros para OvercookedEgoEnv — PLAN.md Etapa 1.2.

`make_partner(spec: dict) -> Agent` construye un compañero (Agent de overcooked-ai)
a partir de una especificación declarativa. Specs soportadas:

  {"type": "greedy"}                              -> GreedyFullTaskPolicy (del zip)
  {"type": "greedy", "sticky_p": 0.3}             -> greedy + StickyActionWrapper (aquí)
  {"type": "greedy", "eps": 0.2}                  -> greedy + EpsilonActionWrapper (del zip)
  {"type": "greedy", "sticky_p": 0.3, "eps": 0.2} -> ambos wrappers combinados
  {"type": "random_motion"}                       -> RandomMotionPolicy (del zip)
  {"type": "stay"}                                -> StayPolicy (del zip)
  {"type": "checkpoint", "path": ...}             -> política SB3 congelada (Etapa 5)
  {"type": "mixture", "specs": [...], "probs":[...]} -> muestrea un spec por episodio

Los valores `sticky_p` y `eps` aceptan un escalar o un rango `[lo, hi]`; en el
segundo caso se muestrea U(lo, hi) al construir el compañero (una vez por episodio,
porque OvercookedEgoEnv rehace el compañero en cada reset).

Nota de paths: importar este módulo pasa por `envs/__init__.py`, que añade el
template del profesor (`overcooked/`) al sys.path.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from overcooked_ai_py.agents.agent import Agent

from policies.basic_policies import (
    GreedyFullTaskPolicy,
    RandomMotionPolicy,
    StayPolicy,
)
from src.policy_wrappers import EpsilonActionWrapper


# ---------------------------------------------------------------------------
# Wrapper "sticky" (el zip no lo trae): repite la acción previa con prob p.
# ---------------------------------------------------------------------------
class StickyActionWrapper(Agent):
    """Con probabilidad `sticky_prob`, repite la acción del paso anterior.

    Simula un compañero con "inercia"/lag: en vez de la acción que propone el
    agente base, reemite la que tomó el paso anterior. Modela los Escenarios 2-3
    del PLAN (compañero pegajoso).
    """

    def __init__(self, base_agent: Agent, sticky_prob: float = 0.0, seed: int | None = None):
        super().__init__()
        self.base_agent = base_agent
        self.sticky_prob = float(sticky_prob)
        self.rng = np.random.default_rng(seed)
        self._last_action = None

    def reset(self):
        super().reset()
        self._last_action = None
        if hasattr(self, "base_agent"):
            self.base_agent.reset()

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        self.base_agent.set_agent_index(agent_index)

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        self.base_agent.set_mdp(mdp)

    def action(self, state):
        base_action, info = self.base_agent.action(state)
        info = dict(info or {})
        if self._last_action is not None and self.sticky_prob > 0 and self.rng.random() < self.sticky_prob:
            action = self._last_action
            info["sticky_repeat"] = True
        else:
            action = base_action
            info["sticky_repeat"] = False
        self._last_action = action
        return action, info


# ---------------------------------------------------------------------------
# Compañero de mezcla: muestrea un sub-compañero por episodio (en reset()).
# ---------------------------------------------------------------------------
class MixtureAgent(Agent):
    """Muestrea uno de varios sub-compañeros por episodio, según `probs`."""

    def __init__(self, specs: Sequence[dict], probs: Sequence[float] | None = None, seed: int | None = None):
        super().__init__()
        if not specs:
            raise ValueError("mixture necesita al menos un spec en 'specs'")
        self.specs = list(specs)
        if probs is None:
            probs = [1.0 / len(self.specs)] * len(self.specs)
        probs = np.asarray(probs, dtype=float)
        if len(probs) != len(self.specs):
            raise ValueError("mixture: len(probs) debe igualar len(specs)")
        self.probs = probs / probs.sum()
        self.rng = np.random.default_rng(seed)
        self._sub: Agent | None = None
        self._mdp = None

    def _sample_sub(self):
        idx = int(self.rng.choice(len(self.specs), p=self.probs))
        sub = make_partner(self.specs[idx])
        # reset() PRIMERO: Agent.reset() de overcooked limpia agent_index y mdp.
        sub.reset()
        if self._mdp is not None:
            sub.set_mdp(self._mdp)
        if self.agent_index is not None:
            sub.set_agent_index(self.agent_index)
        self._sub = sub

    def reset(self):
        super().reset()
        # Agent.__init__ puede llamar reset() antes de que estén los atributos.
        if hasattr(self, "rng"):
            self._sample_sub()

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        self._mdp = mdp
        if self._sub is not None:
            self._sub.set_mdp(mdp)

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        if self._sub is not None:
            self._sub.set_agent_index(agent_index)

    def action(self, state):
        if self._sub is None:
            self._sample_sub()
        return self._sub.action(state)


# ---------------------------------------------------------------------------
# Compañero desde checkpoint SB3 (usado en Etapa 5). Congelado, forward CPU.
# ---------------------------------------------------------------------------
class CheckpointAgent(Agent):
    """Envuelve una política SB3 entrenada para usarla como compañero congelado.

    Necesita featurizar el estado; para eso se enlaza al entorno con `bind_env`
    (OvercookedEgoEnv lo llama tras crear el compañero). Forward determinista
    (argmax) en CPU, sin gradiente.
    """

    def __init__(self, path: str, deterministic: bool = True):
        super().__init__()
        self.path = str(path)
        self.deterministic = bool(deterministic)
        self._model = None
        self._env = None

    def _lazy_load(self):
        if self._model is None:
            from stable_baselines3 import PPO

            self._model = PPO.load(self.path, device="cpu")

    def bind_env(self, env):
        """Enlaza el OvercookedEnv y carga el modelo AQUÍ (fuera del timing de act).

        Cargar el modelo dentro de action() sería fatal: el harness envuelve al agente
        en SafeActionWrapper (SIGALRM a 100 ms) y PPO.load tarda más que eso; el SIGALRM
        interrumpiría una llamada C de torch -> segfault. Por eso se carga en el binding.
        """
        self._env = env
        self._lazy_load()

    def action(self, state):
        from src.constants import action_index_to_overcooked_action

        self._lazy_load()
        if self._env is None:
            raise RuntimeError("CheckpointAgent no está enlazado a un env (llama bind_env).")
        obs = np.asarray(self._env.featurize_state_mdp(state)[self.agent_index], dtype=np.float32)
        action_idx, _ = self._model.predict(obs, deterministic=self.deterministic)
        action_idx = int(np.asarray(action_idx).item())
        return action_index_to_overcooked_action(action_idx), {"policy_name": "checkpoint", "action_index": action_idx}


# ---------------------------------------------------------------------------
# Fábrica principal
# ---------------------------------------------------------------------------
def _sample_scalar(value: Any, rng: np.random.Generator) -> float:
    """Escalar directo, o U(lo, hi) si `value` es un rango [lo, hi]."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Rango debe ser [lo, hi], got {value}")
        lo, hi = float(value[0]), float(value[1])
        return float(rng.uniform(lo, hi))
    return float(value)


def make_partner(spec: dict, rng: np.random.Generator | None = None) -> Agent:
    """Construye un compañero (Agent) a partir de un spec declarativo.

    Se muestrean los rangos [lo, hi] de `sticky_p`/`eps` en el momento de crear el
    compañero. Como OvercookedEgoEnv crea un compañero nuevo por episodio, eso
    equivale a muestrear una "personalidad" por episodio.
    """
    if rng is None:
        rng = np.random.default_rng(spec.get("seed"))
    ptype = str(spec.get("type", "greedy")).lower()

    if ptype == "greedy":
        base: Agent = GreedyFullTaskPolicy(
            ingredient=spec.get("ingredient", "onion"),
            avoid_teammate=bool(spec.get("avoid_teammate", True)),
            seed=spec.get("seed"),
        )
        if "sticky_p" in spec and spec["sticky_p"] is not None:
            sticky_p = _sample_scalar(spec["sticky_p"], rng)
            base = StickyActionWrapper(base, sticky_prob=sticky_p, seed=spec.get("seed"))
        if "eps" in spec and spec["eps"] is not None:
            eps = _sample_scalar(spec["eps"], rng)
            base = EpsilonActionWrapper(base, random_action_prob=eps, seed=spec.get("seed"))
        return base

    if ptype == "random_motion":
        return RandomMotionPolicy(seed=spec.get("seed"))

    if ptype == "stay":
        return StayPolicy()

    if ptype == "checkpoint":
        if "path" not in spec:
            raise ValueError("spec checkpoint requiere 'path'")
        return CheckpointAgent(path=spec["path"], deterministic=bool(spec.get("deterministic", True)))

    if ptype == "mixture":
        return MixtureAgent(specs=spec["specs"], probs=spec.get("probs"), seed=spec.get("seed"))

    raise ValueError(f"Tipo de compañero desconocido: {ptype!r}")


def partner_factory_from_spec(spec: dict) -> Callable[[], Agent]:
    """Devuelve una fábrica sin argumentos que crea un compañero fresco por llamada.

    OvercookedEgoEnv la invoca en cada reset -> nuevo compañero por episodio, con
    rangos/mezclas remuestreados.
    """
    return lambda: make_partner(spec)
