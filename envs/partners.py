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

import os


# Caché de modelos SB3 por ruta, invalidada por mtime. Clave para el self-play de E3T:
# el snapshot del ego se refresca en disco periódicamente; los compañeros lo recargan
# solo cuando cambia el mtime (no en cada episodio -> evita recargar 2MB constantemente).
_MODEL_CACHE: dict[str, tuple[float, Any]] = {}


def _load_sb3(path: str, device: str = "cpu"):
    p = path if str(path).endswith(".zip") else str(path) + ".zip"
    mtime = os.path.getmtime(p)
    cached = _MODEL_CACHE.get(path)
    if cached is None or cached[0] != mtime:
        from stable_baselines3 import PPO

        _MODEL_CACHE[path] = (mtime, PPO.load(path, device=device))
    return _MODEL_CACHE[path][1]


class _EpsilonWrapper(EpsilonActionWrapper):
    """EpsilonActionWrapper que además reenvía bind_env al agente base (para checkpoints)."""

    def bind_env(self, env):
        if hasattr(self.base_agent, "bind_env"):
            self.base_agent.bind_env(env)


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

    def bind_env(self, env):
        if hasattr(self.base_agent, "bind_env"):
            self.base_agent.bind_env(env)

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
        self._env = None

    def _sample_sub(self):
        idx = int(self.rng.choice(len(self.specs), p=self.probs))
        sub = make_partner(self.specs[idx])
        # reset() PRIMERO: Agent.reset() de overcooked limpia agent_index y mdp.
        sub.reset()
        if self._mdp is not None:
            sub.set_mdp(self._mdp)
        if self.agent_index is not None:
            sub.set_agent_index(self.agent_index)
        if self._env is not None and hasattr(sub, "bind_env"):
            sub.bind_env(self._env)
        self._sub = sub

    def bind_env(self, env):
        self._env = env
        if self._sub is not None and hasattr(self._sub, "bind_env"):
            self._sub.bind_env(env)

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
        # Caché por mtime: recarga el snapshot solo si cambió (self-play de E3T).
        self._model = _load_sb3(self.path, "cpu")

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

        if self._model is None:
            self._lazy_load()
        if self._env is None:
            raise RuntimeError("CheckpointAgent no está enlazado a un env (llama bind_env).")
        obs = self._env.featurize_state_mdp(state)[self.agent_index]
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


def _apply_wrappers(base: Agent, spec: dict, rng: np.random.Generator) -> Agent:
    """Aplica sticky/eps (escalar o rango) sobre un agente base (greedy o checkpoint).

    El `eps` implementa la componente random de la mezcla de E3T: `π = ε·base + (1-ε)·random`
    con `random_action_prob = eps`. Se usa `_EpsilonWrapper` para que reenvíe bind_env.
    """
    if spec.get("sticky_p") is not None:
        base = StickyActionWrapper(base, sticky_prob=_sample_scalar(spec["sticky_p"], rng), seed=spec.get("seed"))
    if spec.get("eps") is not None:
        base = _EpsilonWrapper(base, random_action_prob=_sample_scalar(spec["eps"], rng), seed=spec.get("seed"))
    return base


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
        return _apply_wrappers(base, spec, rng)

    if ptype == "random_motion":
        return RandomMotionPolicy(seed=spec.get("seed"))

    if ptype == "stay":
        return StayPolicy()

    if ptype == "checkpoint":
        if "path" not in spec:
            raise ValueError("spec checkpoint requiere 'path'")
        # deterministic False por defecto en self-play: queremos diversidad de conductas.
        base = CheckpointAgent(path=spec["path"], deterministic=bool(spec.get("deterministic", True)))
        return _apply_wrappers(base, spec, rng)

    if ptype == "mixture":
        return MixtureAgent(specs=spec["specs"], probs=spec.get("probs"), seed=spec.get("seed"))

    raise ValueError(f"Tipo de compañero desconocido: {ptype!r}")


def resolve_self_spec(spec: Any, self_path: str) -> Any:
    """Reemplaza recursivamente {'type':'self',...} por un checkpoint a `self_path`.

    El componente 'self' es la copia congelada del ego (E3T). Se usa deterministic=False
    (muestreo estocástico) para dar diversidad de conductas al compañero.
    """
    if not isinstance(spec, dict):
        return spec
    t = str(spec.get("type", "")).lower()
    if t == "self":
        out = dict(spec)
        out["type"] = "checkpoint"
        out["path"] = str(self_path)
        out.setdefault("deterministic", False)
        return out
    if t == "mixture":
        out = dict(spec)
        out["specs"] = [resolve_self_spec(s, self_path) for s in spec["specs"]]
        return out
    return spec


def partner_factory_from_spec(spec: dict) -> Callable[[], Agent]:
    """Devuelve una fábrica sin argumentos que crea un compañero fresco por llamada.

    OvercookedEgoEnv la invoca en cada reset -> nuevo compañero por episodio, con
    rangos/mezclas remuestreados.
    """
    return lambda: make_partner(spec)
