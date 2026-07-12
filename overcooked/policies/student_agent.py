"""StudentAgent — entregable principal basado en el greedy arreglado.

Interfaz esperada por el runner:
    __init__(self, config: dict)
    reset(self)
    act(self, obs) -> int   (0..5)

Requiere `observation.type: state` en la evaluacion: el agente recibe el estado crudo
`{"state": state, "mdp": mdp, "agent_index": idx}` y corre `GreedyFullTaskPolicy` (con
rompe-deadlock). Red de seguridad en capas: cualquier problema -> "stay" (nunca rompe el runner).

Acciones: 0 norte, 1 sur, 2 este, 3 oeste, 4 stay, 5 interact.
"""
from __future__ import annotations

from overcooked_ai_py.mdp.actions import Action

# Mapa Action -> indice publico, robusto entre versiones de overcooked_ai.
_raw = Action.INDEX_TO_ACTION
if hasattr(_raw, "items"):
    _ACTION_TO_INDEX = {a: int(i) for i, a in _raw.items()}
else:
    _ACTION_TO_INDEX = {a: i for i, a in enumerate(_raw)}

# El greedy arreglado vive en el repo junto a este archivo.
try:
    from policies.basic_policies import GreedyFullTaskPolicy
except Exception:  # por si el paquete se importa con otro prefijo
    from overcooked.policies.basic_policies import GreedyFullTaskPolicy


class StudentAgent:
    def __init__(self, config=None):
        self.config = config or {}
        self.greedy = GreedyFullTaskPolicy(
            ingredient=str(self.config.get("ingredient", "onion")),
            avoid_teammate=bool(self.config.get("avoid_teammate", True)),
            seed=self.config.get("seed", None),
        )
        self._mdp = None
        self._idx = None

    def reset(self):
        try:
            self.greedy.reset()
        except Exception:
            pass
        self._mdp = None
        self._idx = None

    def act(self, obs) -> int:
        try:
            # observation.type == "state": {"state": ..., "mdp": ..., "agent_index": ...}
            if isinstance(obs, dict) and "state" in obs and "mdp" in obs:
                state = obs["state"]
                mdp = obs["mdp"]
                idx = int(obs.get("agent_index", 0))
                # (re)configurar el greedy si cambio el mdp (nuevo layout) o el asiento (role swap)
                if mdp is not self._mdp:
                    self.greedy.set_mdp(mdp)
                    self._mdp = mdp
                if idx != self._idx:
                    self.greedy.set_agent_index(idx)
                    self._idx = idx
                action, _ = self.greedy.action(state)
                return int(_ACTION_TO_INDEX[action])
            # Sin estado crudo no podemos correr el greedy -> stay seguro.
            return 4
        except Exception:
            # Capa final: nunca romper el runner.
            return 4  # stay
