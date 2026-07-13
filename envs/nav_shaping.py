"""Shaping de NAVEGACIÓN (potencial por distancia al subobjetivo) — fix de counter_circuit.

Problema: en layouts de anillo largo (counter_circuit) la reward esparsa es inalcanzable
por exploración aleatoria y el shaping por eventos (pickups) premia el interact-spam. El
agente se planta y machaca `interact` sin recorrer el anillo.

Fix: recompensa densa por ACERCARSE al siguiente subobjetivo (dispensador -> olla -> plato
-> entrega). Se reutiliza el oráculo de subtareas del greedy (`_choose_target`) y su BFS de
distancia. Es potential-based DENTRO de cada segmento de subtarea: la recompensa es la
DIFERENCIA de distancia (prev - actual), que telescopa a (d_inicio - d_fin) por segmento
-> no es farmeable (ir y volver se cancela). En la transición de subtarea (cambia el target)
se devuelve 0 y el bonus discreto lo pone el shaping por eventos (+3/+3/+5).
"""

from __future__ import annotations

from policies.basic_policies import GreedyFullTaskPolicy


class NavPotential:
    """Da recompensa densa de navegación hacia el subobjetivo actual del agente ego."""

    def __init__(self, mdp, ingredient: str = "onion"):
        # avoid_teammate=False: la potencial depende del ego + el mundo, no del compañero
        # (más estable; que el compañero se mueva no debe cambiar nuestra potencial).
        self._oracle = GreedyFullTaskPolicy(ingredient=ingredient, avoid_teammate=False)
        self._oracle.set_mdp(mdp)
        self._valid = set(mdp.get_valid_player_positions())
        self._prev_target: tuple[int, int] | None = None
        self._prev_dist: int | None = None

    def reset(self, ego_index: int) -> None:
        self._oracle.set_agent_index(ego_index)
        self._prev_target = None
        self._prev_dist = None

    def _distance(self, state, target) -> int | None:
        o = self._oracle
        goals = {p for p in o._adjacent_positions(target) if p in self._valid}
        if not goals:
            return None
        start = state.players[o.agent_index].position
        path = o._bfs_shortest_path(start, goals, self._valid, set())
        return None if path is None else len(path) - 1

    def step_reward(self, state) -> float:
        """Recompensa de navegación de este paso: (dist_prev - dist_actual) si el target no
        cambió; 0 si cambió (transición de subtarea) o no hay target definido."""
        target = self._oracle._choose_target(state)
        if target is None:
            self._prev_target = self._prev_dist = None
            return 0.0
        dist = self._distance(state, target)
        if dist is None:
            self._prev_target, self._prev_dist = target, None
            return 0.0
        if target == self._prev_target and self._prev_dist is not None:
            r = float(self._prev_dist - dist)
        else:
            r = 0.0  # cambió el subobjetivo -> el bonus lo da el shaping por eventos
        self._prev_target, self._prev_dist = target, dist
        return r
