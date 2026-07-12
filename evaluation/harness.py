"""Harness de score oficial — PLAN.md Etapa 2.

Mide EXACTAMENTE lo que mide el profesor. Es el árbitro de todo el proyecto: TODA
decisión de "¿está listo?" se toma con este harness, nunca con la reward de entreno.

Score oficial por intento (regla 5 del PLAN):

    score = 10000·sopas
          + 10·(horizon − t_última_sopa)
          + (horizon − t_primera_sopa)
          − min(100·timeouts, 5000)

    (0 sopas -> score 0)

Protocolo `evaluate`:
  * Por cada seed corre 2 episodios: rol normal (agente=índice 0) y rol invertido
    (agente=índice 1). Cubre el `swap_agent_positions` de la competencia.
  * Ambos agentes pasan por `SafeActionWrapper` (100 ms) del profesor, para contar
    timeouts reales del agente evaluado.
  * Detecta sopas por la sparse reward (+20 por sopa) y registra el timestep de la
    primera y última entrega.

El agente evaluado puede ser:
  * un objeto estilo-entregable con `act(obs)->int` (se adapta con StudentAgentAdapter), o
  * un Agent de overcooked-ai (política scripted / checkpoint), usado tal cual.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

import numpy as np

from overcooked_ai_py.agents.agent import Agent

from src.constants import overcooked_action_to_index  # noqa: F401  (fuerza carga del template)
from src.environment import build_env
from src.observations import ObservationBuilder
from src.policy_wrappers import SafeActionWrapper, StudentAgentAdapter
from src.runner import set_global_seed

from envs.partners import make_partner

DELIVERY_REWARD = 20.0
DEFAULT_SEEDS = (67, 68, 69)
DEFAULT_HORIZON = 250
MAX_ACTION_TIME_MS = 100


# --------------------------------------------------------------------- scoring
def official_score(
    soups: int,
    t_first_soup: int,
    t_last_soup: int,
    timeouts: int,
    horizon: int = DEFAULT_HORIZON,
) -> float:
    """Score oficial de la competencia para un intento (un episodio)."""
    if soups <= 0:
        return 0.0
    return (
        10000.0 * soups
        + 10.0 * (horizon - t_last_soup)
        + (horizon - t_first_soup)
        - min(100.0 * timeouts, 5000.0)
    )


# ----------------------------------------------------------------- env helpers
def _layout_config(layout: str, horizon: int, old_dynamics: bool) -> dict[str, Any]:
    import os

    s = str(layout)
    is_file = s.endswith(".layout") or os.path.sep in s or os.path.exists(s)
    cfg: dict[str, Any] = {"horizon": int(horizon), "old_dynamics": bool(old_dynamics)}
    if is_file:
        cfg["layout_file"] = s
    else:
        cfg["layout_name"] = s
    return cfg


def _as_overcooked_agent(obj: Any, obs_builder: ObservationBuilder) -> Agent:
    """Normaliza el agente evaluado a un Agent de overcooked-ai."""
    if isinstance(obj, Agent):
        return obj
    if hasattr(obj, "act"):
        return StudentAgentAdapter(obj, obs_builder)
    raise TypeError(
        f"El agente evaluado debe ser un Agent de overcooked-ai o tener .act(obs)->int; got {type(obj)}"
    )


# ------------------------------------------------------------------- one match
def _run_episode(
    env,
    obs_builder: ObservationBuilder,
    test_obj: Any,
    partner: Agent,
    test_index: int,
    horizon: int,
) -> dict[str, Any]:
    """Corre un episodio con el agente evaluado en `test_index` y devuelve su score."""
    test_agent = SafeActionWrapper(
        _as_overcooked_agent(test_obj, obs_builder), max_action_time_ms=MAX_ACTION_TIME_MS
    )
    partner_agent = SafeActionWrapper(partner, max_action_time_ms=MAX_ACTION_TIME_MS)

    partner_index = 1 - test_index
    agents: list[Agent] = [test_agent, test_agent]
    agents[test_index] = test_agent
    agents[partner_index] = partner_agent

    # Compañeros aprendidos (checkpoints) necesitan el env para featurizar.
    if hasattr(partner, "bind_env"):
        partner.bind_env(env)

    for idx, agent in enumerate(agents):
        agent.reset()
        agent.set_mdp(env.mdp)
        agent.set_agent_index(idx)

    env.reset(regen_mdp=False)

    soups = 0
    t_first: int | None = None
    t_last: int | None = None
    done = False
    while not done:
        state = env.state
        a0, _ = agents[0].action(state)
        a1, _ = agents[1].action(state)
        next_state, _sparse_total, done, info = env.step((a0, a1))
        sparse = float(sum(info["sparse_r_by_agent"]))
        if sparse > 0:
            soups += int(round(sparse / DELIVERY_REWARD))
            if t_first is None:
                t_first = int(next_state.timestep)
            t_last = int(next_state.timestep)

    timeouts = int(test_agent.timeout_count)
    tf = t_first if t_first is not None else horizon
    tl = t_last if t_last is not None else horizon
    score = official_score(soups, tf, tl, timeouts, horizon)
    return {
        "score": score,
        "soups": soups,
        "t_first_soup": t_first,
        "t_last_soup": t_last,
        "timeouts": timeouts,
        "test_index": test_index,
    }


# --------------------------------------------------------------------- evaluate
def evaluate(
    agent_ctor: Callable[[], Any],
    layout: str,
    partner_spec: dict,
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    horizon: int = DEFAULT_HORIZON,
    old_dynamics: bool = True,
) -> dict[str, Any]:
    """Evalúa `agent_ctor` contra `partner_spec` en `layout` con el score oficial.

    Devuelve un dict con al menos {score_mean, soups_mean, timeouts_total} y el
    desglose por seed y por rol.
    """
    env = build_env(_layout_config(layout, horizon, old_dynamics))
    obs_builder = ObservationBuilder(env, {"type": "featurized", "include_agent_index": True})

    per_seed: list[dict[str, Any]] = []
    all_scores: list[float] = []
    all_soups: list[int] = []
    timeouts_total = 0

    for seed in seeds:
        attempts = []
        for role_swap in (False, True):
            set_global_seed(int(seed))
            test_obj = agent_ctor()
            partner = make_partner(partner_spec)
            test_index = 1 if role_swap else 0
            res = _run_episode(env, obs_builder, test_obj, partner, test_index, horizon)
            res["role_swap"] = role_swap
            attempts.append(res)
            all_scores.append(res["score"])
            all_soups.append(res["soups"])
            timeouts_total += res["timeouts"]
        per_seed.append(
            {
                "seed": int(seed),
                "attempts": attempts,
                "score_mean": float(np.mean([a["score"] for a in attempts])),
                "soups_mean": float(np.mean([a["soups"] for a in attempts])),
            }
        )

    return {
        "layout": layout,
        "partner_spec": partner_spec,
        "num_attempts": len(all_scores),
        "score_mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "score_std": float(np.std(all_scores)) if all_scores else 0.0,
        "soups_mean": float(np.mean(all_soups)) if all_soups else 0.0,
        "timeouts_total": int(timeouts_total),
        "per_seed": per_seed,
    }


# -------------------------------------------------------------------- CLI utils
def spec_from_string(s: str) -> dict:
    """Convierte 'greedy' | 'stay' | 'random_motion' | 'checkpoint:PATH' en un spec."""
    if s.startswith("checkpoint:"):
        return {"type": "checkpoint", "path": s.split(":", 1)[1]}
    return {"type": s}


def agent_ctor_from_spec(spec: dict) -> Callable[[], Agent]:
    """Fábrica de un agente scripted/checkpoint como 'agente evaluado' (para CLI/controles)."""
    return lambda: make_partner(spec)


def _summary(result: dict) -> dict:
    return {
        "layout": result["layout"],
        "score_mean": round(result["score_mean"], 2),
        "score_std": round(result["score_std"], 2),
        "soups_mean": round(result["soups_mean"], 3),
        "timeouts_total": result["timeouts_total"],
        "num_attempts": result["num_attempts"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Harness de score oficial (Etapa 2).")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--agent", default="greedy", help="greedy|stay|random_motion|checkpoint:PATH")
    parser.add_argument("--partner", default="greedy", help="greedy|stay|random_motion|checkpoint:PATH")
    parser.add_argument("--seeds", default="67,68,69")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args(argv)

    seeds = [int(x) for x in str(args.seeds).split(",") if x != ""]
    result = evaluate(
        agent_ctor=agent_ctor_from_spec(spec_from_string(args.agent)),
        layout=args.layout,
        partner_spec=spec_from_string(args.partner),
        seeds=seeds,
        horizon=args.horizon,
    )
    print(json.dumps(_summary(result), indent=2))


if __name__ == "__main__":
    main()
