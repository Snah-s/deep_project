"""GATE 0 — Smoke test del entorno (ver PLAN.md, Etapa 0).

Verifica que todo el stack del camino principal está vivo y que la plantilla del
profesor produce las observaciones y recompensas esperadas:

  * overcooked_ai_py importa y construye `cramped_room` con old_dynamics=True, horizon 250
  * featurize_state_mdp   -> vector (96,) por agente
  * lossless_state_encoding_mdp -> tensor (H, W, 26) por agente
  * un episodio completo Greedy + RandomMotion entrega al menos una sopa (sparse > 0)
  * stable_baselines3 y torch importan sin error

REGLA: numpy<2 es obligatorio. Si algo revienta al importar overcooked_ai_py,
lo más probable es numpy>=2 (usa np.Inf). Confirmar `numpy<2` en el entorno.
"""

from __future__ import annotations

import numpy as np
import pytest

# El template (src.*, policies.*) se expone vía tests/conftest.py
from src.environment import build_env
from policies.basic_policies import GreedyFullTaskPolicy, RandomMotionPolicy


ENV_CONFIG = {
    "layout_name": "cramped_room",
    "layout_file": None,
    "horizon": 250,
    "old_dynamics": True,
}


@pytest.fixture()
def env():
    e = build_env(ENV_CONFIG)
    e.reset()
    return e


def test_numpy_below_2():
    """numpy<2 es requisito duro del stack (overcooked-ai usa np.Inf)."""
    major = int(np.__version__.split(".")[0])
    assert major < 2, f"Se requiere numpy<2, encontrado {np.__version__}"


def test_env_builds_with_competition_params(env):
    """El env se construye con los parámetros fijos de la competencia.

    `horizon=250` se comprueba directamente. `old_dynamics=True` no se expone como
    atributo consultable en esta versión de overcooked_ai_py (se aplica en la dinámica
    interna del MDP), así que se valida de forma conductual en
    `test_full_episode_delivers_soup`: con old_dynamics la olla arranca a cocinar sola
    al llenarse y el greedy logra entregar sopa.
    """
    assert env.horizon == 250
    assert env.mdp is not None
    assert env.mdp.num_players == 2


def test_featurized_shape_is_96(env):
    """featurize_state_mdp -> (96,) por agente (la obs del entregable)."""
    obs_pair = env.featurize_state_mdp(env.state)
    assert len(obs_pair) == 2
    for agent_obs in obs_pair:
        arr = np.asarray(agent_obs, dtype=np.float32)
        assert arr.shape == (96,), f"esperado (96,), obtenido {arr.shape}"


def test_lossless_grid_last_dim_is_26(env):
    """lossless_state_encoding_mdp -> (H, W, 26) por agente."""
    obs_pair = env.lossless_state_encoding_mdp(env.state)
    assert len(obs_pair) == 2
    for agent_obs in obs_pair:
        arr = np.asarray(agent_obs)
        assert arr.ndim == 3, f"esperado 3 dims (H,W,26), obtenido {arr.shape}"
        assert arr.shape[-1] == 26, f"esperado 26 canales, obtenido {arr.shape[-1]}"


def test_full_episode_delivers_soup():
    """Un episodio Greedy(0) + RandomMotion(1) entrega >= 1 sopa (sparse > 0).

    El agente greedy completa el ciclo aun con un compañero que solo se mueve;
    esto valida que el bucle env.step + políticas del zip funciona de punta a punta.
    Semillas fijas -> episodio determinista y test estable.
    """
    e = build_env(ENV_CONFIG)
    e.reset()

    greedy = GreedyFullTaskPolicy(seed=0)
    partner = RandomMotionPolicy(seed=0)
    greedy.set_mdp(e.mdp)
    partner.set_mdp(e.mdp)
    greedy.set_agent_index(0)
    partner.set_agent_index(1)

    total_sparse = 0.0
    done = False
    steps = 0
    info = {}
    while not done:
        a0, _ = greedy.action(e.state)
        a1, _ = partner.action(e.state)
        _, _, done, info = e.step((a0, a1))
        total_sparse += float(sum(info["sparse_r_by_agent"]))
        steps += 1

    assert steps == 250, f"el episodio debe durar horizon=250 pasos, duró {steps}"
    assert total_sparse > 0, f"sparse reward debe ser > 0 (sopa entregada), fue {total_sparse}"


def test_rl_stack_imports():
    """stable_baselines3 y torch importan sin error (camino principal de RL)."""
    import stable_baselines3  # noqa: F401
    import torch  # noqa: F401

    assert hasattr(stable_baselines3, "PPO")
    assert torch.tensor([1.0]).sum().item() == 1.0
