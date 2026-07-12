"""GATE 1 — Tests del entorno ego/alt (ver PLAN.md, Etapa 1.4).

Criterio de avance:
  * 3 episodios con compañero greedy y acciones random del ego corren sin excepción
  * gymnasium check_env pasa
  * con randomize_index=True, ambos índices del ego aparecen en 20 resets

Además valida el registro de compañeros (partners.py) y el schedule de shaping.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.ego_env import OvercookedEgoEnv, OBS_DIM
from envs.partners import make_partner, partner_factory_from_spec
from envs.reward_shaping import ShapingSchedule


LAYOUT = "cramped_room"


def _make_env(spec=None, randomize_index=True, shaping_schedule=None):
    spec = spec or {"type": "greedy"}
    return OvercookedEgoEnv(
        layout_name_or_file=LAYOUT,
        partner_factory=partner_factory_from_spec(spec),
        horizon=250,
        randomize_index=randomize_index,
        shaping_schedule=shaping_schedule,
    )


# --------------------------------------------------------------------- GATE 1
def test_three_episodes_greedy_partner_no_exception():
    """3 episodios completos con compañero greedy y ego random: sin excepción."""
    env = _make_env({"type": "greedy"})
    rng = np.random.default_rng(0)
    for _ in range(3):
        obs, info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        steps = 0
        while True:
            a = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(a)
            assert obs.shape == (OBS_DIM,)
            assert np.isfinite(reward)
            steps += 1
            if terminated or truncated:
                break
        assert steps == 250, f"episodio debe durar horizon=250, duró {steps}"
        assert "soups_delivered" in info and "sparse_reward" in info


def test_gymnasium_check_env_passes():
    """El env cumple el contrato de gymnasium.Env."""
    env = _make_env({"type": "greedy"})
    # skip_render_check: este env no renderiza (entrenamiento headless / sin GPU).
    check_env(env, skip_render_check=True)


def test_randomize_index_covers_both_roles():
    """Con randomize_index=True ambos índices del ego aparecen en 20 resets."""
    env = _make_env({"type": "greedy"}, randomize_index=True)
    seen = set()
    for seed in range(20):
        env.reset(seed=seed)
        seen.add(env.ego_index)
    assert seen == {0, 1}, f"esperados ambos índices, vistos {seen}"


def test_randomize_index_false_is_stable():
    """Con randomize_index=False el ego es siempre el índice 0."""
    env = _make_env({"type": "greedy"}, randomize_index=False)
    for seed in range(5):
        env.reset(seed=seed)
        assert env.ego_index == 0


# ------------------------------------------------------------------- partners
@pytest.mark.parametrize(
    "spec",
    [
        {"type": "greedy"},
        {"type": "greedy", "sticky_p": 0.3},
        {"type": "greedy", "eps": 0.2},
        {"type": "greedy", "sticky_p": [0.0, 0.4], "eps": [0.0, 0.4]},
        {"type": "random_motion"},
        {"type": "stay"},
        {"type": "mixture", "specs": [{"type": "random_motion"}, {"type": "stay"}], "probs": [0.7, 0.3]},
    ],
)
def test_partner_specs_run_one_episode(spec):
    """Cada tipo de compañero corre un episodio completo en el ego_env sin fallar."""
    env = _make_env(spec)
    obs, _ = env.reset(seed=1)
    for _ in range(250):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert info["soups_delivered"] >= 0


def test_sticky_partner_repeats_actions():
    """Con sticky_p alto, el compañero repite su acción previa a menudo."""
    from envs.partners import StickyActionWrapper
    from policies.basic_policies import RandomMotionPolicy

    env = _make_env({"type": "greedy"})
    env.reset(seed=0)
    mdp = env.env.mdp

    sticky = StickyActionWrapper(RandomMotionPolicy(seed=3), sticky_prob=0.9, seed=3)
    sticky.set_mdp(mdp)
    sticky.set_agent_index(1)
    sticky.reset()

    repeats = 0
    n = 100
    for _ in range(n):
        _, info = sticky.action(env.env.state)
        if info.get("sticky_repeat"):
            repeats += 1
    # Con p=0.9 esperamos claramente más de la mitad de repeticiones.
    assert repeats > n * 0.5, f"esperadas muchas repeticiones, hubo {repeats}/{n}"


def test_mixture_samples_both_components():
    """La mezcla muestrea ambos componentes a lo largo de varios episodios."""
    spec = {"type": "mixture", "specs": [{"type": "random_motion"}, {"type": "stay"}], "probs": [0.5, 0.5]}
    from envs.partners import MixtureAgent

    seen_types = set()
    for i in range(40):
        agent = make_partner(spec)
        assert isinstance(agent, MixtureAgent)
        agent.set_agent_index(1)
        agent.reset()  # muestrea el sub-compañero
        seen_types.add(type(agent._sub).__name__)
    assert len(seen_types) == 2, f"esperados 2 tipos de sub-compañero, vistos {seen_types}"


# --------------------------------------------------------------- reward shaping
def test_shaping_schedule_anneals_linearly():
    """El coeficiente decae linealmente de 1.0 a 0.0 en [0, anneal_end_step]."""
    sched = ShapingSchedule(anneal_end_step=1000)
    assert sched.coef(0) == pytest.approx(1.0)
    assert sched.coef(500) == pytest.approx(0.5)
    assert sched.coef(1000) == pytest.approx(0.0)
    assert sched.coef(2000) == pytest.approx(0.0)  # se mantiene en end_coef


def test_shaping_schedule_from_total_steps():
    """from_total_steps anealiza hasta la fracción indicada (default 60%)."""
    sched = ShapingSchedule.from_total_steps(1_000_000, anneal_fraction=0.6)
    assert sched.anneal_end_step == pytest.approx(600_000)
    assert sched.coef(300_000) == pytest.approx(0.5)


def test_shaping_coef_applied_in_reward():
    """Con schedule, el reward del env aplica el coeficiente al shaped reward."""
    sched = ShapingSchedule(anneal_end_step=10)
    env = _make_env({"type": "greedy"}, shaping_schedule=sched)
    env.reset(seed=0)
    # Al inicio (step 0) coef=1.0; reward = sparse + 1.0*shaped.
    _, reward, _, _, info = env.step(env.action_space.sample())
    expected = info["sparse_reward"] + info["shaping_coef"] * info["shaped_reward"]
    assert reward == pytest.approx(expected)
