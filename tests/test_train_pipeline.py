"""Tests del pipeline de entrenamiento (Etapa 3) — guardas ligeras, sin entrenar de verdad.

El entrenamiento real (5e6 pasos) NO corre aquí (esta máquina no tiene GPU): va a Colab
o a la máquina con GPU. Estos tests validan que el cableado no se rompe y, sobre todo,
guardan la regresión del segfault de CheckpointAgent (cargar el modelo bajo el SIGALRM
del SafeActionWrapper -> segfault).
"""

from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.ego_env import OvercookedEgoEnv
from envs.partners import make_partner, partner_factory_from_spec
from evaluation.harness import evaluate


def _tiny_ppo_saved(path) -> str:
    env = DummyVecEnv(
        [lambda: OvercookedEgoEnv("cramped_room", partner_factory_from_spec({"type": "greedy"}), horizon=250)]
    )
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu", policy_kwargs=dict(net_arch=[64, 64]))
    model.save(str(path))
    env.close()
    return str(path)


def test_checkpoint_agent_eval_no_segfault(tmp_path):
    """Un checkpoint SB3 evaluado como agente-bajo-test corre bajo el harness sin segfault.

    Regresión: el modelo debe cargarse en bind_env (fuera del timer de 100 ms), NO dentro
    de action(). Si se cargara bajo el SIGALRM, esto segfaultearía o contaría timeouts.
    """
    path = _tiny_ppo_saved(tmp_path / "m")
    res = evaluate(
        agent_ctor=lambda: make_partner({"type": "checkpoint", "path": path}),
        layout="cramped_room",
        partner_spec={"type": "greedy"},
        seeds=[67],
    )
    assert res["timeouts_total"] == 0, "la carga del modelo no debe ocurrir bajo el timer"
    assert {"score_mean", "soups_mean", "timeouts_total"} <= set(res.keys())


def test_checkpoint_agent_as_partner(tmp_path):
    """El mismo checkpoint funciona como COMPAÑERO (uso de Etapa 5)."""
    path = _tiny_ppo_saved(tmp_path / "m")
    res = evaluate(
        agent_ctor=lambda: make_partner({"type": "greedy"}),
        layout="cramped_room",
        partner_spec={"type": "checkpoint", "path": path},
        seeds=[67],
    )
    assert res["timeouts_total"] == 0


def test_finetune_transfers_policy_weights(tmp_path):
    """Etapa 4: load_policy_weights copia exactamente los pesos de la política."""
    import torch
    from training.train_ppo import load_policy_weights

    path = _tiny_ppo_saved(tmp_path / "src")
    src = PPO.load(path, device="cpu")

    env = DummyVecEnv(
        [lambda: OvercookedEgoEnv("cramped_room", partner_factory_from_spec({"type": "greedy"}), horizon=250)]
    )
    dst = PPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu", policy_kwargs=dict(net_arch=[64, 64]))
    load_policy_weights(dst, path, "cpu")

    for (k, v_src), (_, v_dst) in zip(src.policy.state_dict().items(), dst.policy.state_dict().items()):
        assert torch.equal(v_src.cpu(), v_dst.cpu()), f"peso no transferido: {k}"
    env.close()
