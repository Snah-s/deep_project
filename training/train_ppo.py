"""Entrenamiento PPO vs un compañero — PLAN.md Etapa 3.

Camino principal del proyecto. Un solo agente (ego) entrena con SB3 PPO contra un
compañero embebido en el entorno (`OvercookedEgoEnv`). El mejor modelo se elige con el
HARNESS OFICIAL (score de la competencia), nunca con la reward de entrenamiento.

Uso (máquina con GPU, u otra máquina):
    python -m training.train_ppo --config training/configs/esc1.yaml
    python -m training.train_ppo --config training/configs/esc1.yaml --layout coordination_ring
    python -m training.train_ppo --config training/configs/esc1.yaml --smoke   # prueba rápida CPU

En Colab: mismo comando vía `!python -m training.train_ppo ...` (ver colab/run_all.ipynb).
Si SubprocVecEnv falla en Colab, usar `--vec dummy`.

Artefactos en <output_dir>/<experiment_name>_<layout>/:
    best_model.zip     (mejor por score oficial)
    last_model.zip     (al terminar)
    eval_history.json  (curva de score del harness)
    config_used.yaml   (config efectiva, para reproducir)
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.ego_env import OvercookedEgoEnv
from envs.partners import partner_factory_from_spec, resolve_self_spec
from envs.reward_shaping import ShapingSchedule
from training.callbacks import (
    ScoreEvalCallback,
    SelfPlayCallback,
    ShapingAnnealCallback,
    linear_schedule,
)


_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


def load_policy_weights(model: PPO, init_from: str, device: str) -> PPO:
    """Finetune: transfiere SOLO los pesos de la política desde un checkpoint SB3.

    Se construye un PPO nuevo con los hiperparámetros de la config (optimizador y
    lr_schedule frescos, ent_coef del escenario) y aquí se copian los pesos aprendidos.
    Requiere MISMA arquitectura (net_arch/activation) que el checkpoint de origen.
    """
    src = PPO.load(str(init_from), device=device)
    model.policy.load_state_dict(src.policy.state_dict())
    del src
    return model


# --------------------------------------------------------------------- env fns
def _layout_list(cfg: dict[str, Any]) -> list[str]:
    """Normaliza cfg['layout'] a lista (soporta pool multi-layout, p.ej. Esc.4)."""
    lay = cfg["layout"]
    return list(lay) if isinstance(lay, list) else [lay]


def _uses_self(spec: Any) -> bool:
    """True si el partner_spec incluye una componente 'self' (self-play de E3T)."""
    if not isinstance(spec, dict):
        return False
    t = str(spec.get("type", "")).lower()
    if t == "self":
        return True
    if t == "mixture":
        return any(_uses_self(s) for s in spec.get("specs", []))
    return False


def _prewarm(cfg: dict[str, Any], layout: str) -> None:
    """Pre-calienta el motion planner (featurize) de un layout, con compañero trivial.

    Usa partner 'stay' para NO depender del snapshot self-play (que aún no existe al
    momento del prewarm). Con fork, los subprocesos heredan el planner (memoria+disco).
    """
    env = OvercookedEgoEnv(
        layout_name_or_file=layout,
        partner_factory=partner_factory_from_spec({"type": "stay"}),
        horizon=int(cfg["horizon"]),
        randomize_index=False,
        old_dynamics=bool(cfg["old_dynamics"]),
    )
    env.reset()
    env.close()


def _make_single_env(cfg: dict[str, Any], shaping_schedule: ShapingSchedule | None, rank: int, layout: str):
    """Crea un OvercookedEgoEnv (Monitor-wrapped) para el índice `rank`, fijado a `layout`.

    NO se resetea aquí: el reset (que crea el compañero) lo hace SB3 al empezar learn(),
    después de que exista el snapshot inicial del self-play.
    """
    env = OvercookedEgoEnv(
        layout_name_or_file=layout,
        partner_factory=partner_factory_from_spec(cfg["partner_spec"]),
        horizon=int(cfg["horizon"]),
        shaping_schedule=shaping_schedule,
        randomize_index=bool(cfg["randomize_index"]),
        old_dynamics=bool(cfg["old_dynamics"]),
    )
    return Monitor(env)


def _make_env_fn(cfg: dict[str, Any], shaping_schedule: ShapingSchedule | None, rank: int, layout: str):
    def _f():
        return _make_single_env(cfg, shaping_schedule, rank, layout)

    return _f


def _build_vecenv(cfg: dict[str, Any], shaping_schedule: ShapingSchedule | None):
    n_envs = int(cfg["n_envs"])
    pool = _layout_list(cfg)
    # Reparte los layouts del pool round-robin entre los workers (multi-layout).
    env_fns = [_make_env_fn(cfg, shaping_schedule, i, pool[i % len(pool)]) for i in range(n_envs)]
    if str(cfg["vec"]).lower() == "dummy" or n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


# ---------------------------------------------------------------------- config
def _load_config(path: str, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def _apply_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    """Prueba de humo: pipeline mínimo en CPU para validar que corre sin errores."""
    cfg = copy.deepcopy(cfg)
    cfg["total_timesteps"] = 2000
    cfg["n_envs"] = 2
    cfg["vec"] = "dummy"
    cfg["device"] = "cpu"
    cfg["ppo"] = dict(cfg["ppo"])
    cfg["ppo"]["n_steps"] = 128
    cfg["ppo"]["batch_size"] = 64
    cfg["eval"] = dict(cfg["eval"])
    cfg["eval"]["freq"] = 1000
    cfg["eval"]["seeds"] = [67]
    cfg["experiment_name"] = cfg["experiment_name"] + "_smoke"
    return cfg


# ------------------------------------------------------------------------ main
def train(cfg: dict[str, Any]) -> dict[str, Any]:
    ppo_cfg = cfg["ppo"]
    total_timesteps = int(cfg["total_timesteps"])

    pool = _layout_list(cfg)
    layout_tag = Path(str(pool[0])).stem if len(pool) == 1 else "multi"
    run_name = f"{cfg['experiment_name']}_{layout_tag}"
    save_dir = Path(cfg["output_dir"]) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Shaping anealado sobre pasos GLOBALES (el ShapingAnnealCallback fija el paso).
    shaping_schedule = ShapingSchedule.from_total_steps(
        total_timesteps, anneal_fraction=float(cfg["shaping"]["anneal_fraction"])
    )

    # Self-play (E3T): las componentes `self` del partner_spec apuntan a un snapshot
    # congelado del ego en disco. Se resuelve a un checkpoint ANTES de construir los envs;
    # el snapshot inicial se guarda tras crear el modelo (los envs lo leen en su 1er reset).
    snapshot_path = None
    if _uses_self(cfg["partner_spec"]):
        snapshot_path = save_dir / "frozen_ego"
        cfg["partner_spec"] = resolve_self_spec(cfg["partner_spec"], str(snapshot_path))

    # Pre-calentar el motion planner (featurize) de CADA layout una vez en el proceso
    # principal: con fork, los subprocesos heredan el planner (memoria y disco) y evitan carreras.
    for lay in pool:
        _prewarm(cfg, lay)

    vec_env = _build_vecenv(cfg, shaping_schedule)

    lr = float(ppo_cfg["learning_rate"])
    learning_rate = linear_schedule(lr) if str(ppo_cfg.get("lr_schedule")) == "linear" else lr

    policy_kwargs = dict(
        net_arch=list(ppo_cfg["net_arch"]),
        activation_fn=_ACTIVATIONS[str(ppo_cfg["activation"]).lower()],
    )

    # tensorboard es opcional: solo se activa si está instalado (no bloquea el pipeline).
    tb_log = None
    try:
        import tensorboard  # noqa: F401

        tb_log = str(save_dir / "tb")
    except ImportError:
        pass

    model = PPO(
        policy=str(ppo_cfg["policy"]),
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg["ent_coef"]),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        device=str(cfg["device"]),
        seed=int(cfg["seed"]),
        tensorboard_log=tb_log,
        verbose=1,
    )

    init_from = cfg.get("init_from")
    if init_from:
        load_policy_weights(model, str(init_from), str(cfg["device"]))
        print(f"[finetune] pesos de política cargados desde: {init_from}")

    eval_cb = ScoreEvalCallback(
        layouts=cfg["layout"],
        partner_spec=cfg["eval"]["partner_spec"],
        seeds=cfg["eval"]["seeds"],
        eval_freq=int(cfg["eval"]["freq"]),
        save_dir=save_dir,
        horizon=int(cfg["horizon"]),
    )
    anneal_cb = ShapingAnnealCallback()

    callbacks = [anneal_cb, eval_cb]

    # Self-play: snapshot inicial ANTES de learn() (los envs lo leen en su 1er reset) +
    # callback que lo refresca periódicamente para que el ego juegue contra copias recientes.
    if snapshot_path is not None:
        model.save(str(snapshot_path))
        refresh_freq = int(cfg.get("selfplay_refresh_freq", 100000))
        callbacks.append(SelfPlayCallback(snapshot_path, refresh_freq, verbose=1))
        print(f"[selfplay] snapshot inicial: {snapshot_path}.zip  (refresh cada {refresh_freq} pasos)")
    ckpt_freq = int(cfg.get("checkpoint_freq", 0) or 0)
    if ckpt_freq > 0:
        # save_freq de CheckpointCallback cuenta pasos POR env -> dividir por n_envs.
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, ckpt_freq // int(cfg["n_envs"])),
                save_path=str(save_dir / "checkpoints"),
                name_prefix="ckpt",
            )
        )

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)

    model.save(str(save_dir / "last_model"))
    vec_env.close()

    print(f"\n[done] run={run_name}  best_score={eval_cb.best_score:.1f}")
    print(f"       artefactos en: {save_dir}")
    return {"save_dir": str(save_dir), "best_score": float(eval_cb.best_score), "history": eval_cb.history}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento PPO vs compañero (Etapa 3).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--layout", default=None, help="override del layout (nombre o archivo .layout)")
    parser.add_argument("--init-from", default=None, help="checkpoint SB3 para finetune (override de init_from)")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--vec", default=None, choices=["subproc", "dummy"])
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="prueba rápida en CPU (valida el pipeline)")
    args = parser.parse_args(argv)

    overrides = {
        "layout": args.layout,
        "init_from": args.init_from,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "vec": args.vec,
        "device": args.device,
        "output_dir": args.output_dir,
        "seed": args.seed,
    }
    cfg = _load_config(args.config, overrides)
    if args.smoke:
        cfg = _apply_smoke(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
