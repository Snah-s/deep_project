"""Callbacks y utilidades de entrenamiento — PLAN.md Etapa 3.

Lo importante aquí es `ScoreEvalCallback`: cada `eval_freq` pasos evalúa la política
con el HARNESS OFICIAL (Etapa 2) y guarda `best_model` por **score oficial**, no por la
reward de entrenamiento (regla 8 del PLAN). También hay:

  * `ShapingAnnealCallback`: empuja el paso global a los envs para anealar el shaping.
  * `SB3PolicyStudent`: adapta una política SB3 a la interfaz `act(obs)->int` del harness.
  * `linear_schedule`: learning rate con decaimiento lineal.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from evaluation.harness import evaluate


class SelfPlayCallback(BaseCallback):
    """Refresca en disco un snapshot congelado del ego (mecanismo self-play de E3T).

    Los compañeros de tipo `self` (checkpoint apuntando a `snapshot_path`) recargan este
    snapshot cuando cambia su mtime (caché en partners._load_sb3). Guarda de forma atómica
    (tmp + os.replace) para que un compañero nunca lea un zip a medio escribir.

    El snapshot INICIAL debe crearse ANTES de learn() (train_ppo lo hace): los envs lo leen
    en su primer reset. Este callback solo hace los refrescos periódicos.
    """

    def __init__(self, snapshot_path: str | Path, refresh_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.snapshot_path = str(snapshot_path)  # sin extensión .zip
        self.refresh_freq = int(refresh_freq)
        self._last_refresh = 0

    def save_snapshot(self) -> None:
        tmp = self.snapshot_path + ".tmp"
        self.model.save(tmp)  # crea tmp.zip
        os.replace(tmp + ".zip", self.snapshot_path + ".zip")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_refresh >= self.refresh_freq:
            self._last_refresh = self.num_timesteps
            self.save_snapshot()
            if self.verbose:
                print(f"[selfplay @ {self.num_timesteps}] snapshot del ego refrescado")
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """LR lineal: en SB3 `progress_remaining` va de 1.0 (inicio) a 0.0 (final)."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


class SB3PolicyStudent:
    """Adapta una política SB3 a la interfaz del entregable/harness: `act(obs)->int`.

    Acepta obs como dict `{"obs": ndarray, "agent_index": int}` (formato del profesor
    con include_agent_index=True) o como ndarray directo. Forward determinista (argmax).
    """

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = bool(deterministic)

    def reset(self):
        pass

    def act(self, obs) -> int:
        x = obs["obs"] if isinstance(obs, dict) else obs
        action, _ = self.model.predict(np.asarray(x), deterministic=self.deterministic)
        return int(np.asarray(action).item())


class ShapingAnnealCallback(BaseCallback):
    """Fija el paso global de entrenamiento en cada env para anealar el shaped reward.

    Se actualiza al inicio de cada rollout (barato) en vez de cada step. Los envs leen
    ese paso para calcular el coeficiente del `ShapingSchedule`.
    """

    def _on_rollout_start(self) -> None:
        self.training_env.env_method("set_progress_step", int(self.num_timesteps))

    def _on_step(self) -> bool:
        return True


class ScoreEvalCallback(BaseCallback):
    """Evalúa con el harness oficial cada `eval_freq` pasos y guarda el mejor por score.

    Guarda:
      * `<save_dir>/best_model.zip`  — mejor modelo SB3 por score_mean oficial.
      * `<save_dir>/last_model.zip`  — se guarda al final desde train_ppo.
      * `<save_dir>/eval_history.json` — historial de evaluaciones.
    """

    def __init__(
        self,
        layouts: str | list[str],
        partner_spec: dict,
        seeds: list[int],
        eval_freq: int,
        save_dir: str | Path,
        horizon: int = 250,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        # Acepta un layout o un pool (generalista multi-layout, p.ej. Esc.4).
        self.layouts = [layouts] if isinstance(layouts, str) else list(layouts)
        self.partner_spec = partner_spec
        self.seeds = list(seeds)
        self.eval_freq = int(eval_freq)
        self.save_dir = Path(save_dir)
        self.horizon = int(horizon)
        self.deterministic = bool(deterministic)
        self.best_score = -np.inf
        self.history: list[dict[str, Any]] = []
        self._last_eval_step = 0

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _run_eval(self) -> dict[str, dict[str, Any]]:
        # El harness usa set_global_seed (reseeds np.random/random). Guardamos y
        # restauramos el estado global para NO perturbar el RNG del entrenamiento.
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            student = SB3PolicyStudent(self.model, deterministic=self.deterministic)
            per_layout = {
                lay: evaluate(
                    agent_ctor=lambda: student,
                    layout=lay,
                    partner_spec=self.partner_spec,
                    seeds=self.seeds,
                    horizon=self.horizon,
                )
                for lay in self.layouts
            }
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        return per_layout

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        per_layout = self._run_eval()
        # Agregado sobre el pool: score/soups promedio, timeouts sumados.
        score = float(np.mean([r["score_mean"] for r in per_layout.values()]))
        soups = float(np.mean([r["soups_mean"] for r in per_layout.values()]))
        timeouts = int(sum(r["timeouts_total"] for r in per_layout.values()))

        entry = {
            "timesteps": int(self.num_timesteps),
            "score_mean": score,
            "soups_mean": soups,
            "timeouts_total": timeouts,
            "per_layout": {lay: {"score_mean": r["score_mean"], "soups_mean": r["soups_mean"]}
                           for lay, r in per_layout.items()},
        }
        self.history.append(entry)
        (self.save_dir / "eval_history.json").write_text(json.dumps(self.history, indent=2))

        self.logger.record("harness/score_mean", score)
        self.logger.record("harness/soups_mean", soups)
        self.logger.record("harness/timeouts_total", timeouts)

        tag = f"score={score:.1f} soups={soups:.2f} timeouts={timeouts}"
        if len(self.layouts) > 1:
            tag += " | " + " ".join(f"{lay[:8]}:{r['soups_mean']:.1f}" for lay, r in per_layout.items())

        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.save_dir / "best_model"))
            if self.verbose:
                print(f"[eval @ {self.num_timesteps}] NUEVO MEJOR {tag} -> best_model.zip")
        elif self.verbose:
            print(f"[eval @ {self.num_timesteps}] {tag} (best={self.best_score:.1f})")

        return True
