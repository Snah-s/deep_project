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
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from evaluation.harness import evaluate


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
        layout: str,
        partner_spec: dict,
        seeds: list[int],
        eval_freq: int,
        save_dir: str | Path,
        horizon: int = 250,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.layout = layout
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

    def _run_eval(self) -> dict[str, Any]:
        # El harness usa set_global_seed (reseeds np.random/random). Guardamos y
        # restauramos el estado global para NO perturbar el RNG del entrenamiento.
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            student = SB3PolicyStudent(self.model, deterministic=self.deterministic)
            result = evaluate(
                agent_ctor=lambda: student,
                layout=self.layout,
                partner_spec=self.partner_spec,
                seeds=self.seeds,
                horizon=self.horizon,
            )
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        return result

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        result = self._run_eval()
        score = result["score_mean"]
        soups = result["soups_mean"]
        timeouts = result["timeouts_total"]

        entry = {
            "timesteps": int(self.num_timesteps),
            "score_mean": score,
            "soups_mean": soups,
            "timeouts_total": timeouts,
        }
        self.history.append(entry)
        (self.save_dir / "eval_history.json").write_text(json.dumps(self.history, indent=2))

        # Logs a tensorboard/stdout de SB3.
        self.logger.record("harness/score_mean", score)
        self.logger.record("harness/soups_mean", soups)
        self.logger.record("harness/timeouts_total", timeouts)

        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.save_dir / "best_model"))
            if self.verbose:
                print(f"[eval @ {self.num_timesteps}] NUEVO MEJOR score={score:.1f} "
                      f"soups={soups:.2f} timeouts={timeouts} -> best_model.zip")
        elif self.verbose:
            print(f"[eval @ {self.num_timesteps}] score={score:.1f} soups={soups:.2f} "
                  f"timeouts={timeouts} (best={self.best_score:.1f})")

        return True
