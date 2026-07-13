"""Autotest del entregable — GATE 7 (PLAN.md Etapa 7).

Valida `deliverable/student_agent.py` con el runner del profesor:
  * sopas_mean >= 1 en los 3 escenarios (vs greedy, 3 seeds, swap activado),
  * latencia de act(): p99 < 20 ms en CPU (margen 5x sobre el límite de 100 ms;
    p99 << 100 ms garantiza 0 timeouts).

Uso (desde la raíz del repo, en Linux):
    micromamba run -n overcooked python -m evaluation.selfcheck

Salida 0 = GATE 7 en verde; salida 1 = falla.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "overcooked"))
sys.path.insert(0, os.path.join(REPO, "deliverable"))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless

# Los 3 escenarios revelados (compañero greedy_full_task, con la perturbación del escenario).
SCENARIOS = [
    ("asymmetric_advantages", 0.0, "Esc1"),
    ("coordination_ring", 0.0, "Esc2"),
    ("counter_circuit", 0.15, "Esc3"),
]
MIN_SOUPS = 1.0
MAX_P99_MS = 20.0
DELIVERABLE = os.path.join(REPO, "deliverable", "student_agent.py")


def _run_scenario(layout: str, random_action_prob: float) -> float:
    from src.runner import run_from_config

    cfg = {
        "seed": 67,
        "environment": {"layout_name": layout, "horizon": 250, "old_dynamics": True},
        "policies": {
            "agent_0": {
                "type": "python_class", "path": DELIVERABLE, "class_name": "StudentAgent",
                "config": {}, "max_action_time_ms": 100,
                "invalid_action": "stay", "timeout_action": "stay",
            },
            "agent_1": {
                "type": "builtin", "name": "greedy_full_task",
                "random_action_prob": random_action_prob, "max_action_time_ms": 100,
                "invalid_action": "stay", "timeout_action": "stay",
            },
        },
        "execution": {"num_episodes": 3, "episode_seeds": [67, 68, 69], "swap_agent_positions": True},
        "observation": {"type": "featurized", "include_agent_index": True},
        "rendering": {"mode": "none", "save_gif": False},
        "logging": {"output_dir": "outputs/selfcheck", "save_step_log": False,
                    "save_episode_summary": False, "save_trajectory_pickle": False},
    }
    return run_from_config(cfg)["mean_return_sparse"] / 20.0  # +20 sparse por sopa


def _latency_p99_ms(n: int = 5000) -> float:
    from student_agent import StudentAgent

    ag = StudentAgent({})
    rng = np.random.default_rng(0)
    ts = np.empty(n)
    for i in range(n):
        o = rng.normal(size=96).astype(np.float32)
        t0 = time.perf_counter()
        ag.act(o)
        ts[i] = (time.perf_counter() - t0) * 1000.0
    return float(np.percentile(ts, 99))


def main() -> int:
    ok = True

    print("== sopas por escenario (vs greedy, 3 seeds, swap) ==")
    for layout, rnd, esc in SCENARIOS:
        soups = _run_scenario(layout, rnd)
        passed = soups >= MIN_SOUPS
        ok &= passed
        print(f"  {esc} {layout:24} soups={soups:.2f}  {'OK' if passed else 'FALLA (<1)'}")

    p99 = _latency_p99_ms()
    lat_ok = p99 < MAX_P99_MS
    ok &= lat_ok
    print(f"== latencia act(): p99={p99:.3f} ms  {'OK' if lat_ok else f'FALLA (>= {MAX_P99_MS})'} ==")

    print(f"\nGATE 7: {'VERDE ✅' if ok else 'ROJO ❌'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
