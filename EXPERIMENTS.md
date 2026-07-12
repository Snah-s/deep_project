# EXPERIMENTS — bitácora de corridas y decisiones

> Formato por entrada: fecha · etapa · config · resultado (score del harness cuando aplique).
> Las decisiones que se desvían de los hiperparámetros base del PLAN se registran aquí
> con su motivo (regla 9 del PLAN).

---

## Etapa 0 — Bootstrap y GATE 0 (2026-07-12)

- **Entorno canónico:** micromamba `overcooked` (Python 3.10.20, numpy 1.26.4,
  stable-baselines3 2.9.0, torch 2.13.0+cpu, gymnasium 1.3.0, overcooked-ai). Se decidió
  usar `overcooked` en lugar del `overcooked-agent` que nombraba el plan: es el env que ya
  tenía el stack completo y funcionando. `setup.sh`, `environment.yml` y las referencias del
  PLAN se renombraron a `overcooked` (solo el nombre; sin tocar pines ni lógica).
- **GATE 0:** `pytest tests/test_env_smoke.py` → 6 passed. Verificado:
  - `featurize_state_mdp` → `(96,)` por agente ✓
  - `lossless_state_encoding_mdp` → `(5,4,26)` en cramped_room = `(H,W,26)` ✓
  - Episodio completo `GreedyFullTaskPolicy(0)` + `RandomMotionPolicy(1)` (semillas fijas,
    determinista) → sparse reward = 100 (5 sopas) > 0 ✓
  - `stable_baselines3` + `torch` importan ✓
- **Lockfile:** `requirements.lock` generado con `pip freeze` tras el GATE en verde.
- **Ajuste al smoke test:** en esta versión de overcooked_ai_py el MDP no expone
  `old_dynamics` como atributo consultable; se valida de forma conductual (la olla arranca
  a cocinar sola y el greedy entrega sopa).

---

## Etapa 1 — Entorno de entrenamiento ego/alt (2026-07-12)

- **`envs/ego_env.py` — `OvercookedEgoEnv(gymnasium.Env)`**: un agente (ego) con el
  compañero (alt) resuelto dentro de `step()`. Obs `Box(96,)` featurized del ego,
  acción `Discrete(6)`. `randomize_index=True` sortea el índice del ego por episodio
  (cubre el role-swap). Reward = `sparse_ego + coef(step)·shaped_ego`. Termina por
  horizonte → se reporta como `truncated`, no `terminated`.
- **`envs/partners.py` — `make_partner(spec)`**: greedy, greedy+sticky, greedy+eps,
  ambos, random_motion, stay, checkpoint (SB3 congelado, Etapa 5), mixture. Rangos
  `[lo,hi]` en sticky_p/eps se muestrean por episodio. `StickyActionWrapper` reimplementado
  (el zip no lo trae); `EpsilonActionWrapper` reutilizado del zip.
- **`envs/reward_shaping.py` — `ShapingSchedule`**: coeficiente lineal 1.0→0.0 hasta
  `anneal_end_step` (helper `from_total_steps(total, 0.6)`). NO reimplementa la detección
  de eventos: el env del profesor ya expone el shaping +3/+3/+5 en `info["shaped_r_by_agent"]`
  (sus `reward_shaping_params` coinciden exactamente con la receta del PLAN).
- **Detalle técnico:** `Agent.reset()` de overcooked limpia `agent_index` y `mdp`, así que
  el orden correcto al preparar un compañero es `reset()` → `set_mdp()` → `set_agent_index()`.
- **GATE 1:** `pytest tests/test_ego_env.py` → 16 passed (3 episodios greedy sin excepción,
  `gymnasium.check_env` OK, ambos índices del ego en 20 resets, specs de compañeros, schedule).
  Suite completa: 22 passed. Todo en CPU (esta máquina no tiene GPU; ver nota de entorno).

---

## Observación para la Etapa 2 (a tener en cuenta)
- Par de control `greedy(0)+greedy(1)` en `cramped_room` con `old_dynamics=True` entregó
  **0 sopas** (ambos greedy con `avoid_teammate=True` se estorban en un layout diminuto).
  El GATE 2 asume `greedy+greedy > 0`; habrá que revisar el par de control (p. ej. otro
  layout, o `avoid_teammate` en uno de ellos) al implementar el harness.
