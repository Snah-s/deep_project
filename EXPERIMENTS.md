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

### Observación para la Etapa 2 (a tener en cuenta)
- Par de control `greedy(0)+greedy(1)` en `cramped_room` con `old_dynamics=True` entregó
  **0 sopas** (ambos greedy con `avoid_teammate=True` se estorban en un layout diminuto).
  El GATE 2 asume `greedy+greedy > 0`; habrá que revisar el par de control (p. ej. otro
  layout, o `avoid_teammate` en uno de ellos) al implementar el harness.
