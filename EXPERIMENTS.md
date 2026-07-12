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

## Etapa 2 — Harness de score oficial (2026-07-12)

- **`evaluation/harness.py`**: `evaluate(agent_ctor, layout, partner_spec, seeds, horizon)`.
  Por seed corre 2 episodios (rol normal + invertido). Ambos agentes pasan por
  `SafeActionWrapper` (100 ms) del profesor para contar timeouts reales. Detecta sopas por
  sparse reward (+20) y registra timestep de primera/última sopa. `official_score()`
  implementa la fórmula del PLAN. El agente evaluado puede ser estilo-entregable
  (`act(obs)->int`, se adapta con `StudentAgentAdapter`) o un `Agent` de overcooked (scripted/
  checkpoint). CLI: `python -m evaluation.harness --layout L --agent A --partner P`.
- **GATE 2:** `pytest tests/test_harness.py` → 7 passed. Controles reproducidos:
  `stay+stay` → score 0; `greedy+greedy` → sopas>0 y score>0. Salida incluye
  `{score_mean, soups_mean, timeouts_total}`.

### Desvío documentado (regla 5 del PLAN) — layout del control greedy
El GATE 2 pedía `greedy+greedy > 0` **en cramped_room**, pero la greedy del profesor
(intencionalmente no óptima) se **atasca en cramped_room → 0 sopas** con cualquier
`avoid_teammate`. Sí coopera en otros layouts. Decisión: el control "greedy entrega"
se valida en `coordination_ring`; el control "stay → score 0" en `cramped_room`. Ambos
casos del GATE quedan cubiertos.

### Línea base `greedy+greedy` por layout (harness oficial, seeds 67,68,69 × 2 roles)

| Layout | soups_mean | score_mean | timeouts |
|---|---|---|---|
| cramped_room             | 0.0 | 0.0     | 0 |
| asymmetric_advantages    | 1.0 | 12200.0 | 0 |
| coordination_ring        | 8.0 | 80247.0 | 0 |
| counter_circuit_o_1order | 3.0 | 31055.0 | 0 |
| forced_coordination      | 0.0 | 0.0     | 0 |

> Estas cifras son el umbral a batir en la Etapa 3 (`harness(agente) ≥ harness(greedy,greedy)`).
> OJO: en `cramped_room` y `forced_coordination` el baseline greedy es 0, así que ahí basta
> con que el agente entrenado entregue ≥1 sopa para superarlo.

---

## Etapa 3 — Agente Escenario 1 (PPO vs greedy) — CÓDIGO LISTO, entreno pendiente (2026-07-12)

Esta máquina NO tiene GPU: aquí se preparó y validó el pipeline; el entreno real de 5e6
pasos corre en **Colab** (`colab/run_all.ipynb`) o en la **máquina con GPU** (12 GB).

- **`training/train_ppo.py`** (CLI + YAML): construye `OvercookedEgoEnv` vectorizado
  (Subproc/Dummy), PPO `MlpPolicy` 2×256 tanh, lr 3e-4 lineal, ent 0.05, γ0.99, λ0.95,
  n_steps 1024, batch 2048, n_epochs 8. Compañero greedy, `randomize_index=True`.
- **`training/callbacks.py`**: `ScoreEvalCallback` evalúa con el **harness oficial** cada
  250k pasos y guarda `best_model` por **score** (no reward); `ShapingAnnealCallback` empuja
  el paso global para anealar el shaped reward; `SB3PolicyStudent` adapta la política a
  `act(obs)->int`. Se guarda/restaura el RNG global alrededor del eval para no perturbar el entreno.
- **`training/configs/esc1.yaml`**: hiperparámetros base del PLAN + `checkpoint_freq`.
- **`colab/run_all.ipynb`**: instalar→reiniciar→GATE 0→Drive→config→entrenar→**evaluar GATE 3**→curva.

### Decisiones registradas
- **`n_steps=1024`** (el PLAN no lo fija): buffer = 1024·8 = 8192, divisible por batch 2048
  (4 minibatches). clip_range/vf_coef/max_grad_norm en defaults de SB3.
- **`tensorboard` opcional**: `train_ppo` lo detecta; se añadió a `environment.yml`.
- **`device: auto`**: MlpPolicy(96) es diminuta; la GPU aporta poco (el cuello de botella es
  el step del entorno, CPU). El paralelismo `n_envs` es lo que manda.
- **Bug corregido (segfault):** `CheckpointAgent` cargaba `PPO.load` dentro de `action()`,
  que corre bajo el `SafeActionWrapper` (SIGALRM 100 ms); la carga tarda más → SIGALRM
  interrumpe una llamada C de torch → **segfault**. Fix: cargar en `bind_env`, fuera del timer.
  Guardado con `tests/test_train_pipeline.py`.

### Validación en CPU (pipeline, NO es el entreno real)
- Smoke `--smoke` (2000 pasos, 2 envs, dummy, CPU): corre entero, `ScoreEvalCallback` llama
  al harness, guarda `best_model.zip`/`last_model.zip`/`eval_history.json`/`config_used.yaml`.
- Suite: **31 passed** (incluye regresión del segfault del checkpoint).

### GATE 3: ✅ PASA (entreno real en máquina con GPU, cramped_room, 5e6 pasos)
Artefactos en `esc1/esc1_greedy_cramped_room/` (commiteados). Verificado con el harness
oficial en esta máquina (CPU):

| | soups_mean | score_mean | timeouts |
|---|---|---|---|
| **best_model vs greedy** | **4.0** | **40365.5** | 0 |
| baseline greedy vs greedy | 0.0 | 0.0 | 0 |

Los 6 intentos (3 seeds × 2 roles) entregan 4 sopas cada uno; role-swap simétrico
(40371/40360). Curva de entreno: 1→3→4 sopas, convergió hacia ~3.5M pasos.

---

## Etapa 4 — Robustez Esc. 2-3-4 — CÓDIGO LISTO, entrenos pendientes (2026-07-12)

Finetune/entreno para tres variantes robustas. Entrenos reales en GPU/Colab; aquí se
preparó y validó el cableado en CPU.

- **Finetune en `train_ppo.py`**: `init_from` (config) / `--init-from` (CLI) →
  `load_policy_weights()` construye un PPO fresco con los hiperparámetros del escenario
  (optimizador y lr_schedule nuevos) y transfiere SOLO los pesos de la política del
  checkpoint. Requiere MISMA arquitectura (net_arch/activation). El annealing del shaping
  cuenta desde 0 sobre el horizonte del finetune (PPO fresco, no continúa timesteps).
- **`training/configs/esc2.yaml`** — finetune desde `esc1/.../best_model` vs
  `greedy+sticky_p[0,0.4]`, 2e6 pasos, `ent_coef=0.01`. Eval GATE 4 vs greedy+sticky (sopas≥2).
- **`training/configs/esc3.yaml`** — finetune (por defecto desde esc1 best, robusto; opción
  de cadena esc2→esc3 vía `--init-from`) vs `greedy+sticky[0,0.4]+eps[0,0.4]`, 2e6, ent 0.01.
- **`training/configs/esc4.yaml`** — DESDE CERO (opción warm-start desde esc3) vs mezcla
  `random_motion(0.7)/stay(0.3)`, 5e6, `ent_coef=0.05` (explorar para cocinar solo).
  Eval GATE 4 vs random_motion (sopas≥1).
- **`colab/run_all.ipynb`** generalizado: variable `CONFIG` (esc1..esc4) + `INIT_FROM`,
  evalúa vs compañero del escenario y vs greedy limpio (regresión).

### Decisiones registradas
- **`ent_coef=0.01` en esc2/esc3** (baja desde 0.05): explícitamente sancionado por el PLAN
  (Etapa 3: "bajar a 0.01 si la política ya entrega sopas"). El agente de Etapa 3 ya entrega.
- **esc3 finetunea desde esc1 por defecto** (no desde esc2): evita depender de la salida
  efímera de esc2 en `outputs/` (gitignored). Cadena esc1→esc2→esc3 disponible vía `--init-from`.
- **esc4 desde cero**: random_motion/stay son compañeros inútiles → el agente debe cocinar
  solo; empezar desde cero evita arrastrar la dependencia del compañero greedy.

### Validación en CPU
- Smoke `esc2 --smoke` con `init_from` al best_model REAL de Etapa 3: carga pesos OK y el
  agente ya entrega 2-3.5 sopas vs sticky desde el primer eval (hereda la competencia).
- Suite: **32 passed** (nuevo test: `load_policy_weights` transfiere pesos exactos).

**GATE 4: PENDIENTE** — requiere los entrenos reales y luego `sopas_mean ≥ 2` vs greedy+sticky
(esc2/esc3) y `sopas_mean ≥ 1` vs random_motion (esc4), sin regresión catastrófica vs greedy limpio.
