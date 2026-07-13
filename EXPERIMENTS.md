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
  Eval GATE 4 vs random_motion (sopas≥1). **GENERALISTA MULTI-LAYOUT** (ver abajo).
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

### Esc.4 como GENERALISTA multi-layout (decisión del usuario)
La obs `featurized` es `(96,)` en TODOS los layouts (solo `lossless_grid` cambia con H×W),
así que un mismo `MlpPolicy` puede entrenar en varios layouts. Se decidió que **solo Esc.4**
(el agente que cocina solo, el más "de propósito general") sea multi-layout → sirve como
`default.pt` y hedge de layouts sorpresa. Esc.2/3 siguen per-layout (especialistas).

**Pool de Esc.4 = solo layouts SOLO-VIABLES.** Medido con greedy cocinando solo vs el
compañero real del escenario (`random_motion`, mediana de 3 seeds):

| Layout | sopas greedy-solo | ¿en el pool? |
|---|---|---|
| cramped_room             | 5 | ✅ |
| coordination_ring        | 4 | ✅ |
| counter_circuit_o_1order | 2 | ✅ |
| asymmetric_advantages    | 0 | ❌ (greedy no lo resuelve solo) |
| forced_coordination      | 0 | ❌ (imposible solo POR DISEÑO: fuerza handoff) |

**Implementación multi-layout:** `cfg['layout']` acepta lista; `train_ppo` reparte un layout
por worker del VecEnv (round-robin), pre-calienta el motion planner de cada uno, y
`ScoreEvalCallback` evalúa en CADA layout del pool eligiendo `best_model` por el score
promedio (registra el desglose por layout en `eval_history.json`). `--layout <x>` sigue
sirviendo para forzar un especialista de un solo layout. Smoke multi-layout OK (evalúa los 3).

### Nota para la Etapa 7 (empaquetado)
En el smoke se observó `timeouts=1` en el PRIMER eval: el primer `predict()` de un modelo SB3
recién cargado puede tardar >100 ms (init perezoso). En el entregable, `StudentAgent.__init__`
debe **calentar el modelo con un forward dummy** para evitar un timeout en el primer `act()`.

### GATE 4: ✅ PASA (escenarios) — entrenos reales en cramped_room (2026-07-12)
Los tres corridos en cramped_room single (esc4 en single, no multi-layout todavía).
Artefactos en `esc2/`, `esc3/`, `esc4/`. Evaluado con el harness oficial (seeds 67,68,69 ×2 roles):

| Variante | vs compañero escenario | vs greedy limpio | GATE |
|---|---|---|---|
| esc2 (finetune vs sticky)     | 3.83 sopas (≥2) | 4.00 (sin regresión) | ✅ |
| esc3 (finetune vs sticky+eps) | 3.83 sopas (≥2) | **1.50 (REGRESIÓN)** | ✅ escenario |
| esc4 (solo, desde cero)       | 5.00 vs random_motion; 2.50 vs stay | **0.00 vs greedy** | ✅ escenario |

Curvas: esc2 2.5→3.8 sopas; esc3 1.7→3.7; esc4 0→5.0 (aprendió el ciclo solo hacia ~500k).

### Hallazgos que orientan Esc 5-6
1. **esc3 regresó vs greedy limpio (4.0→1.5):** entrenar vs compañero ruidoso (eps) sacrifica
   la sinergia con un greedy competente. La robustez de-partner tiene trade-off.
2. **esc4 es solo-specialist puro:** 5.0 sopas solo, pero **0.0 vs greedy competente** (en
   cramped_room dos cocineros activos se estorban → mismo deadlock que greedy+greedy).
   Aprendió a cocinar solo, NO a cooperar.

**Conclusión:** los especialistas NO se componen en un generalista. Ningún checkpoint cubre
todos los compañeros; el finetuning de robustez sacrifica el caso limpio; el solo-specialist
falla con un compañero competente. → Refuerza el pivote a **E3T-lite**: entrenar vs una MEZCLA
de todos los tipos a la vez (greedy + sticky + eps + random + stay + copia-congelada del ego).
Como la mezcla incluye greedy limpio, arregla la regresión de esc3 de paso.

### esc4 GENERALISTA multi-layout (esc4_alt/esc4_solo_multi, n_envs=9, 5e6, desde cero)
Pool [cramped_room, coordination_ring, counter_circuit]. Harness vs random_motion:

| Layout | soups | GATE≥1 |
|---|---|---|
| cramped_room             | 4.83 | ✅ |
| coordination_ring        | 3.83 | ✅ |
| counter_circuit_o_1order | 0.00 | ❌ |

**Aprendió 2 de 3.** `counter_circuit` se quedó en 0 los 5M pasos (el más difícil solo: circuito
largo). Arrancar desde cero + 1/3 de los envs + 0 señal temprana → nunca bootstrapeó; el
best_model (por score promedio) tolera counter=0 porque cramped+coord compensan.
**Aprendizaje:** el multi-layout desde cero no cubre layouts difíciles por sí solo; necesita
warm-start desde un agente competente y/o upweight del layout difícil (curriculum). Insumo
directo para el diseño E3T (warm-start + curriculum importan).

### Diagnóstico counter_circuit solo (warm-start esc1, 16 envs enfocados, 1M, CPU)
`outputs/esc4diag/` (gitignored). Resultado matizado:
- El `ep_rew_mean` subió a ~1.1 hacia 475k pasos, pero era **shaped reward** (subtareas:
  cebollas a la olla), **NO entregas**: el harness dio **0 sopas en los 4 evals** (250k-1M).
- Colapsó a 0 hacia 640k: el shaping se aneló a 0 al 60% de 1M (=600k); como el agente nunca
  logró una entrega sparse, al apagarse el shaping su reward→0 y la política se congeló.

**Conclusión:** counter_circuit solo NO es imposible (aprende subtareas) pero es MUY difícil:
1M warm-starteado no completó ni una sopa. El ciclo largo (cargar por todo el circuito) da
sparse demasiado raro para bootstrappear, y el anneal rápido mató la guía. Nota: greedy (BFS
perfecto) hace 2 sopas ahí → **greedy-viable ≠ RL-aprendible** en horizonte largo.

**Decisión práctica:** sacar `counter_circuit` del pool solo de esc4. El generalista solo =
`[cramped_room, coordination_ring]` (ambos pasan GATE 4). counter_circuit se maneja como layout
de COOPERACIÓN (funciona con compañero competente), no solo — salvo que se invierta una corrida
GPU grande (3-5M) con anneal de shaping más lento y/o shaping por distancia (POT/SOUP_DISTANCE,
hoy en 0) para guiar el acarreo largo.

---

## Referencia: caracterización de los 45 layouts (proxy greedy, old_dynamics, h=250)
`coop` = sopas greedy+greedy (¿cooperación viable?); `solo` = sopas greedy+random_motion
(¿un cocinero solo?). Proxy con greedy (BFS perfecto): el RL puede superar (coop) o quedarse
corto (horizonte largo). Ordenado por categoría.

- **COOP+SOLO (versátiles):** marshmallow_experiment 24/12 · inverse_marshmallow 18/12 ·
  simple_o 12/7 · simple_o_t 12/7 · centre_pots 8/6 · coordination_ring 8/3 · five_by_five 6/4 ·
  scenario2 6/4 · scenario3 4/4 · counter_circuit_o_1order 3/2 · centre_objects 1/5 · scenario2_s 1/5
- **COOP-ONLY (solo=0, necesitan 2):** tutorial_0/2/3 18 · marshmallow_experiment_coordination 12 ·
  asymmetric_advantages_tomato 3 · asymmetric_advantages 1 · unident 1
- **SOLO-ONLY (greedy-pair se atasca):** mdp_test 12 · cramped_room_tomato 8 · pipeline 6 ·
  schelling_s 6 · **cramped_room 5** · cramped_room_o_3orders 5 · bottleneck 4 · large_room 4 ·
  scenario1_s 4 · schelling 4 · scenario4 3 · m_shaped_s 1
- **HARD/0 (greedy no logra ninguno):** forced_coordination(+tomato) · counter_circuit ·
  you_shall_not_pass · soup_coordination · small_corridor · corridor · cramped_corridor ·
  long_cook_time · bonus_order_test · simple_tomato · tutorial_1
- **SKIP:** cramped_room_single (1p) · multiplayer_schelling (4p)

**Insight clave:** `cramped_room` es SOLO-only (greedy+greedy=0); nuestro agente RL sí coopera
ahí (4 sopas) → el RL supera a greedy en coordinación. `counter_circuit` (plano) es HARD/0 =
layout de cooperación, no solo. Layouts tomate necesitan recipe tomate (greedy=onion no aplica).

### Pools propuestos para la fase E3T (a confirmar con el usuario)
- **Cooperación (agente robusto vs mezcla):** coordination_ring, counter_circuit_o_1order,
  five_by_five, scenario2/3, centre_pots, simple_o, marshmallow_experiment (+ cramped_room, que
  el RL coopera). Incluir algún HARD/0 (forced_coordination, counter_circuit) como reto.
- **Solo generalista (esc4):** cramped_room, coordination_ring (confirmados). Ampliable con
  cramped_room_o_3orders, large_room, bottleneck, scenario1_s.
