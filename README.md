# Overcooked-AI — Agente RL para los 3 escenarios de la competencia

Agente de aprendizaje por refuerzo (PPO) que coopera con el compañero scripted
`greedy_full_task` en los tres escenarios evaluados de la competencia. **Un único modelo**
(`deliverable/weights/default.pt`, torch puro) cubre los tres.

Basado en la plantilla de https://github.com/HumanCompatibleAI/overcooked_ai (incluida en
`overcooked/`).

## Los 3 escenarios

| # | Layout | Compañero |
|---|--------|-----------|
| 1 | `asymmetric_advantages` | `greedy_full_task` |
| 2 | `coordination_ring` | `greedy_full_task` + sticky actions |
| 3 | `counter_circuit` | `greedy_full_task` + sticky + random actions |

## Resultados (harness oficial, 3 seeds, swap activado)

| Escenario | greedy limpio | perturbación moderada | agresiva (sticky 0.4) |
|-----------|:---:|:---:|:---:|
| 1 · asymmetric | 5.5 | 5.5 | 5.5 |
| 2 · coordination | 8.0 | 5.8 | 3.4 |
| 3 · counter | 6.0–12 | 5.8 | 5.6 |

Puntuación = sopas entregadas (cada sopa = +20 sparse). Latencia `act()`: **p99 ≈ 0.07 ms**
(límite 100 ms) → 0 timeouts.

## Enfoque

- **PPO ego/alt**: un agente (ego) entrena con Stable-Baselines3 PPO contra el compañero
  embebido en el entorno (`envs/ego_env.py`). El mejor modelo se elige con el **harness
  oficial** (score de la competencia), no con la reward de entrenamiento.
- **Observación** `featurized` → vector `(96,)` por agente (constante en todos los layouts).
  Política `MlpPolicy` 2×256 tanh.
- **Reward shaping por eventos** (+3 ingrediente en olla, +3 recoger plato, +5 recoger sopa)
  anelado a 0, más **nav-shaping** (`envs/nav_shaping.py`): recompensa densa potential-based
  por acercarse al siguiente subobjetivo. Fue la pieza clave para resolver `counter_circuit`
  (anillo largo donde la reward esparsa es inalcanzable por exploración aleatoria).
- **Compañero de entrenamiento**: mezcla de `greedy_full_task` a distintos niveles de
  perturbación (sticky/random) que cubre los 3 escenarios y endurece la robustez.
- **Modelo final** (`esc_enhanced.yaml`): fine-tune del generalista contra sticky agresivo
  en los 3 layouts a la vez, para un único checkpoint robusto.

## Estructura

```
deliverable/          Entregable final (torch puro, sin SB3)
  student_agent.py       clase StudentAgent: act(obs)->int
  weights/default.pt     pesos de la política (verificados idénticos a SB3)
envs/                 Entorno ego/alt, compañeros, reward + nav shaping
training/             PPO, callbacks, configs (training/configs/*.yaml)
evaluation/           harness oficial + selfcheck (GATE 7)
scripts/              setup + export_weights (SB3 -> torch)
overcooked/           Plantilla del profesor (runner, policies, rendering)
colab/                Notebook para entrenar en Colab
```

## Reproducción

### 1. Entorno

**Vertiente micromamba (Linux/cluster/local):**
```bash
./setup.sh                       # crea el env 'overcooked' desde environment.yml
micromamba run -n overcooked python -m pytest tests/test_env_smoke.py -q   # GATE 0
```

**Vertiente pip/Colab:** ver `colab/run_all.ipynb` (instala con `numpy<2`, **reiniciar
runtime** una vez). Todas las deps también en `requirements.txt`.

> Regla inviolable: **`numpy<2`** (overcooked-ai usa `np.Inf`, removido en NumPy 2.0).
> No se necesita GPU: el cuello de botella es el CPU (env stepping). Usar `--device cpu` y
> `--n-envs` = nº de cores físicos.

### 2. Entrenar el modelo final

```bash
# Generalista base (3 layouts, nav-shaping)
micromamba run -n overcooked python -m training.train_ppo \
  --config training/configs/esc_scenarios.yaml \
  --nav-shaping-coef 0.5 --experiment-name esc_scenarios_nav \
  --device cpu --n-envs 40 --vec subproc

# Endurecido vs sticky agresivo (warm-start del anterior) -> MODELO FINAL
micromamba run -n overcooked python -m training.train_ppo \
  --config training/configs/esc_enhanced.yaml \
  --init-from esc_scenarios_nav_multi/best_model \
  --device cpu --n-envs 40 --vec subproc
```

Artefactos en `esc_enhanced_multi/`: `best_model.zip` (mejor por score oficial),
`eval_history.json`, `config_used.yaml`.

### 3. Empaquetar el entregable (SB3 → torch puro)

```bash
micromamba run -n overcooked python scripts/export_weights.py \
  --checkpoint esc_enhanced_multi/best_model \
  --out deliverable/weights/default.pt
```

Verifica que el `state_dict` extraído reproduce la política SB3 (argmax idéntico) antes de
guardarlo.

### 4. Autotest — GATE 7

```bash
micromamba run -n overcooked python -m evaluation.selfcheck
```

Corre el entregable por el runner del profesor en los 3 escenarios (vs greedy, 3 seeds,
swap) y mide la latencia. Verde si sopas ≥ 1 en cada escenario y p99 < 20 ms.

## El entregable

`deliverable/student_agent.py` es autocontenido: solo depende de `torch` y `numpy`.
Reconstruye la MLP `96→256→256→6` (tanh, argmax) y carga `weights/default.pt` con ruta
relativa a `__file__`. Todo `act()` va en `try/except` → ante cualquier fallo devuelve `4`
(stay) para nunca romper el runner. Un solo modelo para los 3 escenarios; si el grader pasa
el layout en `config` y existe `weights/<layout>.pt`, se prefiere, si no `default.pt`.
