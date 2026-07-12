# PLAN MAESTRO — Agente RL para Overcooked-AI (Competencia del curso)

> **Documento de ejecución para Claude Code.** Este plan es prescriptivo. Claude Code debe
> ejecutarlo etapa por etapa, EN ORDEN, y **solo puede avanzar a la etapa N+1 si el GATE de
> la etapa N pasa**. Si un gate falla, debe arreglar la etapa actual, nunca saltarla ni
> improvisar una alternativa no listada en este documento.

---

## 0. REGLAS INVIOLABLES (leer antes de escribir cualquier línea de código)

1. **Stack único de entrenamiento:** Python ≥3.10, `numpy<2`, `overcooked-ai` (PyPI),
   `stable-baselines3>=2.0`, `torch`, `gymnasium`. NADA MÁS para el camino principal.
2. **PROHIBIDO instalar o clonar** PantheonRL, human_aware_rl, COLE-Platform, ZSC-Eval o
   cualquier repo con `gym==0.21/0.22`, Python 3.7 o TensorFlow 1. Está verificado que NO
   compilan en Python ≥3.11. Solo se toman sus *ideas* (patrón ego/alt, FCP, HSP,
   reward shaping), reimplementadas desde cero en SB3.
3. **JaxMARL es SOLO plan B** (Etapa 8, opcional). No tocarlo antes.
4. **El entorno de evaluación es la plantilla del profesor** (carpeta `overcooked_template/`
   con `src/`, `policies/`, `configs/`). Es la fuente de verdad. No modificar sus archivos;
   solo importarlos.
5. **Parámetros fijos de la competencia (no cambiar jamás):**
   - `old_dynamics: true`, `horizon: 250`
   - Observación del agente entregable: `featurized` → vector `(96,)` por agente
     (dict `{"obs": np.ndarray, "agent_index": int}` cuando `include_agent_index: true`)
   - Acciones: `0=norte, 1=sur, 2=este, 3=oeste, 4=stay, 5=interact`
   - Límite por acción: **100 ms** (timeout → stay + penalización)
   - `swap_agent_positions: true` → el agente DEBE funcionar como índice 0 y como índice 1
   - Score oficial: `10000·sopas + 10·(horizon − t_última_sopa) + (horizon − t_primera_sopa) − min(100·timeouts, 5000)`;
     0 sopas → score 0
6. **`pip install "numpy<2"` SIEMPRE va primero** (antes de importar cualquier cosa). El
   paquete `overcooked-ai` de PyPI usa `np.Inf` y rompe con NumPy 2.x. En Colab, tras
   instalarlo, **reiniciar el runtime** antes de importar.
7. **El entregable final** es exactamente: un archivo `student_agent.py` con la clase
   `StudentAgent` (`__init__(self, config)`, `reset(self)`, `act(self, obs) -> int`) + sus
   archivos de pesos. Inferencia en CPU con torch. Ante CUALQUIER excepción interna,
   `act()` devuelve `4` (stay). Nunca lanzar excepciones hacia el runner.
8. **Toda decisión de "¿está listo?" se toma con el harness de score oficial** (Etapa 2),
   nunca con la reward de entrenamiento.
9. Claude Code **no inventa etapas nuevas, no cambia hiperparámetros base sin registrar el
   motivo en `EXPERIMENTS.md`, y no elimina gates**.
10. Cada etapa termina con un commit git: `git commit -m "Etapa N completada: <resumen>"`.

---

## 1. ESTRUCTURA DEL REPO (crear en Etapa 0, no desviarse de ella)

```
overcooked-agent/
├── PLAN.md                      # este archivo
├── EXPERIMENTS.md               # log de corridas: fecha, config, score del harness
├── README.md                    # instrucciones de reproducción (ambas vertientes)
├── requirements.txt             # numpy<2, overcooked-ai, stable-baselines3, torch, gymnasium, pyyaml, tqdm
├── environment.yml              # entorno micromamba/conda reproducible (vertiente local preferida)
├── setup.sh                     # check+install de dependencias EXTERNAS a Python (git, micromamba, GPU, etc.)
├── overcooked_template/         # plantilla del profesor, COPIADA TAL CUAL, intocable
├── envs/
│   ├── __init__.py
│   ├── ego_env.py               # gymnasium.Env ego/alt (patrón PantheonRL reimplementado)
│   ├── partners.py               # registro de compañeros: scripted (del zip) + wrappers + aprendidos
│   └── reward_shaping.py        # shaping denso + annealing (receta human_aware_rl)
├── training/
│   ├── train_ppo.py             # CLI: entrena PPO vs un compañero o mezcla
│   ├── train_selfplay_pop.py    # Etapa 5: población FCP (multi-seed + checkpoints)
│   ├── train_biased.py          # Etapa 6: compañeros HSP con rewards sesgadas
│   └── configs/                 # un .yaml por experimento
├── evaluation/
│   ├── harness.py               # score oficial: 3 seeds × 2 roles, promedio
│   └── selfcheck.py             # valida el entregable con la plantilla del profe
├── deliverable/
│   ├── student_agent.py         # LA clase a entregar
│   └── weights/                 # <layout>.pt por escenario
├── colab/
│   └── run_all.ipynb            # vertiente Colab: pipeline completo por celdas
└── scripts/
    ├── setup_local.sh           # vertiente local
    └── setup_colab.sh           # pip installs para Colab (con aviso de reinicio)
```

---

## 2. LAS DOS VERTIENTES

Ambas comparten el 100% del código del repo. Solo cambia el bootstrap y dónde persisten
los artefactos. Claude Code implementa AMBAS; el usuario elige cuál correr.

### Vertiente A — Colab (y derivados: Kaggle, Lightning, etc.)

- `scripts/setup_colab.sh`:
  ```bash
  pip install -q "numpy<2" overcooked-ai stable-baselines3 gymnasium pyyaml tqdm
  ```
  Tras esto el notebook debe imprimir en negrita: **"REINICIA EL RUNTIME AHORA
  (Entorno de ejecución → Reiniciar) y vuelve a correr desde la celda 2"**. Motivo:
  Colab precarga numpy 2 y el downgrade solo aplica tras reinicio.
- `colab/run_all.ipynb` con esta secuencia exacta de celdas:
  1. Clonar el repo (o montar Drive donde ya esté) + `setup_colab.sh` + aviso de reinicio.
  2. Verificación de entorno (mismo test del GATE 0).
  3. Configurar layout objetivo (variable `LAYOUT`).
  4. Entrenamiento (llama a `training/train_ppo.py` vía `!python`).
  5. Evaluación con `evaluation/harness.py`.
  6. Exportar pesos a Drive (`/content/drive/MyDrive/overcooked_ckpts/`) — **obligatorio**,
     Colab pierde el disco al desconectar.
  7. Autotest del entregable (`evaluation/selfcheck.py`).
- Reglas Colab: guardar checkpoint cada ≤10 min de entrenamiento (Colab free se desconecta);
  todos los paths salen de una sola variable `BASE_DIR`; `device="cuda"` si disponible.

### Vertiente B — Local

**Camino preferido: micromamba** (entorno reproducible con Python fijado en 3.10, la misma
versión de la plantilla del profesor). El flujo local empieza SIEMPRE por `setup.sh`, que
comprueba e instala las dependencias externas a Python (git/curl/tar/bzip2/unzip, micromamba
en espacio de usuario sin sudo, detección de GPU y de entorno headless), y luego crea el
entorno desde `environment.yml`:

  ```bash
  ./setup.sh --check    # solo diagnóstico: reporta qué falta, no instala nada (exit 1 si falta algo)
  ./setup.sh            # instala lo que falte + crea el entorno 'overcooked-agent'
  ./setup.sh --cpu      # igual, pero con torch solo-CPU (máquinas sin GPU NVIDIA)
  micromamba activate overcooked-agent
  pytest tests/test_env_smoke.py        # GATE 0
  ```

  Notas para Claude Code sobre `setup.sh`:
  - Si `setup.sh` corre dentro de Colab/Kaggle, se auto-detecta y redirige a
    `scripts/setup_colab.sh` (en esas plataformas NO se usa micromamba).
  - El GATE 0 no puede ejecutarse hasta que `./setup.sh --check` devuelva exit 0.
  - No modificar la lógica de detección/instalación; si un caso no está cubierto,
    documentarlo en `EXPERIMENTS.md` y preguntar al usuario (regla 5 de imprevistos).

  Reglas para Claude Code sobre `environment.yml`:
  - NO cambiar `python=3.10` ni `numpy=1.26.*` bajo ninguna circunstancia.
  - Si un paquete nuevo hace falta, agregarlo al yml (no instalar suelto con pip) y
    registrar el motivo en `EXPERIMENTS.md`.
  - Tras el primer GATE 0 en verde, generar el lockfile exacto y commitearlo:
    `pip freeze > requirements.lock`.

**Camino alternativo: venv + pip** (si no hay micromamba/conda disponible) —
`scripts/setup_local.sh`:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -U pip
  pip install "numpy<2" overcooked-ai stable-baselines3 gymnasium pyyaml tqdm
  # torch: CPU por defecto; si hay GPU NVIDIA, instalar la variante cu-* correspondiente
  ```
- Todo se corre por CLI: `python training/train_ppo.py --config training/configs/esc1.yaml`.
- Nota Windows: el timeout del profesor usa `SIGALRM` (solo Unix). El entrenamiento local
  en Windows funciona, pero el autotest de timeouts (Etapa 7) debe correrse en Linux/WSL/Colab.
- `n_envs` para vectorización: `min(8, os.cpu_count())` local; 8 en Colab.

---

## 3. ETAPAS Y GATES

> Formato de cada etapa: **Objetivo → Tareas → GATE (criterio binario de avance)**.
> Claude Code ejecuta el gate como script/test real, no como inspección visual.

---

### ETAPA 0 — Bootstrap y verificación del entorno

**Objetivo:** repo creado, dependencias instaladas, plantilla del profesor funcionando.

**Tareas:**
1. Crear la estructura de carpetas de la sección 1. `git init`.
2. Copiar `overcooked_template/` del zip del profesor dentro del repo (intocable).
3. Copiar `setup.sh` y `environment.yml` provistos (no modificar la lógica ni los pines);
   escribir `requirements.txt`, `setup_local.sh` (alternativo), `setup_colab.sh`.
   En local, ejecutar `./setup.sh` y verificar que `./setup.sh --check` devuelve exit 0
   antes de continuar.
4. Escribir `tests/test_env_smoke.py` que:
   - importa `overcooked_ai_py`, construye `cramped_room` con `old_dynamics=True`, horizon 250;
   - verifica `featurize_state_mdp` → shape `(96,)` y `lossless_state_encoding_mdp` → shape `(H,W,26)`;
   - corre un episodio completo `GreedyFullTaskPolicy` (del zip) + `RandomMotionPolicy` y
     verifica sparse reward > 0;
   - importa `stable_baselines3` y `torch` sin error.

**GATE 0:** `pytest tests/test_env_smoke.py` pasa en verde. Si falla por numpy → verificar
que `numpy<2` esté activo. **No escribir código de RL hasta que este gate pase.**

---

### ETAPA 1 — Entorno de entrenamiento ego/alt

**Objetivo:** un `gymnasium.Env` de un solo agente donde el compañero vive dentro del `step()`.

**Tareas:**
1. `envs/ego_env.py` → clase `OvercookedEgoEnv(gymnasium.Env)`:
   - `__init__(layout_name_or_file, partner_factory, horizon=250, shaping_schedule=None, randomize_index=True)`
   - `reset()`: crea el env de la plantilla (`src.environment.build_env`), sortea
     `ego_index ∈ {0,1}` si `randomize_index` (cubre el role swap), instancia el compañero
     vía `partner_factory()` y le hace `set_mdp` + `set_agent_index`.
   - `step(a_ego)`: obtiene `a_alt = partner.action(state)`, arma la tupla ordenada por
     índice, hace `env.step`, devuelve obs featurized del ego, reward
     (sparse + shaped según schedule), `terminated/truncated`, info con
     `sparse_reward` y `soups_delivered`.
   - Observación: `Box(shape=(96,), dtype=float32)`; Acción: `Discrete(6)`.
2. `envs/partners.py` → `make_partner(spec: dict) -> Agent`, con specs:
   - `{"type":"greedy"}` → `GreedyFullTaskPolicy` del zip
   - `{"type":"greedy","sticky_p":0.3}` → wrapper sticky (repite acción previa con prob p; implementarlo, el zip no lo trae)
   - `{"type":"greedy","eps":0.2}` → `EpsilonActionWrapper` del zip
   - `{"type":"random_motion"}`, `{"type":"stay"}` → del zip
   - `{"type":"checkpoint","path":...}` → política SB3 congelada usada como compañero (Etapa 5)
   - `{"type":"mixture","specs":[...],"probs":[...]}` → muestrea un spec por episodio
   - Los valores `sticky_p`/`eps` aceptan `[lo, hi]` → muestrear U(lo,hi) por episodio.
3. `envs/reward_shaping.py` → shaping por eventos (receta human_aware_rl):
   `+3` ingrediente en olla, `+3` recoger plato, `+5` recoger sopa; coeficiente que anealiza
   linealmente de 1.0 a 0.0 entre el paso 0 y `anneal_end_step` (default: 60% del total).
4. `tests/test_ego_env.py`: 3 episodios con compañero greedy y acciones random del ego
   corren sin excepción; `check_env` de gymnasium pasa; con `randomize_index=True` ambos
   índices aparecen en 20 resets.

**GATE 1:** `pytest tests/test_ego_env.py` en verde.

---

### ETAPA 2 — Harness de score oficial

**Objetivo:** medir EXACTAMENTE lo que mide el profesor. Es el árbitro de todo el proyecto.

**Tareas:**
1. `evaluation/harness.py` → función y CLI:
   `evaluate(agent_ctor, layout, partner_spec, seeds=[67,68,69], horizon=250) -> dict`
   - Por cada seed corre 2 episodios (rol normal y rol invertido), usando los wrappers del
     profesor (`SafeActionWrapper` con 100 ms) para contar timeouts reales.
   - Registra timestep de primera y última sopa entregada (detectar por sparse reward +20 del env).
   - Calcula el score oficial por intento y reporta: media, por-seed, sopas, timeouts.
2. Validarlo con pares conocidos: `greedy+greedy` debe dar sopas > 0 en `cramped_room`;
   `stay+stay` debe dar score 0.

**GATE 2:** el harness reproduce esos dos casos de control y su salida incluye
`{score_mean, soups_mean, timeouts_total}`. Registrar en `EXPERIMENTS.md` la línea base
`greedy+greedy` por layout conocido.

---

### ETAPA 3 — Agente Escenario 1 (PPO vs greedy)

**Objetivo:** primer agente entrenado que supere a greedy+greedy en el harness.

**Tareas:**
1. `training/train_ppo.py` (CLI con config YAML). Config base (NO cambiar sin anotar en
   `EXPERIMENTS.md`):
   - Política `MlpPolicy` (2×256, tanh) sobre obs `(96,)`
   - `n_envs=8` (SubprocVecEnv local / DummyVecEnv si falla en Colab)
   - `total_timesteps=5e6`, `lr=3e-4` con decaimiento lineal, `ent_coef=0.05`→ bajar a
     `0.01` si la política ya entrega sopas, `gamma=0.99`, `gae_lambda=0.95`,
     `batch_size=2048`, `n_epochs=8`
   - Shaping con annealing hasta el 60% de los pasos
   - Compañero: `{"type":"greedy"}`; `randomize_index=True`
   - `EvalCallback` propio cada 250k pasos que llama al harness y guarda `best_model.pt`
     por score oficial (no por reward).
2. Entrenar en `cramped_room` como piloto. Cuando el profe revele los layouts reales
   (domingo), re-lanzar con `--layout <nombre_o_archivo.layout>`; el pipeline debe aceptar
   layouts custom `.layout` vía la plantilla.
3. Registrar todo en `EXPERIMENTS.md`.

**GATE 3:** en el layout objetivo, `harness(best_model, partner=greedy) ≥ harness(greedy, greedy)`
y sopas_mean ≥ 1 en los 3 seeds × 2 roles. Si tras 5M de pasos no se cumple: (a) subir
`anneal_end_step` a 80%, (b) duplicar timesteps. Solo esas dos perillas, en ese orden.

---

### ETAPA 4 — Robustez Escenarios 2-3 (sticky + random) y Escenario 4 (random_motion)

**Objetivo:** tres variantes del agente, cada una validada contra el compañero exacto de su escenario.

**Tareas:**
1. **Esc. 2:** finetune del checkpoint de Etapa 3 (1-2M pasos) con compañero
   `{"type":"greedy","sticky_p":[0.0,0.4]}`.
2. **Esc. 3:** finetune con `{"type":"greedy","sticky_p":[0.0,0.4],"eps":[0.0,0.4]}`.
3. **Esc. 4:** entrenamiento desde cero (o desde Esc. 3) con
   `{"type":"mixture","specs":[{"type":"random_motion"},{"type":"stay"}],"probs":[0.7,0.3]}`
   → el agente aprende a completar el ciclo completo SOLO. Este es el requisito duro del
   escenario 4 (random_motion nunca hace interact).
4. Evaluar cada variante con el harness contra su compañero de escenario Y contra greedy
   limpio (verificar que no hubo regresión catastrófica).

**GATE 4:** sopas_mean ≥ 2 vs greedy+sticky (umbral de clasificación del Esc. 2);
sopas_mean ≥ 1 vs random_motion (umbral del Esc. 4). Registrar en `EXPERIMENTS.md`.

---

### ETAPA 5 — Población FCP (para Escenarios 5-6)

**Objetivo:** población de compañeros aprendidos, diversos en convención y en skill.
Referencia conceptual: Fictitious Co-Play (DeepMind NeurIPS 2021) — reimplementado, no clonado.

**Tareas:**
1. `training/train_selfplay_pop.py`:
   - Entrena `N=4` (mínimo; 8 si hay tiempo/GPU) corridas de PPO **self-play** con seeds
     distintos (self-play = el compañero es una copia congelada del propio agente,
     refrescada cada 100k pasos).
   - Guarda checkpoints al 25%, 50% y 100% de cada corrida → población de 12-24 compañeros.
2. Extender `partners.py` para cargar estos checkpoints como compañeros congelados
   (`{"type":"checkpoint","path":...}`) — forward CPU, sin grad.

**GATE 5:** los N agentes self-play entregan ≥1 sopa consigo mismos en el harness, y
cargar cualquier checkpoint como compañero en `OvercookedEgoEnv` corre sin error.

---

### ETAPA 6 — Compañeros sesgados (HSP-lite) + agente final robusto

**Objetivo:** compañeros con "personalidades" extremas y el agente definitivo para Esc. 5-6.
Referencia conceptual: Hidden-utility Self-Play / behavior-preferring partners de ZSC-Eval — reimplementado.

**Tareas:**
1. `training/train_biased.py`: 3 corridas cortas (1-2M pasos) de PPO self-play con shaping
   sesgado: (a) solo recompensa por cebollas en olla, (b) solo por platos/sopas,
   (c) penalización por moverse (agente semi-estático). Guardar checkpoint final de cada una.
2. Agente final: entrenar/finetunear con compañero
   `{"type":"mixture"}` sobre: población FCP (Etapa 5) + sesgados (6.1) + greedy + greedy
   ruidoso + random_motion + stay, con probs uniformes.
3. Evaluar con el harness contra CADA tipo de compañero por separado y registrar la matriz
   en `EXPERIMENTS.md`.

**GATE 6:** el agente final logra sopas_mean ≥ 2 contra ≥80% de los tipos de compañero de
la matriz (excepto `stay`, donde el criterio es ≥1: debe poder solo).

---

### ETAPA 7 — Empaquetado del entregable y autotest

**Objetivo:** `StudentAgent` listo para enviar, probado con el runner del profesor.

**Tareas:**
1. `deliverable/student_agent.py`:
   - `__init__(config)`: carga `weights/` con ruta relativa a `__file__` (torch, CPU,
     `eval()`, `torch.no_grad()`). Si `config` trae el nombre del layout, seleccionar el
     checkpoint correspondiente; si no, usar `weights/default.pt`.
   - `act(obs)`: acepta dict (`obs["obs"]`) o ndarray directo; forward; `argmax`; devuelve `int`.
   - `reset()`: no-op (o limpiar estado si se añadió memoria).
   - TODO el cuerpo de `act` en `try/except` → ante excepción devuelve `4` (stay).
   - Cero prints, cero imports pesados fuera de torch/numpy.
2. `evaluation/selfcheck.py`:
   - Ejecuta `python -m src.evaluate` de la plantilla con un `evaluate.yaml` generado que
     apunta a `deliverable/student_agent.py`, 3 seeds, swap activado, compañero greedy.
   - Mide latencia de `act()`: p99 < 20 ms en CPU (margen 5× sobre los 100 ms).
   - Verifica timeouts_total == 0 y sopas > 0.
3. `README.md` final con reproducción completa de ambas vertientes.

**GATE 7 (definitivo):** `selfcheck.py` pasa: 0 timeouts, ≥1 sopa promedio, p99 < 20 ms,
en un entorno LIMPIO (venv nuevo o runtime Colab recién reiniciado) instalando solo
`requirements.txt`. Si pasa, el entregable está listo.

---

### ETAPA 8 — (OPCIONAL, solo si Etapas 0-7 completas y falta velocidad para layouts sorpresa)

**Objetivo:** reentrenamiento exprés con JaxMARL para los layouts revelados con pocas horas.

**Condición de entrada:** GATE 7 aprobado + evidencia en `EXPERIMENTS.md` de que el
pipeline SB3 tarda demasiado para la ventana de la competencia. Si no, NO ejecutar.

**Tareas:** instalar `jaxmarl` (verificado compatible con py3.12); entrenar IPPO en el
layout; **destilar** la política a la arquitectura MLP-featurized del entregable
(recolectar (obs_featurized, acción) rodando la política en la plantilla del profe y
entrenar por imitación); validar con el harness. Si la destilación no supera al agente
SB3 existente en el harness, descartar y quedarse con SB3.

---

## 4. CRONOGRAMA DE REFERENCIA (ajustar a las fechas reales del curso)

| Momento | Etapas |
|---|---|
| Hoy | 0, 1, 2 (piloto en `cramped_room`) + Etapa 3 piloto |
| Domingo (layouts 1-3 revelados) | Etapa 3 real por layout + Etapa 4 (Esc. 2-3) |
| Lunes temprano (layout 4) | Etapa 4 (Esc. 4) + Etapa 7 (primer entregable) |
| Antes de la competencia | Etapas 5, 6 + re-empaquetado (Etapa 7 de nuevo) |
| Competencia (layouts 5-6) | Reentrenar con pipeline probado; Etapa 8 solo si hace falta |

---

## 5. QUÉ HACER ANTE IMPREVISTOS (única lista de desvíos permitidos)

- **Falla import por numpy** → confirmar `numpy<2`; en Colab, reiniciar runtime.
- **SubprocVecEnv falla en Colab** → usar `DummyVecEnv` (más lento pero estable).
- **PPO no aprende (0 sopas tras 5M)** → perillas del GATE 3, en ese orden. Nada más.
- **Layout sorpresa con formato raro** → cargarlo con `load_custom_layout_dict` de la
  plantilla; si falla, transcribir el grid a mano a un `.layout` propio.
- **Cualquier otra cosa** → detenerse, documentar en `EXPERIMENTS.md`, y preguntar al
  usuario. NO improvisar soluciones fuera de este plan.
