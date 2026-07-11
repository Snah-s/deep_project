# Planeación — Agente RL (PPO) para la competencia Overcooked-AI

> Documento de diseño. **No contiene código**: consolida el contexto, los datos explorados,
> las decisiones tomadas (priorizando plausibilidad y score) y las ideas de implementación.
> Las referencias citadas están descargadas en `docs/` (ver §12) y los repos externos en
> `external/` (se obtienen con `./setup.sh`).

---

## 0. TL;DR — Decisiones tomadas

| Dimensión | Decisión | Alternativa descartada (por qué) |
|---|---|---|
| Algoritmo principal | **PPO + E3T-lite** (compañero = mezcla ego+aleatorio) | FCP/MEP/COLE completos (población → 10-25× más cómputo); SP puro (se sobreajusta a sí mismo) |
| Plan B | **FCP-lite** (población 4-8 × 3 checkpoints, reciclando checkpoints del baseline SP) | FCP publicado (10⁹ pasos, clúster) |
| Librería | **Stable-Baselines3 (PPO)** sobre PyTorch | CleanRL (más código propio); MAPPO-RNN de ZSC-Eval (más lento, más complejo) |
| Observación | **lossless_grid + CNN** con padding a tamaño fijo | featurized+MLP (dimensión cambia por layout → mala para escenarios secretos 4-6) |
| Entrenamiento | **Multi-layout** (pool ≥15 layouts) + asiento aleatorio + shaping annealado | un modelo por layout (no cubre los escenarios 4-6) |
| Data humana | **Proxy BC para evaluación** + warm-start opcional | fuente principal del agente (frágil: 64 % de acciones son "stay") |
| Seguridad en evaluación | Pesos cargados en `__init__`, forward <1 ms, **fallback heurístico reactivo** parseando el grid | depender solo de la política (riesgo de score 0 en layouts raros) |
| Cómputo | Prototipos y corridas sueltas **en esta CPU** (~2.3 h/10M pasos); barridos y población **en la GPU 12 GB**; Colab para seeds paralelos | "todo en GPU" (innecesario: el cuello es CPU/entorno) |

---

## 1. Objetivo y reglas de la competencia

- Entregar **un único agente** (`StudentAgent` en `overcooked/policies/template.py`) con interfaz
  `__init__(config)`, `reset()`, `act(obs) → int` (acciones 0-5).
- Evaluación por escenario: **3 intentos con 3 seeds**, promedio. En algunos escenarios hay **cambio de rol**
  (`swap_agent_positions`), así que el agente debe jugar en ambos asientos.
- Escenarios 1-3 **conocidos**; escenarios 4-6 **secretos**, revelados en vivo → exige **generalización**.
- El compañero de cada ronda lo define la organización → exige **coordinación zero-shot (ZSC)**.

### Score oficial

`Score = 10000·sopas + 10·(horizon − t_última_sopa) + (horizon − t_primera_sopa) − min(100·timeouts, 5000)`

Implicaciones directas de diseño:

1. **El número de sopas domina** (×10000). Todo lo demás es desempate.
2. Si no se entrega **ninguna sopa el score es 0** → hace falta una red de seguridad que garantice ≥1 sopa
   incluso en layouts extraños (fallback heurístico, §6.6).
3. La **penalización solo castiga timeouts** (>100 ms por decisión). Nuestra inferencia medida es <1 ms
   (margen ×100), pero hay que cargar los pesos en `__init__` y "calentar" el forward en `reset()` para
   no pagar inicialización perezosa en la primera acción.

---

## 2. Estado del repo y hallazgos verificados

### 2.1 Infraestructura del proyecto
- El runner carga al alumno vía `python_class` (`src/policy_loader.py`) y lo envuelve en
  `SafeActionWrapper` (límite 100 ms con SIGALRM, acción de reemplazo "stay") — `src/policy_wrappers.py`.
- Observaciones generadas por `src/observations.py` con `featurize_state_mdp` / `lossless_state_encoding_mdp`
  **estándar del repo base** → cualquier dato guardado como estados crudos de overcooked_ai es compatible.
- `GreedyFullTaskPolicy` (baseline con BFS) existe pero el commit dice "Heuritic with bugs" → **arreglarlo**:
  sirve como compañero de entrenamiento/evaluación y como base del fallback.
- Convención de acciones verificada: `INDEX_TO_ACTION = [N, S, E, W, stay, interact]` = 0-5 del proyecto.
- `env.step` devuelve `sparse_r_by_agent` (sopa = +20) y `shaped_r_by_agent`
  (cebolla en olla +3, plato +3, recoger sopa +5) → todo lo necesario para el reward de PPO.

### 2.2 Data humana validada (ya instalada con el paquete)
En `site-packages/overcooked_ai_py/data/human_data/`: `clean_train_trials.pickle` (46 729 pasos) y
`clean_test_trials.pickle` (44 373). DataFrames con `state` crudo + `joint_action` + `layout_name`.

| Layout | ¿Escenario? | Trayectorias (train) | Sopas/partida | Pasos/partida |
|---|---|---|---|---|
| cramped_room | conocido | 8 | 4.38 | ~1195 |
| asymmetric_advantages | conocido | 9 | 6.50 | ~1196 |
| coordination_ring | conocido | 8 | 3.81 | ~1202 |
| random0 (= counter_circuit) | oficial | 6 | 4.21 | ~1192 |
| random3 (= forced_coordination) | oficial | 8 | 3.00 | ~1203 |

Hallazgos que condicionan el diseño:
- **64 % de las acciones humanas son "stay"** → una BC ingenua aprende pasividad; hay que filtrar o
  submuestrear los frames de inactividad al construir el dataset BC.
- Partidas de ~1200 pasos vs. horizonte de evaluación 250-400 → segmentar; la "vara humana" ≈ **1-2 sopas por
  250-400 pasos**.
- No hay data de los escenarios secretos → la data sirve para proxy/warm-start, **no** resuelve generalización.

### 2.3 Layouts disponibles (pool de entrenamiento)
- **45** instalados con el paquete (incluye los 14 del README).
- **+60** en `external/ZSC-Eval/zsceval/envs/overcooked_new/.../layouts` con **el mismo esquema** del proyecto
  (verificado cargando `corner_onion_tomato`, `many_orders`, `distant_tomato` con el loader del repo);
  15 no están en el paquete. Incluyen multi-receta (cebolla+tomate) → más diversidad.
- **+179** en `icaros-usc/overcooked_env_gen` (78 GAN-small + 78 GAN-large + 23 base) en **esquema viejo**
  (`start_order_list`, `cook_time`) → conversión trivial: extraer el `grid` y reenvolverlo en el esquema nuevo.
  Ese repo además trae **generador procedural** (DCGAN + Quality-Diversity/CMA-ME) si necesitamos diversidad infinita.
- `external/ZSC-Eval` también publica un **zoo de agentes preentrenados** (HuggingFace, vía LFS) útiles como
  compañeros de evaluación held-out.

### 2.4 Benchmarks medidos en esta máquina (i7-1260P, 16 hilos, sin GPU)
- `env.step` puro: **~8 800 pasos/s** (1 proceso); con lossless encoding: **~3 900 pasos/s**.
- CNN pequeña (3 conv 3×3 + cabeza): forward **~55 k muestras/s**, update **~19.7 k muestras/s** (8 hilos).
- Corrida E3T single-task (10M pasos, ppo_epoch 15): **~2.3 h/seed** — el costo dominante son los updates de
  PPO, no el entorno. Con la GPU de 12 GB se estima 0.5-1.5 h/seed (×2-3, no ×100: las redes son diminutas).
- **Inferencia** del agente final: <1 ms por decisión → sobra margen contra el límite de 100 ms; la evaluación
  del torneo puede correr en cualquier CPU.

---

## 3. Encuadre del problema

La combinación "compañero desconocido + cambio de rol + layouts secretos" es exactamente
**Zero-Shot Coordination (ZSC) + generalización cross-layout**:

- **Carroll et al. 2019** demostraron que self-play/PBT coordinan bien consigo mismos pero fallan con
  compañeros distintos (humanos o políticas ajenas) → SP puro solo como baseline.
- La generalización a **layouts nunca vistos** es un problema abierto (OGC 2024/2025): casi toda la literatura
  entrena un modelo **por layout**. Es nuestro mayor riesgo y lo atacamos con entrenamiento multi-layout +
  arquitectura tolerante a tamaño variable + fallback heurístico.

---

## 4. Métodos de referencia: puntajes, estrategias y hardware

Puntajes con compañero experto, horizonte 400 (~20 pts/sopa), de la tabla de COLE / ZSC-Eval:

| Layout | COLE | FCP | MEP | Self-Play |
|---|---|---|---|---|
| Cramped Room | **212.8** | 207.2 | 196.8 | 165.7 |
| Coordination Ring | **166.3** | 144.0 | 124.7 | 133.3 |
| Counter Circuit | **105.8** | 46.5 | 76.7 | 65.7 |

Con proxy humano (paper FCP): FCP 10.6 sopas · BCP 7.5 · SP ~5.0.

| Método | Estrategia | Cómputo/hardware reportado | ¿Viable aquí? |
|---|---|---|---|
| **FCP** (NeurIPS 2021) | Población de 32 agentes SP × 3 checkpoints → best-response contra todos | **1 GPU por agente**, N×200 envs, **10⁹ pasos**, 3-8 días (clúster) | Solo la versión reducida |
| **MEP** (AAAI 2023) | Población 5 (→15 con checkpoints) con bonus de entropía poblacional | 1.1×10⁷ pasos/agente, 50 envs; no reporta GPU concreta | Sí (~1 día en 12 GB) |
| **COLE** (2023/24) | Grafo de estrategias + valor de Shapley (incompatibilidad cooperativa) | No reporta hardware; población iterativa compleja | Posible pero caro/complejo |
| **E3T** (NeurIPS 2023) | **Sin población**: compañero = mezcla (política ego + política aleatoria) + módulo de predicción del compañero | En ZSC-Eval: **10⁷ pasos, 100 envs CPU, episode_length 400, ppo_epoch 15, 1 GPU** | **Sí — el punto dulce** |
| **ZSC-Eval** (NeurIPS 2024 D&B) | Benchmark que implementa todos los anteriores sobre MAPPO recurrente | 50-125 envs, 10⁷ pasos, single GPU | Es nuestro perfil de referencia |

**Por qué E3T-lite como apuesta principal (plausibilidad × score):** rendimiento comparable o superior a los
métodos con población en Overcooked, con ~½ del cómputo y sin mantener una población; su presupuesto (10M pasos)
está medido y cabe en nuestra CPU (~2.3 h) y sobra en la GPU. FCP-lite queda como plan B porque su etapa 1
(entrenar varios SP con seeds distintos) la produce de todos modos el baseline, y ensamblar la población de
checkpoints es barato.

---

## 5. Estrategia por fases (cada fase deja evidencia para el informe)

**F0. Preparación** — arreglar los bugs de `GreedyFullTaskPolicy`; consolidar el pool de layouts
(copiar los 15 nuevos de ZSC-Eval + convertir un subconjunto de env_gen).

**F1. Pipeline + baseline SP-PPO** — wrapper Gym single-agent, CNN, PPO de SB3, self-play en `cramped_room`.
Criterio de éxito: ≥3 sopas/episodio (horizonte 400) y curvas estables. Sirve de sanity-check y de fuente de
checkpoints para FCP-lite.

**F2. Proxy humano (BC)** — dataset (obs lossless, acción) reproduciendo los estados de la data humana por el
encoder del proyecto; filtrar/submuestrear "stay"; entrenar BC pequeña. Uso doble: compañero de evaluación
held-out (como en los papers) y warm-start opcional del torso de la CNN. Baseline extra: **PPO_BC**
(PPO con el proxy BC como compañero, receta de Carroll 2019).

**F3. E3T-lite single-layout** — compañero = mezcla: en cada paso del compañero, con probabilidad ε acción
aleatoria, si no, la política ego (copia congelada reciente). ε≈0.5 según el paper; annealing de entropía tipo
ZSC-Eval (0.2 → 0.05 → 0.01). Presupuesto: 10M pasos/layout. Criterio: superar a SP-PPO al emparejar con
compañeros held-out (BC-proxy, greedy, SP de otra seed).

**F4. Multi-layout + generalización** — mismo E3T-lite muestreando layout por episodio del pool (3 conocidos +
12-20 diversos); padding del grid a tamaño máximo; asiento aleatorio por episodio. Validación en layouts
**held-out** (nunca entrenados) para estimar el comportamiento en los escenarios 4-6.
Si el tiempo lo permite: módulo de predicción del compañero (E3T completo) o FCP-lite como refuerzo.

**F5. Endurecimiento y entrega** — `StudentAgent` final (carga de pesos + fallback), evaluación oficial con
`configs/evaluate.yaml` (3 seeds × ambos roles × varios compañeros), ablaciones para el informe.

---

## 6. Ideas de implementación (sin código)

### 6.1 Wrapper de entrenamiento (Gym)
- Envolver `OvercookedEnv` como entorno single-agent Gymnasium: PPO controla un asiento; el otro lo llena un
  "PartnerController" intercambiable (stay / random / greedy / BC / copia congelada del ego para E3T).
- Observación: lossless encoding del asiento controlado, canal-primero, padded (§6.3).
- Recompensa: sparse **de equipo** (suma de ambos agentes; la sopa vale 20 la entregue quien la entregue)
  + coeficiente × shaped del propio agente, con el coeficiente **annealado a 0** hacia el ~50-60 % del
  entrenamiento (receta de Carroll/ZSC-Eval, shaping los primeros ~5M pasos).
- Episodios: `horizon` 400 para entrenar (estándar de la literatura); validar también a 250 porque
  `evaluate.yaml` usa 250 (ver riesgos, §8).
- Asiento y layout se sortean en cada `reset` (fases F4+).

### 6.2 Compañero E3T-lite
- Mantener una copia congelada de la política ego, refrescada cada K updates (evita perseguirse la cola).
- En cada paso del compañero: con probabilidad ε usar acción uniforme, si no, la copia congelada.
- ε fijo ≈0.5 como en el paper; opcionalmente barrer {0.3, 0.5, 0.7} en la GPU.
- Extensión opcional (E3T completo): cabeza auxiliar que predice la acción del compañero a partir del
  historial; se concatena al torso. Solo si F4 va sobrada de tiempo.

### 6.3 Red y tamaño variable de layouts
- CNN pequeña: 3 convoluciones 3×3 (25-64 filtros) + cabezas política/valor. Del orden de 10⁵ parámetros —
  entrena y evalúa en CPU sin esfuerzo.
- **Padding**: rellenar el grid con canal "pared" hasta un tamaño fijo generoso (p. ej. 15×10, mayor que todo
  el pool y con margen para los secretos). Alternativa si molesta: red totalmente convolucional + pooling
  global. El padding es más simple con SB3 y es lo que usa la literatura de generalización.
- El lossless encoding es **egocéntrico por asiento** (canales "yo"/"otro") → la misma red juega ambos roles.

### 6.4 Dataset BC (proxy humano)
- Reconstruir cada `state` del pickle humano, pasarlo por el encoder lossless del proyecto, emparejar con la
  acción del jugador correspondiente (cada trayectoria conjunta da 2 single-agent).
- Filtrado: descartar/submuestrear frames "stay" hasta ~25-30 % del dataset; separar por layout.
- Entrenar BC con la misma arquitectura CNN (permite reutilizar el torso como warm-start de PPO).

### 6.5 Entrenamiento (hiperparámetros de partida)
- Los de ZSC-Eval/Carroll como referencia: episode_length 400, ~10M pasos por configuración, ppo_epoch 8-15,
  minibatch 1-4, lr ~2.5e-4 con clip 0.2, GAE λ 0.95, γ 0.99, entropía annealada 0.2→0.01.
- Envs paralelos: 16-32 en esta CPU; 64-128 en la máquina de 12 GB (el límite es núcleos, no VRAM).
- Checkpointing periódico (sirve para FCP-lite y para elegir el mejor por validación, no el último).

### 6.6 `StudentAgent` final (el único entregable de código de agente)
- `__init__`: cargar pesos desde archivo junto al .py; construir la red; **forward de calentamiento**.
- `act(obs)`: extraer el array del dict, padear al tamaño de entrenamiento, forward determinista, devolver int.
- **Fallback en capas**: (1) try/except → "stay" jamás rompe; (2) watchdog de progreso: si en N pasos no hay
  cambio útil (política ciclando), conmutar a heurística reactiva; (3) la heurística parsea el lossless grid
  (los canales identifican ollas, dispensadores, entrega, objetos y jugadores) y aplica las reglas del greedy
  arreglado: entregar si llevo sopa → recoger si hay sopa lista → cargar olla → buscar ingrediente.
  Esto protege el peor caso (score 0) en layouts muy fuera de distribución.
- Verificar el tiempo de decisión end-to-end con el runner real (margen esperado ×100 frente a 100 ms).

### 6.7 Protocolo de evaluación interna
- Compañeros held-out: greedy arreglado, BC-proxy, SP de otra seed, stay y random (los dos últimos detectan
  dependencia del compañero: un buen agente debe poder completar sopas solo).
- 3 seeds × ambos asientos × {layouts conocidos + layouts held-out no entrenados}.
- Reportar sopas/episodio y el **score oficial** con su fórmula (incluyendo tiempos de primera/última sopa).
- Selección de modelo por promedio en held-out, no por reward de entrenamiento.

---

## 7. Plan de cómputo

| Tarea | Dónde | Estimación |
|---|---|---|
| Desarrollo, smoke tests, BC, evaluaciones | Esta CPU (i7-1260P) | minutos-horas |
| E3T-lite single-layout, 1 seed | Esta CPU | ~2.3-4 h (ojo throttling: es CPU móvil) |
| E3T-lite multi-layout, 3 seeds | GPU 12 GB (o noches en CPU) | ~½-1 día |
| FCP-lite (población 4-8) / barridos de ε | GPU 12 GB | 1-2 días |
| Seeds/experimentos en paralelo | Colab (15 GB) | checkpoints a Drive (se desconecta) |

Regla general: **la VRAM nunca es el límite** (redes ~10⁵ params); el lever es núcleos de CPU (envs paralelos)
y horas de pared. La GPU da ×2-3 en los updates de PPO.

---

## 8. Riesgos y mitigaciones

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| Escenarios 4-6 muy fuera de distribución | Media | Alto (score 0) | Multi-layout + padding generoso + fallback heurístico (§6.6) |
| Sobreajuste al compañero de entrenamiento | Alta si SP | Alto | E3T-lite (mezcla) + evaluación con pool held-out |
| Config oficial distinta a la asumida (obs type, horizonte 250 vs 400, old_dynamics) | Media | Medio | Confirmar con la cátedra; validar a 250 y 400; el agente no depende del horizonte |
| Bugs del greedy contaminan la evaluación (compañero/fallback) | Alta (ya conocidos) | Medio | Arreglarlo en F0 |
| Regresión de entorno (numpy 2 rompe overcooked_ai) | Ya ocurrió una vez | Medio | `environment.yml` pinea 1.26.4; `setup.sh` verifica imports al final |
| Throttling térmico en corridas largas locales | Media | Bajo | Corridas largas en la GPU/Colab; checkpoints frecuentes |
| Colab se desconecta | Alta | Bajo | Guardar checkpoints a Drive cada N updates |

---

## 9. Cronograma (sprint de 2 días)

| Día | Fases | Hito verificable |
|---|---|---|
| 1 | F0-F3 | Greedy arreglado; pipeline validado con SP-PPO (smoke test); BC-proxy entrenado; E3T-lite single-layout lanzado (~2.3 h/seed en CPU, corridas nocturnas si hace falta) |
| 2 | F4-F5 | Multi-layout en la GPU (3 seeds en paralelo); evaluación held-out; `StudentAgent` final con fallback y tabla de scores oficiales |

Las fases F2/F3 y las corridas largas se solapan: mientras entrena una configuración se implementa/evalúa la
siguiente. Todo lo que exceda las ~4 h de pared se delega a la máquina de 12 GB o a Colab.

---

## 10. Métricas de éxito del proyecto

1. **Primario**: sopas/episodio promedio (3 seeds × ambos roles) con compañeros held-out, en layouts
   conocidos y held-out. Traducción directa al score oficial (×10000).
2. Secundario: t de primera/última sopa (desempates del score), 0 timeouts.
3. Comparativas para el informe: SP-PPO vs PPO_BC vs E3T-lite (misma semilla de evaluación), con/sin
   multi-layout, con/sin warm-start BC.

---

## 11. Estructura futura del código (cuando se autorice implementar)

Módulos previstos, todos dentro de `overcooked/`:
- `src/rl/` → wrapper Gym, controlador de compañeros, pool/carga/padding de layouts, entrenamiento PPO,
  construcción del dataset BC, evaluación batch y cálculo del score oficial.
- `policies/` → `StudentAgent` final (red + fallback) con archivo de pesos adjunto.
- `configs/` → variantes de entrenamiento/evaluación; carpeta de layouts consolidada.
- `external/` (ignorada por git) → ZSC-Eval y overcooked_env_gen clonados por `setup.sh`.

---

## 12. Referencias

Papers (PDFs en `docs/`, descargables con `./setup.sh`):

| Archivo | Referencia | Aporte |
|---|---|---|
| `2019_carroll_utility_of_learning_about_humans.pdf` | Carroll et al., NeurIPS 2019 (arXiv:1910.05789) | Entorno, PPO+shaping, BC, PPO_BC; SP falla con humanos |
| `2021_knott_robustness_collaborative_agents.pdf` | Knott et al., 2021 (arXiv:2101.05507) | Robustez; 4 layouts extra con data humana MTurk |
| `2021_strouse_fcp.pdf` | Strouse et al., NeurIPS 2021 (arXiv:2110.08176) | FCP: población+checkpoints, ZSC sin data humana |
| `2022_zhao_mep.pdf` | Zhao et al., AAAI 2023 (arXiv:2112.11701) | MEP: diversidad por entropía poblacional |
| `2023_li_cole.pdf` | Li et al., ICML 2023 (arXiv:2302.04831) | COLE: mejores puntajes publicados por layout |
| `2023_yan_e3t.pdf` | Yan et al., NeurIPS 2023 | **E3T: método elegido** (mezcla ego+random, sin población) |
| `2024_wang_zsc_eval.pdf` | Wang et al., NeurIPS 2024 D&B (arXiv:2310.05208) | Benchmark ZSC; hiperparámetros de referencia |
| `2025_ruhdorfer_ogc.pdf` | Ruhdorfer et al., TMLR 2025 (arXiv:2406.17949) | Generalización a layouts nuevos (UED) |

Repos externos (en `external/`, clonados por `./setup.sh`):
- `sjtu-marl/ZSC-Eval` — 60 layouts esquema nuevo + implementaciones FCP/MEP/HSP/COLE/E3T + zoo de agentes.
- `icaros-usc/overcooked_env_gen` — 179 layouts (GAN) + generador procedural (DCGAN + CMA-ME).
- Base: `HumanCompatibleAI/overcooked_ai` (ya instalado como paquete en el env `overcooked`).

Datos: `overcooked_ai_py/data/human_data/clean_{train,test}_trials.pickle` (incluidos en el paquete pip).
