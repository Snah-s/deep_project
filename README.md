# Proyecto: Overcooked

Primero construiremos un dataset colectivo de partidas en Overcooked-AI. El objetivo es recolectar demostraciones que luego serán usadas para entrenar un primer modelo mediante Imitation Learning.

## Grabaciones

- Se debe realizar un total de **20 grabaciones**.
- Cada partida tiene una duración de **250 timesteps** (verifica en el archivo `collect_demonstrations.yaml`, en `environment.horizon: 250`).
- En total, cada grupo debería aportar `250 x 20 = 5000` transiciones humanas.
- Las 20 grabaciones deben ser en **10 escenarios distintos**, 2 grabaciones por escenario.

Al ejecutar el código de grabación se crearán las siguientes carpetas:

- `data/demonstrations/`: ahí se guardan los archivos que luego usarás para entrenar tu modelo.
- `outputs/collect_demonstrations/`: ahí se guardan archivos auxiliares del runner (logs/debug).

## Agentes

Nota que el juego cuenta con dos agentes: uno es automático y el otro es controlado por ustedes.

El agente automático es configurado en el archivo `collect_demonstrations.yaml`, en `policies.agent_0.name: greedy_full_task`. Las opciones de agentes disponibles son: `stay`, `random_motion` y `greedy_full_task`. Asigna de manera aleatoria esos agentes en sus grabaciones.

## Escenarios

Como fue indicado previamente, cada grupo debe recolectar datos en 10 escenarios distintos. Los disponibles en el repositorio oficial son:

- `asymmetric_advantages`
- `coordination_ring`
- `counter_circuit`
- `cramped_room`
- `forced_coordination`
- `large_room`
- `simple_o`
- `simple_tomato`
- `small_corridor`
- `soup_coordination`
- `tutorial_0`
- `tutorial_1`
- `tutorial_2`
- `tutorial_3`

Puede usar cualquiera de ellos o proponer su propio escenario (opción recomendada). Si crean un escenario custom, deben guardar también el archivo `.layout` correspondiente.

## Dependencias

```bash
pip install overcooked-ai
pip install "numpy<2"
pip install PyYAML>=6.0 Pillow>=10.0 imageio>=2.31
```

## Uso

### Run random

```bash
python -m src.run_game --config configs/play.yaml
```

### Create dataset

```bash
python -m src.collect_demonstrations --config configs/collect_demonstrations.yaml
```

## Agente autónomo

Después se debe diseñar un agente autónomo capaz de colaborar con otro agente en el entorno Overcooked-AI. El objetivo es preparar y entregar la mayor cantidad posible de sopas dentro de un episodio limitado por tiempo.

Cada grupo entregará un único agente para jugar Overcooked-AI. En cada escenario, el agente será evaluado junto con un compañero definido para esa ronda. El objetivo es obtener el mayor puntaje posible preparando y entregando sopas.

Cada escenario tendrá tres intentos con tres seeds distintos. El puntaje oficial del escenario será el promedio de los tres intentos. En algunos escenarios se evaluará también el cambio de rol del agente.

### Score

```
Score = 10000 * sopas + 10 * (horizon - timestep de última sopa) + (horizon - timestep de primera sopa) - penalización
```

Si no se entrega ninguna sopa, el score del intento será `0`.

### Penalización

```
Penalización = min(100 * timeouts, 5000)
```

El número de sopas es el factor principal. El tiempo funciona como criterio de desempate entre agentes que entregan la misma cantidad de sopas. Las penalizaciones solo afectan errores técnicos de ejecución, como exceder el tiempo máximo permitido para decidir una acción.

Los escenarios 1, 2 y 3 serán conocidos. Los escenarios 4, 5 y 6 serán nuevos layouts y serán revelados durante la competencia.

Cada escenario otorga una nota máxima. El grupo conserva la nota más alta alcanzada.
