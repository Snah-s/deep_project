#!/usr/bin/env python
"""E3T-lite + PPO para Overcooked-AI — script AUTONOMO (no clona el repo).

Todo lo necesario del proyecto esta inline aqui (mapa de acciones, construccion del MDP y
politicas de compañero greedy/stay/random). Solo requiere paquetes pip, no el repositorio.

Requisitos (una sola vez):
    pip install overcooked-ai "stable-baselines3>=2.0" "gymnasium>=0.29" "numpy<2"

Uso rapido:
    python train_e3t_standalone.py --timesteps 3000000
    python train_e3t_standalone.py --layouts cramped_room --timesteps 3000000     # F3 single-layout
    python train_e3t_standalone.py --ckpt-dir /content/drive/MyDrive/e3t/ckpt      # checkpoints a Drive (Colab)
    python train_e3t_standalone.py --eval-only                                     # solo evaluar el ultimo checkpoint
    python train_e3t_standalone.py --no-subproc                                    # DummyVecEnv (debug)

Reanudacion: reejecuta el mismo comando; detecta el ultimo checkpoint en --ckpt-dir y continua.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import json
from collections import deque
from typing import Iterable

import numpy as np

# ===========================================================================
# CONFIG (editable; argparse puede sobreescribir lo principal)
# ===========================================================================
TOTAL_TIMESTEPS = 3_000_000
SUBPROC         = True                       # SubprocVecEnv: recoleccion en paralelo (mas CPU/RAM)
N_ENVS          = os.cpu_count() or 8
TORCH_THREADS   = 1                          # clave: redes chicas -> 1 thread es ~12x mas rapido
HORIZON         = 400
SAVE_EVERY_STEPS = 100_000

E3T_EPSILON     = 0.5
FROZEN_REFRESH  = 200_000
SCRIPTED_PARTNER_FRAC = 0.15

# --- Shaping anti-atasco / progreso (idea adaptada de DoomBot) ---
# Reward auxiliar que empuja al agente a NO quedarse pasivo/ciclando (debilidad medida: 0 sopas
# en solitario). Se aplica solo cuando NO hubo progreso real de la tarea y se annela junto al
# shaping (via shaping_coef) para no contaminar el objetivo cooperativo.
PROGRESS_SHAPING = True
NOVELTY_BONUS    = 0.02   # bono por pisar una celda nueva en el episodio (exploracion)
STUCK_PEN        = 0.05   # penalizacion/paso cuando el estado (pos,orient,objeto) no cambia
STUCK_STEPS      = 8      # nº de pasos idénticos para considerarse "atascado"

# --- Watchdog anti-ciclo del StudentAgent entregable (fallback §6.6, idea de DoomBot) ---
# Si la obs no cambia durante WATCHDOG_STEPS pasos, el agente esta ciclando -> inyecta una
# accion de escape para romper el bucle (protege contra el peor caso: score 0 por quedarse quieto).
WATCHDOG_STEPS   = 10

LR = 2.5e-4; N_STEPS = 400; N_EPOCHS = 10; CLIP = 0.2; GAMMA = 0.99; GAE = 0.95
ENT0, ENT1 = 0.10, 0.01
SHAPING_END_FRAC = 0.6

KNOWN_LAYOUTS   = ["cramped_room", "asymmetric_advantages", "coordination_ring"]
HELDOUT_LAYOUTS = ["counter_circuit", "forced_coordination"]
TRAIN_LAYOUTS   = list(KNOWN_LAYOUTS)         # [] = todos los validos del paquete menos held-out
DIM_CAP         = 15
PAD_MARGIN      = 2

CKPT_DIR   = "./e3t_checkpoints"
EXPORT_DIR = "./e3t_export"
SEED = 0

# Globals de runtime (se rellenan en build_pool)
TRAIN_POOL: list[str] = []
HELDOUT: list[str] = []
P0 = P1 = C = 0
OBS_SHAPE: tuple = ()
_MDP_CACHE: dict = {}

# ===========================================================================
# Imports de overcooked-ai
# ===========================================================================
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.mdp.actions import Action, Direction
import overcooked_ai_py

# ===========================================================================
# Inline del repo: mapa de acciones (de src/constants.py)
# ===========================================================================
_raw_index_to_action = Action.INDEX_TO_ACTION
if hasattr(_raw_index_to_action, "items"):
    INDEX_TO_OVERCOOKED_ACTION = {int(i): a for i, a in _raw_index_to_action.items()}
else:
    INDEX_TO_OVERCOOKED_ACTION = {i: a for i, a in enumerate(_raw_index_to_action)}
NUM_ACTIONS = len(INDEX_TO_OVERCOOKED_ACTION)

def action_index_to_overcooked_action(i):
    return INDEX_TO_OVERCOOKED_ACTION[int(i)]

def get_mdp(layout_name):
    if layout_name not in _MDP_CACHE:
        _MDP_CACHE[layout_name] = OvercookedGridworld.from_layout_name(layout_name, old_dynamics=True)
    return _MDP_CACHE[layout_name]

# ===========================================================================
# Inline del repo: politicas de compañero (de policies/basic_policies.py)
# ===========================================================================
class StayPolicy(Agent):
    def action(self, state):
        return Action.STAY, {"policy_name": "stay"}

class RandomMotionPolicy(Agent):
    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.actions = list(Action.MOTION_ACTIONS)
        if Action.STAY not in self.actions:
            self.actions.append(Action.STAY)
    def action(self, state):
        idx = int(self.rng.integers(0, len(self.actions)))
        return self.actions[idx], {"policy_name": "random_motion", "sampled_idx": idx}

class GreedyFullTaskPolicy(Agent):
    """Baseline hecho a mano que intenta completar el pipeline de la sopa (no optimo)."""
    def __init__(self, ingredient="onion", avoid_teammate=True, seed=None):
        super().__init__()
        if ingredient not in {"onion", "tomato"}:
            raise ValueError("ingredient must be 'onion' or 'tomato'")
        self.ingredient = ingredient
        self.avoid_teammate = bool(avoid_teammate)
        self.rng = np.random.default_rng(seed)

    def action(self, state):
        player = state.players[self.agent_index]
        held = player.held_object
        try:
            target = self._choose_target(state)
            if target is None:
                return Action.STAY, {"policy_name": "greedy_full_task", "target": None}
            action = self._move_or_interact_towards(state, target)
            return action, {"policy_name": "greedy_full_task",
                            "held_object": None if held is None else held.name, "target": target}
        except Exception as exc:
            return Action.STAY, {"policy_name": "greedy_full_task", "fallback": True, "error": repr(exc)}

    def _choose_target(self, state):
        mdp = self.mdp
        player = state.players[self.agent_index]
        held = player.held_object
        pot_states = mdp.get_pot_states(state)
        if held is not None:
            if held.name == "soup":
                return self._nearest(player.position, mdp.get_serving_locations())
            if held.name == "dish":
                ready_pots = list(pot_states.get("ready", []))
                if ready_pots:
                    return self._nearest(player.position, ready_pots)
                almost_ready = list(pot_states.get("cooking", [])) + list(
                    pot_states.get(f"{Recipe.MAX_NUM_INGREDIENTS}_items", []))
                if almost_ready:
                    return self._nearest(player.position, almost_ready)
                return None
            if held.name in {"onion", "tomato"}:
                return self._nearest(player.position, self._pots_that_can_accept_ingredients(state, pot_states))
            return None
        ready_pots = list(pot_states.get("ready", []))
        if ready_pots:
            counter_dishes = self._counter_objects_by_name(state, "dish")
            if counter_dishes:
                return self._nearest(player.position, counter_dishes)
            dish_disps = mdp.get_dish_dispenser_locations()
            if dish_disps:
                return self._nearest(player.position, dish_disps)
        pots_needing_items = self._pots_that_can_accept_ingredients(state, pot_states)
        if pots_needing_items:
            counter_ingredients = self._counter_objects_by_name(state, self.ingredient)
            if counter_ingredients:
                return self._nearest(player.position, counter_ingredients)
            ingredient_disps = self._ingredient_dispenser_locations()
            if ingredient_disps:
                return self._nearest(player.position, ingredient_disps)
        full_not_cooking = list(pot_states.get(f"{Recipe.MAX_NUM_INGREDIENTS}_items", []))
        if full_not_cooking:
            return self._nearest(player.position, full_not_cooking)
        if list(pot_states.get("cooking", [])):
            dish_disps = mdp.get_dish_dispenser_locations()
            if dish_disps:
                return self._nearest(player.position, dish_disps)
        return None

    def _ingredient_dispenser_locations(self):
        if self.ingredient == "onion":
            return list(self.mdp.get_onion_dispenser_locations())
        return list(self.mdp.get_tomato_dispenser_locations())

    def _pots_that_can_accept_ingredients(self, state, pot_states):
        cand = []
        cand.extend(list(pot_states.get("empty", [])))
        for k in range(1, Recipe.MAX_NUM_INGREDIENTS):
            cand.extend(list(pot_states.get(f"{k}_items", [])))
        return cand

    def _counter_objects_by_name(self, state, object_name):
        return [obj.position for obj in state.objects.values() if obj.name == object_name]

    def _move_or_interact_towards(self, state, target):
        player = state.players[self.agent_index]
        pos = player.position
        orientation = player.orientation
        if self._is_adjacent(pos, target):
            desired = self._direction_from_to(pos, target)
            if orientation == desired:
                return Action.INTERACT
            return desired
        next_pos = self._next_step_towards_interaction_tile(state, target)
        if next_pos is None:
            return Action.STAY
        return Action.determine_action_for_change_in_pos(pos, next_pos)

    def _next_step_towards_interaction_tile(self, state, target):
        player = state.players[self.agent_index]
        start = player.position
        valid_positions = set(self.mdp.get_valid_player_positions())
        blocked = set()
        if self.avoid_teammate:
            for idx, other in enumerate(state.players):
                if idx != self.agent_index:
                    blocked.add(other.position)
        goals = [p for p in self._adjacent_positions(target) if p in valid_positions and p not in blocked]
        if not goals:
            goals = [p for p in self._adjacent_positions(target) if p in valid_positions]
        if not goals:
            return None
        path = self._bfs_shortest_path(start, set(goals), valid_positions, blocked)
        if path is None or len(path) < 2:
            return None
        return path[1]

    def _bfs_shortest_path(self, start, goals, valid_positions, blocked):
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            if pos in goals:
                return path
            for direction in Direction.ALL_DIRECTIONS:
                nxt = Action.move_in_direction(pos, direction)
                if nxt not in valid_positions:      continue
                if nxt in blocked and nxt not in goals: continue
                if nxt in visited:                  continue
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
        return None

    @staticmethod
    def _nearest(origin, positions):
        positions = list(positions)
        if not positions:
            return None
        return min(positions, key=lambda p: abs(p[0] - origin[0]) + abs(p[1] - origin[1]))

    @staticmethod
    def _is_adjacent(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    @staticmethod
    def _adjacent_positions(pos):
        return [Action.move_in_direction(pos, d) for d in Direction.ALL_DIRECTIONS]

    @staticmethod
    def _direction_from_to(a, b):
        direction = (b[0] - a[0], b[1] - a[1])
        if direction not in Direction.ALL_DIRECTIONS:
            raise ValueError(f"Positions are not adjacent: {a} -> {b}")
        return direction

# ===========================================================================
# Pool de layouts (F0)
# ===========================================================================
REF_CHANNELS = 26

def _layout_shape(name):
    try:
        mdp = get_mdp(name)
        if mdp.num_players != 2:
            return None
        env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0); env.reset()
        arr = np.asarray(mdp.lossless_state_encoding(env.state, horizon=HORIZON)[0])
        d0, d1, c = arr.shape
        if c != REF_CHANNELS or max(d0, d1) > DIM_CAP:
            return None
        return (d0, d1, c)
    except Exception:
        return None

def build_pool():
    global TRAIN_POOL, HELDOUT, P0, P1, C, OBS_SHAPE, OBS_SPACE, ACT_SPACE
    lay_dir = os.path.join(os.path.dirname(overcooked_ai_py.__file__), "data", "layouts")
    all_names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(lay_dir, "*.layout")))
    valid = {n: s for n in all_names if (s := _layout_shape(n))}

    HELDOUT = [n for n in HELDOUT_LAYOUTS if n in valid]
    if TRAIN_LAYOUTS:
        TRAIN_POOL = [n for n in TRAIN_LAYOUTS if n in valid and n not in HELDOUT]
        missing = [n for n in TRAIN_LAYOUTS if n not in valid]
        if missing:
            print("AVISO: layouts pedidos no validos (omitidos):", missing)
    else:
        TRAIN_POOL = [n for n in valid if n not in HELDOUT]
    assert TRAIN_POOL, "Pool de entrenamiento vacio: revisa TRAIN_LAYOUTS"

    dims = np.array([valid[n][:2] for n in valid])
    P0 = int(dims[:, 0].max()) + PAD_MARGIN
    P1 = int(dims[:, 1].max()) + PAD_MARGIN
    C = REF_CHANNELS
    OBS_SHAPE = (C, P0, P1)

    from gymnasium import spaces
    OBS_SPACE = spaces.Box(0.0, 50.0, shape=OBS_SHAPE, dtype=np.float32)
    ACT_SPACE = spaces.Discrete(6)
    OvercookedGymEnv.TRAIN_POOL = TRAIN_POOL

    print(f"Layouts validos: {len(valid)} | train: {len(TRAIN_POOL)} | held-out: {HELDOUT}")
    print("TRAIN_POOL:", TRAIN_POOL)
    print("Obs shape (C,P0,P1):", OBS_SHAPE)

# ===========================================================================
# Codificador + padding (compartido por agente, compañero y StudentAgent)
# ===========================================================================
def encode_padded(mdp, state, agent_index, horizon=None):
    horizon = HORIZON if horizon is None else horizon
    arr = np.asarray(mdp.lossless_state_encoding(state, horizon=horizon)[agent_index], dtype=np.float32)
    d0, d1, c = arr.shape
    out = np.zeros((C, P0, P1), dtype=np.float32)
    dd0, dd1 = min(d0, P0), min(d1, P1)
    out[:c, :dd0, :dd1] = np.transpose(arr[:dd0, :dd1, :], (2, 0, 1))
    return out

# ===========================================================================
# Copia congelada del ego + compañeros + wrapper Gym
# ===========================================================================
import gymnasium as gym

class WorkerFrozen:
    """Copia congelada del ego PROPIA de cada entorno (compatible con SubprocVecEnv)."""
    def __init__(self): self.policy = None; self.ready = False
    def load(self, state_dict):
        if self.policy is None:
            from stable_baselines3.common.policies import ActorCriticPolicy
            self.policy = ActorCriticPolicy(OBS_SPACE, ACT_SPACE, lambda _: 0.0, **POLICY_KWARGS)
            self.policy.to("cpu").eval()
        self.policy.load_state_dict(state_dict); self.ready = True

class FrozenEgoAgent(Agent):
    def __init__(self, frozen, rng): super().__init__(); self.frozen = frozen; self.rng = rng
    def action(self, state):
        if not self.frozen.ready:
            return action_index_to_overcooked_action(int(self.rng.integers(0, 6))), {"policy_name": "frozen_ego_random"}
        obs = encode_padded(self.mdp, state, self.agent_index)
        a, _ = self.frozen.policy.predict(obs[None], deterministic=False)
        return action_index_to_overcooked_action(int(a[0])), {"policy_name": "frozen_ego"}

class EpsilonAgent(Agent):
    def __init__(self, base, epsilon, rng):
        super().__init__(); self.base = base; self.epsilon = float(epsilon); self.rng = rng
    def set_mdp(self, mdp): super().set_mdp(mdp); self.base.set_mdp(mdp)
    def set_agent_index(self, i): super().set_agent_index(i); self.base.set_agent_index(i)
    def reset(self):
        super().reset()
        if hasattr(self, "base"): self.base.reset()
    def action(self, state):
        if self.rng.random() < self.epsilon:
            return action_index_to_overcooked_action(int(self.rng.integers(0, 6))), {"policy_name": "eps_random"}
        return self.base.action(state)

def make_partner(frozen, rng):
    if rng.random() < SCRIPTED_PARTNER_FRAC:
        pick = rng.integers(0, 3)
        if pick == 0: return GreedyFullTaskPolicy(seed=int(rng.integers(1e9)))
        if pick == 1: return StayPolicy()
        return RandomMotionPolicy(seed=int(rng.integers(1e9)))
    return EpsilonAgent(FrozenEgoAgent(frozen, rng), E3T_EPSILON, rng)

class OvercookedGymEnv(gym.Env):
    metadata = {"render_modes": []}
    TRAIN_POOL: list[str] = []
    def __init__(self, seed=0):
        super().__init__()
        self.frozen = WorkerFrozen()
        self.shaping_coef = 1.0
        self.observation_space = OBS_SPACE
        self.action_space = ACT_SPACE
        self._rng = np.random.default_rng(seed)

    def set_shaping_coef(self, v): self.shaping_coef = float(v)
    def load_frozen(self, state_dict): self.frozen.load(state_dict)

    def reset(self, *, seed=None, options=None):
        if seed is not None: self._rng = np.random.default_rng(seed)
        self.layout = self.TRAIN_POOL[int(self._rng.integers(0, len(self.TRAIN_POOL)))]
        self.mdp = get_mdp(self.layout)
        self.oc = OvercookedEnv.from_mdp(self.mdp, horizon=HORIZON, info_level=0)
        self.oc.reset()
        self.seat = int(self._rng.integers(0, 2))
        self.partner = make_partner(self.frozen, self._rng)
        self.partner.reset(); self.partner.set_mdp(self.mdp); self.partner.set_agent_index(1 - self.seat)
        self._visited = set(); self._last_sig = None; self._stuck = 0  # tracking anti-atasco
        obs = encode_padded(self.mdp, self.oc.state, self.seat)
        return obs, {"layout": self.layout, "seat": self.seat}

    def _aux_reward(self, next_state, sparse_team, shaped_own):
        """Reward de exploracion/anti-atasco (solo si no hubo progreso real de la tarea)."""
        if not PROGRESS_SHAPING or sparse_team != 0 or shaped_own != 0:
            self._stuck = 0
            return 0.0
        try:
            p = next_state.players[self.seat]
            sig = (p.position, p.orientation, None if p.held_object is None else p.held_object.name)
            aux = 0.0
            if p.position not in self._visited:
                self._visited.add(p.position); aux += NOVELTY_BONUS
            if sig == self._last_sig:
                self._stuck += 1
                if self._stuck >= STUCK_STEPS: aux -= STUCK_PEN
            else:
                self._stuck = 0
            self._last_sig = sig
            return aux
        except Exception:
            return 0.0

    def step(self, action):
        state = self.oc.state
        our = action_index_to_overcooked_action(int(action))
        p_act, _ = self.partner.action(state)
        joint = [None, None]; joint[self.seat] = our; joint[1 - self.seat] = p_act
        next_state, sparse_team, done, info = self.oc.step(tuple(joint))
        shaped_own = float(info["shaped_r_by_agent"][self.seat])
        aux = self._aux_reward(next_state, sparse_team, shaped_own)   # anti-atasco (DoomBot)
        reward = float(sparse_team) + self.shaping_coef * (shaped_own + aux)  # aux se annela con el shaping
        obs = encode_padded(self.mdp, next_state, self.seat)
        info["sparse_team"] = float(sparse_team)
        return obs, reward, bool(done), False, info

# ===========================================================================
# Red CNN + PPO + callbacks
# ===========================================================================
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=64):
        super().__init__(obs_space, features_dim)
        c = obs_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, *obs_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())
    def forward(self, x): return self.linear(self.cnn(x))

POLICY_KWARGS = dict(
    features_extractor_class=SmallCNN,
    features_extractor_kwargs=dict(features_dim=64),
    net_arch=[64, 64],
    normalize_images=False,
)

class E3TCallback(BaseCallback):
    """Difunde la copia congelada a los workers y annela shaping+entropia (una vez por rollout)."""
    def __init__(self, total):
        super().__init__(); self.total = total; self._last_refresh = -1
    def _on_rollout_start(self):
        t = self.num_timesteps
        if self._last_refresh < 0 or (t - self._last_refresh) >= FROZEN_REFRESH:
            sd = {k: v.detach().cpu() for k, v in self.model.policy.state_dict().items()}
            self.training_env.env_method("load_frozen", sd)
            self._last_refresh = t
        frac = min(1.0, t / self.total)
        self.training_env.env_method("set_shaping_coef", max(0.0, 1.0 - frac / SHAPING_END_FRAC))
        self.model.ent_coef = ENT0 + (ENT1 - ENT0) * frac
    def _on_step(self): return True

def _make_env_fn(rank):
    def _f():
        if SUBPROC:
            import torch as _t; _t.set_num_threads(1)
        return Monitor(OvercookedGymEnv(seed=SEED * 1000 + rank))
    return _f

def make_venv():
    fns = [_make_env_fn(i) for i in range(N_ENVS)]
    if SUBPROC and N_ENVS > 1:
        return SubprocVecEnv(fns, start_method="fork")
    return DummyVecEnv(fns)

# ===========================================================================
# Utilidades de checkpoint
# ===========================================================================
def latest_ckpt(d):
    zips = glob.glob(os.path.join(d, "e3t_*_steps.zip"))
    if not zips:
        return None
    def steps(p):
        m = re.search(r"_(\d+)_steps\.zip$", p); return int(m.group(1)) if m else -1
    return max(zips, key=steps)

def load_ppo(path, venv=None, device="cpu"):
    return PPO.load(path, env=venv, device=device,
                    custom_objects={"policy_kwargs": POLICY_KWARGS, "lr_schedule": lambda _: 0.0,
                                    "clip_range": lambda _: CLIP, "clip_range_vf": None})

# ===========================================================================
# Entrenamiento (reanudable)
# ===========================================================================
def train():
    os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(EXPORT_DIR, exist_ok=True)
    venv = make_venv()                                  # crear (y forkear) ANTES de iniciar CUDA
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(TORCH_THREADS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"VecEnv: {'SubprocVecEnv' if (SUBPROC and N_ENVS > 1) else 'DummyVecEnv'} | "
          f"N_ENVS={N_ENVS} | device={device} | torch_threads={TORCH_THREADS}")

    ckpt = latest_ckpt(CKPT_DIR)
    if ckpt:
        print("Reanudando desde:", ckpt)
        model = load_ppo(ckpt, venv, device)
    else:
        print("Entrenando desde cero.")
        import importlib.util
        tb_log = os.path.join(EXPORT_DIR, "tb") if importlib.util.find_spec("tensorboard") else None
        model = PPO("MlpPolicy", venv, policy_kwargs=POLICY_KWARGS,
                    learning_rate=LR, n_steps=N_STEPS, batch_size=max(N_ENVS * N_STEPS // 4, 1),
                    n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE, clip_range=CLIP,
                    ent_coef=ENT0, vf_coef=0.5, max_grad_norm=0.5, seed=SEED,
                    device=device, verbose=1, tensorboard_log=tb_log)

    checkpoint_cb = CheckpointCallback(save_freq=max(SAVE_EVERY_STEPS // N_ENVS, 1),
                                       save_path=CKPT_DIR, name_prefix="e3t")
    e3t_cb = E3TCallback(TOTAL_TIMESTEPS)
    remaining = max(TOTAL_TIMESTEPS - model.num_timesteps, 0)
    print(f"Pasos hechos: {model.num_timesteps:,} | restantes: {remaining:,}")
    import importlib.util as _ilu
    use_pbar = bool(_ilu.find_spec("tqdm") and _ilu.find_spec("rich"))
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=[checkpoint_cb, e3t_cb],
                    reset_num_timesteps=False, progress_bar=use_pbar)
    model.save(os.path.join(EXPORT_DIR, "e3t_final"))
    venv.close()
    print("Modelo final en:", os.path.join(EXPORT_DIR, "e3t_final.zip"))
    return model

# ===========================================================================
# Evaluacion con score oficial
# ===========================================================================
def _eval_partner(kind, seed):
    if kind == "greedy": return GreedyFullTaskPolicy(seed=seed)
    if kind == "stay":   return StayPolicy()
    return RandomMotionPolicy(seed=seed)

def eval_episode(model, mdp, partner, seat, horizon, seed):
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0); env.reset()
    partner.reset(); partner.set_mdp(mdp); partner.set_agent_index(1 - seat)
    soups, t_first, t_last = 0, None, None
    for t in range(horizon):
        obs = encode_padded(mdp, env.state, seat, horizon=horizon)
        a, _ = model.predict(obs[None], deterministic=True)
        our = action_index_to_overcooked_action(int(a[0]))
        p_act, _ = partner.action(env.state)
        joint = [None, None]; joint[seat] = our; joint[1 - seat] = p_act
        _, sparse_team, done, _ = env.step(tuple(joint))
        if sparse_team > 0:
            soups += int(round(sparse_team / 20.0))
            if t_first is None: t_first = t
            t_last = t
        if done: break
    if soups == 0:
        return 0.0, 0
    return float(10000 * soups + 10 * (horizon - t_last) + (horizon - t_first)), soups

def evaluate(model, seeds=(67, 68, 69)):
    print("\n==== Evaluacion (score oficial) ====")
    header = f"{'split':9} {'layout':22} {'partner':7} {'H':>4} {'soups':>6} {'score':>9}"
    print(header)
    for split, layouts in [("known", KNOWN_LAYOUTS), ("held-out", HELDOUT)]:
        for lay in layouts:
            if lay not in _MDP_CACHE and _layout_shape(lay) is None:
                continue
            mdp = get_mdp(lay)
            for pk in ["greedy", "stay", "random"]:
                for H in [250, 400]:
                    res = [eval_episode(model, mdp, _eval_partner(pk, sd), seat, H, sd)
                           for seat in (0, 1) for sd in seeds]
                    soups = np.mean([r[1] for r in res]); score = np.mean([r[0] for r in res])
                    print(f"{split:9} {lay:22} {pk:7} {H:>4} {soups:>6.2f} {score:>9.0f}")

# ===========================================================================
# Exportar StudentAgent (autocontenido; solo necesita SB3 + numpy en evaluacion)
# ===========================================================================
STUDENT_SRC = r'''
# StudentAgent E3T-lite: politica PPO (SB3) + fallback en capas.
from __future__ import annotations
import json, os
from collections import deque
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

def _build_smallcnn():
    import torch
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    class SmallCNN(BaseFeaturesExtractor):
        def __init__(self, obs_space, features_dim=64):
            super().__init__(obs_space, features_dim)
            c = obs_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Flatten())
            with torch.no_grad():
                n = self.cnn(torch.zeros(1, *obs_space.shape)).shape[1]
            self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())
        def forward(self, x):
            return self.linear(self.cnn(x))
    return SmallCNN

class StudentAgent:
    def __init__(self, config=None):
        self.config = config or {}
        with open(os.path.join(_HERE, "e3t_meta.json")) as f:
            self.meta = json.load(f)
        self.C, self.P0, self.P1 = self.meta["C"], self.meta["P0"], self.meta["P1"]
        self.watchdog_steps = int(self.meta.get("watchdog_steps", 10))
        self._hist = deque(maxlen=self.watchdog_steps)
        self._break_i = 0
        self.model = None
        try:
            from stable_baselines3 import PPO
            SmallCNN = _build_smallcnn()
            policy_kwargs = dict(features_extractor_class=SmallCNN,
                                 features_extractor_kwargs=dict(features_dim=64),
                                 net_arch=[64, 64], normalize_images=False)
            custom_objects = {"policy_kwargs": policy_kwargs, "lr_schedule": lambda _: 0.0,
                              "clip_range": lambda _: 0.2, "clip_range_vf": None}
            self.model = PPO.load(os.path.join(_HERE, "e3t_model.zip"),
                                  device="cpu", custom_objects=custom_objects)
            self.reset()
        except Exception as exc:
            print("StudentAgent: no se pudo cargar el modelo, uso fallback:", repr(exc))

    def reset(self):
        self._hist.clear(); self._break_i = 0
        if self.model is not None:
            try:
                self.model.predict(np.zeros((1, self.C, self.P0, self.P1), np.float32), deterministic=True)
            except Exception:
                pass

    def _pad(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        d0, d1, c = arr.shape
        out = np.zeros((self.C, self.P0, self.P1), np.float32)
        dd0, dd1 = min(d0, self.P0), min(d1, self.P1)
        out[:min(c, self.C), :dd0, :dd1] = np.transpose(arr[:dd0, :dd1, :min(c, self.C)], (2, 0, 1))
        return out

    def act(self, obs):
        try:
            arr = obs["obs"] if isinstance(obs, dict) else obs
            x = self._pad(arr)
            a, _ = self.model.predict(x[None], deterministic=True)
            action = int(a[0])
            # Watchdog anti-ciclo (idea de DoomBot): si la obs no cambia por N pasos, romper el bucle.
            self._hist.append(hash(x.tobytes()))
            if len(self._hist) == self._hist.maxlen and len(set(self._hist)) == 1:
                self._break_i = (self._break_i + 1) % 5
                return [5, 0, 1, 2, 3][self._break_i]  # interact, luego moverse en las 4 direcciones
            return action
        except Exception:
            return 4  # stay
'''

def export_student(out_dir="."):
    model_zip = os.path.join(EXPORT_DIR, "e3t_final.zip")
    assert os.path.exists(model_zip), f"No existe {model_zip}; entrena primero."
    shutil.copy(model_zip, os.path.join(out_dir, "e3t_model.zip"))
    with open(os.path.join(out_dir, "e3t_meta.json"), "w") as f:
        json.dump(dict(obs_shape=list(OBS_SHAPE), P0=P0, P1=P1, C=C, horizon=HORIZON,
                       channels=REF_CHANNELS, train_pool=TRAIN_POOL, heldout=HELDOUT,
                       watchdog_steps=WATCHDOG_STEPS), f, indent=2)
    with open(os.path.join(out_dir, "student_agent_e3t.py"), "w") as f:
        f.write(STUDENT_SRC)
    print("Exportado en", os.path.abspath(out_dir), ": student_agent_e3t.py + e3t_model.zip + e3t_meta.json")

# ===========================================================================
# CLI
# ===========================================================================
def _resolve_drive(path):
    if path.startswith("/content/drive"):
        try:
            from google.colab import drive
            if not os.path.ismount("/content/drive"):
                drive.mount("/content/drive")
        except Exception as exc:
            print("Aviso: no se pudo montar Drive:", repr(exc))
    return path

def parse_args():
    p = argparse.ArgumentParser(description="E3T-lite + PPO standalone para Overcooked-AI")
    p.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    p.add_argument("--layouts", type=str, default=",".join(TRAIN_LAYOUTS),
                   help="lista separada por comas; vacio ('') = todos menos held-out")
    p.add_argument("--heldout", type=str, default=",".join(HELDOUT_LAYOUTS))
    p.add_argument("--n-envs", type=int, default=N_ENVS)
    p.add_argument("--no-subproc", action="store_true", help="usar DummyVecEnv (debug)")
    p.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    p.add_argument("--ckpt-dir", type=str, default=CKPT_DIR)
    p.add_argument("--export-dir", type=str, default=EXPORT_DIR)
    p.add_argument("--out-dir", type=str, default=".", help="donde escribir el StudentAgent entregable")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()

def main():
    global TOTAL_TIMESTEPS, TRAIN_LAYOUTS, HELDOUT_LAYOUTS, N_ENVS, SUBPROC
    global TORCH_THREADS, CKPT_DIR, EXPORT_DIR, SEED
    args = parse_args()
    TOTAL_TIMESTEPS = args.timesteps
    TRAIN_LAYOUTS   = [s for s in args.layouts.split(",") if s.strip()]
    HELDOUT_LAYOUTS = [s for s in args.heldout.split(",") if s.strip()]
    N_ENVS          = args.n_envs
    SUBPROC         = not args.no_subproc
    TORCH_THREADS   = args.torch_threads
    CKPT_DIR        = _resolve_drive(args.ckpt_dir)
    EXPORT_DIR      = _resolve_drive(args.export_dir)
    SEED            = args.seed

    build_pool()

    if args.eval_only or args.export_only:
        model = None
        if not args.export_only:
            path = os.path.join(EXPORT_DIR, "e3t_final.zip")
            if not os.path.exists(path):
                path = latest_ckpt(CKPT_DIR)
            assert path, "No hay modelo para evaluar."
            model = load_ppo(path, None, "cpu")
        if args.eval_only:
            evaluate(model)
        if args.export_only:
            export_student(args.out_dir)
        return

    model = train()
    if not args.skip_eval:
        evaluate(model)
    export_student(args.out_dir)

if __name__ == "__main__":
    main()
