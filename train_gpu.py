#!/usr/bin/env python
"""train_gpu.py — Entrenamiento RL 'hecho bien' para GPU LOCAL.

Incorpora TODO lo que en las corridas previas nos faltaba y por lo que el agente colapsaba:
  1. ESCALA: millones de pasos, muchos envs en paralelo, updates en GPU.
  2. SHAPING DENSO (PBRS, potencial = -dist al subobjetivo del greedy) + shaping por evento
     annealado LENTO (sobre millones de pasos, no 240k) -> hay gradiente para bootstrapear.
  3. WARM-START con BEHAVIOR CLONING sobre los datos humanos (receta Carroll/PPO_BC/BCP):
     el torso CNN arranca sabiendo, y opcionalmente el BC es el companero competente.
  4. SELF-PLAY con parametros compartidos (2 asientos = misma politica, ambos entrenan).

Asume el REPO presente (usa policies.basic_policies del repo con el greedy arreglado).
Requiere: overcooked-ai, stable-baselines3, torch+CUDA, numpy<2, pandas.

Uso tipico (GPU local):
    python train_gpu.py --layouts cramped_room,asymmetric_advantages,coordination_ring \
                        --timesteps 20000000 --n-envs 16 --mode selfplay
    python train_gpu.py --skip-bc                 # sin warm-start BC
    python train_gpu.py --eval-only               # evaluar el ultimo checkpoint
Reanuda solo: reejecuta el mismo comando (detecta el ultimo checkpoint).
"""
from __future__ import annotations

import argparse, ast, glob, os, re, json, sys
import numpy as np

# --- Repo: poner overcooked/ en el path para importar el greedy arreglado ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_ROOT, "overcooked")
if os.path.isdir(_PKG) and _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld, OvercookedState, SoupState, ObjectState, PlayerState, Recipe)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action
import overcooked_ai_py

from policies.basic_policies import GreedyFullTaskPolicy, StayPolicy  # greedy ARREGLADO del repo

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ===========================================================================
# CONFIG (argparse sobreescribe lo principal)
# ===========================================================================
HORIZON        = 400
DENSE_COEF     = 1.0
SHAPING_ANNEAL_STEPS = 6_000_000   # el shaping (denso+evento) se annela LENTO sobre estos pasos
ENT0, ENT1     = 0.02, 0.002       # entropia moderada->baja (0.10 ahogaba; 0.001 no exploraba)
LR, N_STEPS, N_EPOCHS, CLIP, GAMMA, GAE = 2.5e-4, 400, 10, 0.2, 0.99, 0.95
REF_CHANNELS   = 26
DIM_CAP, PAD_MARGIN = 15, 2
BC_EPOCHS, BC_STAY_KEEP = 8, 0.15  # submuestrear 'stay' al 15% (64% de la data es stay)
SEED = 0

# rellenados en build_pool
TRAIN_POOL, P0, P1, C, OBS_SHAPE, OBS_SPACE, ACT_SPACE = [], 0, 0, 0, (), None, None
_MDP_CACHE = {}
def get_mdp(name):
    if name not in _MDP_CACHE:
        _MDP_CACHE[name] = OvercookedGridworld.from_layout_name(name, old_dynamics=True)
    return _MDP_CACHE[name]

_A2I = {a: int(i) for i, a in ((Action.INDEX_TO_ACTION.items()) if hasattr(Action.INDEX_TO_ACTION, "items")
                               else enumerate(Action.INDEX_TO_ACTION))}
def a2oc(i): return Action.INDEX_TO_ACTION[int(i)]

# ===========================================================================
# Pool de layouts + codificador
# ===========================================================================
def _shape_ok(name):
    try:
        mdp = get_mdp(name)
        if mdp.num_players != 2: return None
        env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0); env.reset()
        a = np.asarray(mdp.lossless_state_encoding(env.state, horizon=HORIZON)[0])
        if a.shape[2] != REF_CHANNELS or max(a.shape[0], a.shape[1]) > DIM_CAP: return None
        return a.shape[:2]
    except Exception:
        return None

def build_pool(train_layouts):
    global TRAIN_POOL, P0, P1, C, OBS_SHAPE, OBS_SPACE, ACT_SPACE
    lay_dir = os.path.join(os.path.dirname(overcooked_ai_py.__file__), "data", "layouts")
    valid = {}
    for p in glob.glob(os.path.join(lay_dir, "*.layout")):
        n = os.path.splitext(os.path.basename(p))[0]; s = _shape_ok(n)
        if s: valid[n] = s
    TRAIN_POOL = [n for n in train_layouts if n in valid] or [n for n in valid]
    dims = np.array(list(valid.values()))
    P0, P1, C = int(dims[:, 0].max()) + PAD_MARGIN, int(dims[:, 1].max()) + PAD_MARGIN, REF_CHANNELS
    OBS_SHAPE = (C, P0, P1)
    OBS_SPACE = spaces.Box(0.0, 50.0, shape=OBS_SHAPE, dtype=np.float32)
    ACT_SPACE = spaces.Discrete(6)
    print(f"Pool entrenamiento: {TRAIN_POOL} | obs {OBS_SHAPE}")

def encode(mdp, state, idx, horizon=HORIZON):
    a = np.asarray(mdp.lossless_state_encoding(state, horizon=horizon)[idx], dtype=np.float32)
    out = np.zeros((C, P0, P1), np.float32)
    d0, d1 = min(a.shape[0], P0), min(a.shape[1], P1)
    out[:a.shape[2], :d0, :d1] = np.transpose(a[:d0, :d1, :], (2, 0, 1))
    return out

# ===========================================================================
# Red CNN compartida (extractor de SB3, reusable por el BC para warm-start)
# ===========================================================================
class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=64):
        super().__init__(obs_space, features_dim)
        c = obs_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, *obs_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())
    def forward(self, x): return self.linear(self.cnn(x))

def policy_kwargs():
    return dict(features_extractor_class=SmallCNN, features_extractor_kwargs=dict(features_dim=64),
                net_arch=[64, 64], normalize_images=False)

# ===========================================================================
# (3) Behavior Cloning sobre datos humanos: converter viejo->nuevo + dataset + entrenamiento
# ===========================================================================
_COOK = getattr(Recipe, "COOK_TIME", 20) if hasattr(Recipe, "COOK_TIME") else 20

def _conv_obj(od):
    name = od["name"]; pos = tuple(od["position"])
    if name == "soup":
        ing, num, cook = od["state"]
        no = num if ing == "onion" else 0; nt = num if ing == "tomato" else 0
        if (no + nt) < 3:
            return SoupState.get_soup(pos, num_onions=no, num_tomatoes=nt, cooking_tick=-1)
        fin = cook >= _COOK
        return SoupState.get_soup(pos, num_onions=no, num_tomatoes=nt,
                                  cooking_tick=(_COOK if fin else max(cook, 0)), finished=fin)
    return ObjectState(name, pos)

def _old_state_to_new(s, mdp):
    players = [PlayerState(tuple(p["position"]), tuple(p["orientation"]),
                           held_object=(_conv_obj(p["held_object"]) if p.get("held_object") else None))
               for p in s["players"]]
    raw = s.get("objects", {}); items = raw.values() if isinstance(raw, dict) else raw
    objs = {}
    for od in items:
        o = _conv_obj(od); objs[o.position] = o
    return OvercookedState(players, objs, all_orders=mdp.start_all_orders)

def _action_to_idx(a):
    if isinstance(a, str):
        return _A2I[Action.INTERACT] if a == "interact" else _A2I[Action.STAY]
    t = tuple(a)
    return _A2I.get(t, _A2I[Action.STAY])

def build_bc_dataset(rng):
    """Reconstruye los estados humanos, codifica por asiento y empareja con la accion. Filtra 'stay'."""
    import pandas as pd
    d = os.path.join(os.path.dirname(overcooked_ai_py.__file__), "data", "human_data")
    df = pd.read_pickle(os.path.join(d, "clean_train_trials.pickle"))
    X, Y = [], []
    for _, r in df.iterrows():
        lay = r["layout_name"]
        try:
            mdp = get_mdp(lay); st = _old_state_to_new(ast.literal_eval(r["state"]), mdp)
            ja = ast.literal_eval(r["joint_action"])
        except Exception:
            continue
        for idx in (0, 1):
            ai = _action_to_idx(ja[idx])
            if ai == 4 and rng.random() > BC_STAY_KEEP:   # submuestrear stay
                continue
            try:
                X.append(encode(mdp, st, idx)); Y.append(ai)
            except Exception:
                pass
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int64)
    print(f"BC dataset: {len(Y)} muestras | %% stay={100*(Y==4).mean():.0f}")
    return X, Y

def train_bc(device, rng):
    X, Y = build_bc_dataset(rng)
    if len(Y) < 1000:
        print("BC: pocas muestras, se omite."); return None
    ext = SmallCNN(OBS_SPACE).to(device)
    head = nn.Linear(64, 6).to(device)
    opt = torch.optim.Adam(list(ext.parameters()) + list(head.parameters()), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(Y); n = len(Yt); bs = 256
    for ep in range(BC_EPOCHS):
        perm = torch.randperm(n); tot = 0.0; correct = 0
        for i in range(0, n, bs):
            b = perm[i:i + bs]; xb = Xt[b].to(device); yb = Yt[b].to(device)
            logits = head(ext(xb)); loss = lossf(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); correct += (logits.argmax(1) == yb).sum().item()
        print(f"  BC epoch {ep+1}/{BC_EPOCHS}  loss={tot/n:.3f}  acc={correct/n:.2f}")
    return ext.cpu().eval(), head.cpu().eval()

class BCPartner(Agent):
    """Companero = politica BC (competente, aprendida de humanos). Receta BCP de Carroll 2019."""
    def __init__(self, ext, head, rng):
        super().__init__(); self.ext = ext; self.head = head; self.rng = rng
    def action(self, state):
        try:
            obs = encode(self.mdp, state, self.agent_index)
            with torch.no_grad():
                logits = self.head(self.ext(torch.as_tensor(obs[None])))
                p = torch.softmax(logits, 1).numpy()[0]
            a = int(self.rng.choice(6, p=p))
            return a2oc(a), {"policy_name": "bc"}
        except Exception:
            return Action.STAY, {"policy_name": "bc_fallback"}

# ===========================================================================
# (2) Shaping denso PBRS (potencial = -dist al subobjetivo del greedy)
# ===========================================================================
def make_dense(seat):
    g = GreedyFullTaskPolicy(seed=0)
    st = {"d": None, "t": None}
    def compute(mdp, state):
        try:
            g.set_mdp(mdp); g.set_agent_index(seat)
            tgt = g._choose_target(state)
            if tgt is None: st["d"] = st["t"] = None; return 0.0
            p = state.players[seat].position; d = abs(p[0] - tgt[0]) + abs(p[1] - tgt[1])
            delta = (st["d"] - d) if (st["t"] == tgt and st["d"] is not None) else 0.0
            st["d"], st["t"] = d, tgt; return float(delta)
        except Exception:
            st["d"] = st["t"] = None; return 0.0
    def reset(): st["d"] = st["t"] = None
    compute.reset = reset
    return compute

# ===========================================================================
# (4) Self-play VecEnv: 2 asientos = misma politica, ambos entrenan (+ dense shaping)
# ===========================================================================
class SelfPlayVecEnv(VecEnv):
    def __init__(self, n_games, seed=0):
        self.n_games = n_games; self.shaping_coef = 1.0
        super().__init__(2 * n_games, OBS_SPACE, ACT_SPACE)
        self._rng = np.random.default_rng(seed)
        self.games = [None] * n_games; self.mdps = [None] * n_games
        self.dense = [[make_dense(0), make_dense(1)] for _ in range(n_games)]; self._act = None
    def set_shaping_coef(self, v): self.shaping_coef = float(v)
    def _rg(self, g):
        lay = TRAIN_POOL[int(self._rng.integers(0, len(TRAIN_POOL)))]
        self.mdps[g] = get_mdp(lay); e = OvercookedEnv.from_mdp(self.mdps[g], horizon=HORIZON, info_level=0)
        e.reset(); self.games[g] = e
        for dfn in self.dense[g]: dfn.reset()
    def reset(self):
        o = np.zeros((self.num_envs,) + OBS_SHAPE, np.float32)
        for g in range(self.n_games):
            self._rg(g); s = self.games[g].state
            o[2 * g] = encode(self.mdps[g], s, 0); o[2 * g + 1] = encode(self.mdps[g], s, 1)
        return o
    def step_async(self, a): self._act = a
    def step_wait(self):
        o = np.zeros((self.num_envs,) + OBS_SHAPE, np.float32); r = np.zeros(self.num_envs, np.float32)
        d = np.zeros(self.num_envs, bool); inf = [{} for _ in range(self.num_envs)]
        for g in range(self.n_games):
            e = self.games[g]; mdp = self.mdps[g]
            ns, sp, done, info = e.step((a2oc(self._act[2 * g]), a2oc(self._act[2 * g + 1])))
            for k in (0, 1):
                sh = float(info["shaped_r_by_agent"][k]); de = self.dense[g][k](mdp, ns)
                r[2 * g + k] = sp + self.shaping_coef * (sh + DENSE_COEF * de)
            d[2 * g] = d[2 * g + 1] = done
            if done:
                inf[2 * g]["terminal_observation"] = encode(mdp, ns, 0)
                inf[2 * g + 1]["terminal_observation"] = encode(mdp, ns, 1)
                self._rg(g); ns = self.games[g].state; mdp = self.mdps[g]
            o[2 * g] = encode(mdp, ns, 0); o[2 * g + 1] = encode(mdp, ns, 1)
        return o, r, d, inf
    def close(self): pass
    def _n(self, i): return self.num_envs if i is None else len(np.atleast_1d(i))
    def get_attr(self, n, indices=None): return [getattr(self, n, None)] * self._n(indices)
    def set_attr(self, n, v, indices=None): setattr(self, n, v)
    def env_method(self, m, *a, indices=None, **k):
        fn = getattr(self, m, None); res = fn(*a, **k) if fn else None; return [res] * self._n(indices)
    def env_is_wrapped(self, w, indices=None): return [False] * self._n(indices)
    def render(self, *a, **k): return None
    def seed(self, s=None): self._rng = np.random.default_rng(s); return [s]

# ===========================================================================
# Callback: anneal LENTO de shaping (sobre SHAPING_ANNEAL_STEPS) + entropia
# ===========================================================================
class AnnealCB(BaseCallback):
    def _on_rollout_start(self):
        t = self.num_timesteps
        f = min(1.0, t / SHAPING_ANNEAL_STEPS)
        self.training_env.env_method("set_shaping_coef", max(0.0, 1.0 - f))
        self.model.ent_coef = ENT0 + (ENT1 - ENT0) * min(1.0, t / SHAPING_ANNEAL_STEPS)
    def _on_step(self): return True

# ===========================================================================
# Entrenamiento
# ===========================================================================
def latest_ckpt(d):
    zs = glob.glob(os.path.join(d, "gpu_*_steps.zip"))
    return max(zs, key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1))) if zs else None

def load_ppo(path, venv, device):
    return PPO.load(path, env=venv, device=device, custom_objects={
        "policy_kwargs": policy_kwargs(), "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: CLIP, "clip_range_vf": None})

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("AVISO: sin CUDA. El script corre, pero esto esta pensado para GPU local.")
    os.makedirs(args.ckpt_dir, exist_ok=True); os.makedirs(args.export_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(SEED)

    # (3) BC warm-start
    bc = None
    if not args.skip_bc:
        print("== Behavior Cloning sobre datos humanos ==")
        bc = train_bc(device, rng)

    # VecEnv self-play (fork antes de tocar CUDA en el proceso padre no aplica a este VecEnv custom)
    torch.set_num_threads(1)
    venv = SelfPlayVecEnv(args.n_envs, seed=SEED)

    ckpt = latest_ckpt(args.ckpt_dir)
    if ckpt:
        print("Reanudando:", ckpt); model = load_ppo(ckpt, venv, device)
    else:
        import importlib.util as _ilu
        tb = os.path.join(args.export_dir, "tb") if _ilu.find_spec("tensorboard") else None
        model = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs(), learning_rate=LR, n_steps=N_STEPS,
                    batch_size=args.n_envs * 2 * N_STEPS // 4, n_epochs=N_EPOCHS, gamma=GAMMA,
                    gae_lambda=GAE, clip_range=CLIP, ent_coef=ENT0, vf_coef=0.5, max_grad_norm=0.5,
                    seed=SEED, device=device, verbose=1, tensorboard_log=tb)
        if bc is not None:   # WARM-START: cargar el torso CNN del BC en la politica PPO
            try:
                model.policy.features_extractor.load_state_dict(bc[0].state_dict())
                model.policy.pi_features_extractor.load_state_dict(bc[0].state_dict())
                model.policy.vf_features_extractor.load_state_dict(bc[0].state_dict())
                print("Warm-start BC -> extractor CNN de PPO: OK")
            except Exception as e:
                print("Warm-start fallo (se sigue sin el):", repr(e)[:120])

    cbs = [CheckpointCallback(save_freq=max(args.save_every // (args.n_envs * 2), 1),
                              save_path=args.ckpt_dir, name_prefix="gpu"), AnnealCB()]
    remaining = max(args.timesteps - model.num_timesteps, 0)
    print(f"Device={device} | envs={args.n_envs} (x2 asientos) | pasos hechos={model.num_timesteps:,} restantes={remaining:,}")
    if remaining > 0:
        import importlib.util as ilu
        pbar = bool(ilu.find_spec("tqdm") and ilu.find_spec("rich"))
        model.learn(total_timesteps=remaining, callback=cbs, reset_num_timesteps=False, progress_bar=pbar)
    model.save(os.path.join(args.export_dir, "gpu_final"))
    venv.close()
    return model

# ===========================================================================
# Evaluacion (score oficial) + export del StudentAgent RL
# ===========================================================================
def eval_episode(model, mdp, partner, seat, horizon, seed):
    e = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0); e.reset()
    partner.reset(); partner.set_mdp(mdp); partner.set_agent_index(1 - seat)
    soups, tf, tl = 0, None, None
    for t in range(horizon):
        a, _ = model.predict(encode(mdp, e.state, seat, horizon)[None], deterministic=True)
        pj = partner.action(e.state)[0]
        joint = [None, None]; joint[seat] = a2oc(int(a[0])); joint[1 - seat] = pj
        _, sp, done, _ = e.step(tuple(joint))
        if sp > 0:
            soups += int(round(sp / 20)); tf = t if tf is None else tf; tl = t
        if done: break
    return (0.0, 0) if soups == 0 else (float(10000 * soups + 10 * (horizon - tl) + (horizon - tf)), soups)

def evaluate(model, layouts):
    print("\n== Evaluacion (score oficial) ==")
    for lay in layouts:
        if lay not in _MDP_CACHE and _shape_ok(lay) is None: continue
        mdp = get_mdp(lay)
        for pk, mk in [("greedy", lambda s: GreedyFullTaskPolicy(seed=s)), ("stay", lambda s: StayPolicy())]:
            res = [eval_episode(model, mdp, mk(sd), seat, 400, sd) for seat in (0, 1) for sd in (67, 68, 69)]
            print(f"  {lay:22} vs {pk:6}: {np.mean([r[1] for r in res]):.1f} sopas  score={np.mean([r[0] for r in res]):.0f}")

STUDENT_SRC = r'''
# StudentAgent RL (PPO+BC) — carga los pesos y corre la politica; fallback en capas.
from __future__ import annotations
import json, os
from collections import deque
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
def _cnn():
    import torch, torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    class SmallCNN(BaseFeaturesExtractor):
        def __init__(self, s, features_dim=64):
            super().__init__(s, features_dim); c=s.shape[0]
            self.cnn=nn.Sequential(nn.Conv2d(c,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU(),
                                   nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.Flatten())
            with torch.no_grad(): n=self.cnn(torch.zeros(1,*s.shape)).shape[1]
            self.linear=nn.Sequential(nn.Linear(n,features_dim),nn.ReLU())
        def forward(self,x): return self.linear(self.cnn(x))
    return SmallCNN
class StudentAgent:
    def __init__(self, config=None):
        self.meta=json.load(open(os.path.join(_HERE,"gpu_meta.json")))
        self.C,self.P0,self.P1=self.meta["C"],self.meta["P0"],self.meta["P1"]
        self.ws=int(self.meta.get("watchdog_steps",10)); self._h=deque(maxlen=self.ws); self._bi=0; self.model=None
        try:
            from stable_baselines3 import PPO
            pk=dict(features_extractor_class=_cnn(),features_extractor_kwargs=dict(features_dim=64),net_arch=[64,64],normalize_images=False)
            self.model=PPO.load(os.path.join(_HERE,"gpu_model.zip"),device="cpu",
                                custom_objects={"policy_kwargs":pk,"lr_schedule":lambda _:0.,"clip_range":lambda _:0.2,"clip_range_vf":None})
            self.reset()
        except Exception as e: print("StudentAgent fallback:",repr(e))
    def reset(self):
        self._h.clear(); self._bi=0
        if self.model is not None:
            try: self.model.predict(np.zeros((1,self.C,self.P0,self.P1),np.float32),deterministic=True)
            except Exception: pass
    def _pad(self,arr):
        arr=np.asarray(arr,np.float32); d0,d1,c=arr.shape; out=np.zeros((self.C,self.P0,self.P1),np.float32)
        a,b=min(d0,self.P0),min(d1,self.P1); out[:min(c,self.C),:a,:b]=np.transpose(arr[:a,:b,:min(c,self.C)],(2,0,1)); return out
    def act(self,obs):
        try:
            x=self._pad(obs["obs"] if isinstance(obs,dict) else obs)
            a,_=self.model.predict(x[None],deterministic=True); act=int(a[0])
            self._h.append(hash(x.tobytes()))
            if len(self._h)==self._h.maxlen and len(set(self._h))==1:
                self._bi=(self._bi+1)%5; return [5,0,1,2,3][self._bi]
            return act
        except Exception: return 4
'''

def export_student(model, args):
    zip_path = os.path.join(args.export_dir, "gpu_final.zip")
    if not os.path.exists(zip_path): model.save(os.path.join(args.export_dir, "gpu_final"))
    import shutil
    shutil.copy(zip_path, os.path.join(args.out_dir, "gpu_model.zip"))
    json.dump(dict(C=C, P0=P0, P1=P1, obs_shape=list(OBS_SHAPE), horizon=HORIZON, watchdog_steps=10,
                   train_pool=TRAIN_POOL), open(os.path.join(args.out_dir, "gpu_meta.json"), "w"), indent=2)
    open(os.path.join(args.out_dir, "student_agent_gpu.py"), "w").write(STUDENT_SRC)
    print("Exportado:", args.out_dir, "-> student_agent_gpu.py + gpu_model.zip + gpu_meta.json")

# ===========================================================================
def main():
    p = argparse.ArgumentParser(description="Entrenamiento RL para GPU local (BC + dense + self-play + escala)")
    p.add_argument("--layouts", default="cramped_room,asymmetric_advantages,coordination_ring")
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=16, help="numero de JUEGOS (x2 asientos self-play)")
    p.add_argument("--mode", default="selfplay", choices=["selfplay"])
    p.add_argument("--skip-bc", action="store_true")
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--ckpt-dir", default="./gpu_checkpoints")
    p.add_argument("--export-dir", default="./gpu_export")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    layouts = [s for s in args.layouts.split(",") if s.strip()]
    build_pool(layouts)

    if args.eval_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        path = os.path.join(args.export_dir, "gpu_final.zip") or latest_ckpt(args.ckpt_dir)
        if not os.path.exists(path): path = latest_ckpt(args.ckpt_dir)
        model = load_ppo(path, None, device)
        evaluate(model, layouts + ["counter_circuit", "forced_coordination"]); return

    model = train(args)
    if not args.skip_eval:
        evaluate(model, layouts + ["counter_circuit", "forced_coordination"])
    export_student(model, args)

if __name__ == "__main__":
    main()
