
# StudentAgent E3T-lite: politica PPO (SB3) + fallback en capas.
from __future__ import annotations
import json, os
from collections import deque
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

def _build_smallcnn():
    # La MISMA CNN del entrenamiento. Hay que reconstruirla: SB3 no recupera la clase
    # custom por cloudpickle entre versiones de Python (Colab 3.12 -> eval 3.10).
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
            # custom_objects: reconstruye la politica sin depender del cloudpickle guardado.
            custom_objects = {"policy_kwargs": policy_kwargs, "lr_schedule": lambda _: 0.0,
                              "clip_range": lambda _: 0.2, "clip_range_vf": None}
            self.model = PPO.load(os.path.join(_HERE, "e3t_model.zip"),
                                  device="cpu", custom_objects=custom_objects)
            self.reset()  # calentar el forward
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
                return [5, 0, 1, 2, 3][self._break_i]  # interact, luego moverse
            return action
        except Exception:
            # Capa 1: nunca romper el runner. TODO F5: heuristica reactiva parseando el grid (§6.6).
            return 4  # stay
