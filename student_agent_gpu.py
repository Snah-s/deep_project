
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
