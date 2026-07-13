"""Extrae los pesos de la política SB3 a torch puro para el entregable — PLAN.md Etapa 7.

Convierte un checkpoint SB3 (MlpPolicy 96->256->256->6, tanh) en un state_dict de un
`nn.Sequential` sin dependencia de stable-baselines3, y VERIFICA que reproduce la política
original (argmax idéntico en obs aleatorias) antes de guardarlo.

Uso:
    micromamba run -n overcooked python scripts/export_weights.py \
        --checkpoint esc_enhanced_multi/best_model \
        --out deliverable/weights/default.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn


def build_net() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(96, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, 6),
    )


def policy_to_net(policy) -> nn.Sequential:
    """Remapea el state_dict de una MlpPolicy SB3 (en memoria) al nn.Sequential entregable."""
    sd = policy.state_dict()
    net = build_net()
    with torch.no_grad():
        net[0].weight.copy_(sd["mlp_extractor.policy_net.0.weight"])
        net[0].bias.copy_(sd["mlp_extractor.policy_net.0.bias"])
        net[2].weight.copy_(sd["mlp_extractor.policy_net.2.weight"])
        net[2].bias.copy_(sd["mlp_extractor.policy_net.2.bias"])
        net[4].weight.copy_(sd["action_net.weight"])
        net[4].bias.copy_(sd["action_net.bias"])
    net.eval()
    return net


def save_deliverable(model, out: str) -> None:
    """Guarda los pesos entregable (.pt torch puro) desde un PPO en memoria. Sin zip."""
    torch.save(policy_to_net(model.policy).state_dict(), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="ruta al .zip SB3 (sin extensión ok)")
    ap.add_argument("--out", default="deliverable/weights/default.pt")
    ap.add_argument("--verify-n", type=int, default=2000)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.checkpoint, device="cpu")
    net = policy_to_net(model.policy)

    # Verificación: argmax(net) == predict determinista de SB3.
    rng = np.random.default_rng(0)
    mism = 0
    with torch.no_grad():
        for _ in range(args.verify_n):
            o = rng.normal(size=96).astype(np.float32)
            a_sb3, _ = model.predict(o, deterministic=True)
            if int(a_sb3) != int(net(torch.from_numpy(o)).argmax().item()):
                mism += 1
    if mism:
        raise SystemExit(f"ERROR: la extracción NO reproduce SB3 ({mism}/{args.verify_n} mismatches)")

    torch.save(net.state_dict(), args.out)
    print(f"OK: {args.verify_n}/{args.verify_n} idénticos -> {args.out}")


if __name__ == "__main__":
    main()
