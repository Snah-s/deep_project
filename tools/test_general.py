"""Test de generalización (CPU, esta máquina).

Mide, para un conjunto de layouts, cuánto rinde NUESTRO agente:
  * solo   = compañero `stay` (pasivo): sopas que produce el agente POR SÍ MISMO.
             Es la señal honesta de generalización — sin esto, greedy puede cargar solo
             y esconder que nuestro agente se queda parado (lo que pasa en scenario_4).
  * equipo = compañero greedy_full_task: score realista de la competencia.

Un agente que generaliza tiene solo>0 en layouts abiertos. En forced_coordination y
similares, solo=0 es esperado (imposible cocinar sin el otro) -> ahí manda `equipo`.

Uso:
  micromamba run -n overcooked python tools/test_general.py                       # default.pt, todos
  micromamba run -n overcooked python tools/test_general.py --weights ruta.pt     # otro checkpoint
  micromamba run -n overcooked python tools/test_general.py --layouts scenario_4 cramped_room
"""
import argparse
import os
import sys

import numpy as np
import torch

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "overcooked"))

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from src.environment import build_mdp
from src.constants import action_index_to_overcooked_action as idx2act
from src.policy_loader import build_builtin_agent

import importlib.util
_spec = importlib.util.spec_from_file_location("student_agent", os.path.join(R, "deliverable/student_agent.py"))
_sa = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sa)

# Pool entrenado + held-out (nunca entrenado). scenario_4 = nuestro layout custom.
TRAINED = ["asymmetric_advantages", "coordination_ring", "counter_circuit", "forced_coordination",
           "scenario_4", "bottleneck", "centre_objects", "corridor", "cramped_room",
           "cramped_room_o_3orders", "five_by_five", "large_room", "m_shaped_s", "pipeline",
           "scenario1_s", "scenario2", "scenario3", "schelling_s", "small_corridor", "unident"]
HELDOUT = ["centre_pots", "long_cook_time", "schelling", "scenario2_s", "scenario4", "simple_o", "tutorial_0"]

CUSTOM = {"scenario_4": os.path.join(R, "overcooked/configs/layouts/scenario_4.layout")}


def _make_env(layout):
    if layout in CUSTOM:
        mdp = build_mdp({"layout_file": CUSTOM[layout], "old_dynamics": True})
    else:
        mdp = build_mdp({"layout_name": layout, "old_dynamics": True})
    return OvercookedEnv.from_mdp(mdp, horizon=250)


def _make_agent(weights):
    agent = _sa.StudentAgent({})
    if weights:
        agent.net.load_state_dict(torch.load(weights, map_location="cpu"))
        agent.net.eval()
    return agent


def _rollout(env, agent, partner_kind, ego_index, seeds):
    total = 0.0
    for sd in seeds:
        env.reset()
        if hasattr(agent, "reset"):
            agent.reset()
        if partner_kind == "greedy":
            partner = build_builtin_agent("greedy_full_task", env)
            partner.set_mdp(env.mdp); partner.set_agent_index(1 - ego_index)
            if hasattr(partner, "reset"):
                partner.reset(); partner.set_mdp(env.mdp); partner.set_agent_index(1 - ego_index)
        for _ in range(250):
            a = agent.act(env.featurize_state_mdp(env.state)[ego_index])
            joint = [None, None]
            joint[ego_index] = idx2act(int(a))
            joint[1 - ego_index] = (0, 0) if partner_kind == "stay" else partner.action(env.state)[0]
            _, _, done, info = env.step(tuple(joint))
            total += sum(info["sparse_r_by_agent"]) / 20.0
            if done:
                break
    return total / len(seeds)


def score(layout, agent, seeds):
    env = _make_env(layout)
    # promedia ambos roles (el grader puede ponernos en cualquier índice)
    solo = np.mean([_rollout(env, agent, "stay", i, seeds) for i in (0, 1)])
    team = np.mean([_rollout(env, agent, "greedy", i, seeds) for i in (0, 1)])
    return solo, team


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(R, "deliverable/weights/default.pt"))
    ap.add_argument("--layouts", nargs="*", default=None, help="subset; default = trained + held-out")
    ap.add_argument("--seeds", type=int, nargs="*", default=[67, 68, 69])
    args = ap.parse_args()

    agent = _make_agent(args.weights)
    print(f"pesos: {args.weights}\n")
    groups = [("TRAINED", TRAINED), ("HELD-OUT", HELDOUT)] if args.layouts is None \
        else [("LAYOUTS", args.layouts)]

    print(f"{'layout':32s} {'solo':>6s} {'equipo':>7s}   señal")
    print("-" * 62)
    for gname, lays in groups:
        print(f"# {gname}")
        for lay in lays:
            try:
                solo, team = score(lay, agent, args.seeds)
                flag = "OK" if solo > 0.5 else ("(coop)" if team > 0.5 else "FALLA")
                print(f"{lay:32s} {solo:6.2f} {team:7.2f}   {flag}")
            except Exception as e:
                print(f"{lay:32s}   error: {e}")


if __name__ == "__main__":
    main()
