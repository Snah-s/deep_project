"""Normalización de la observación featurizada (96,) de Overcooked.

Único punto de verdad: lo usan el ego (ego_env), el partner-checkpoint (partners)
y DEBE replicarlo el entregable (deliverable/student_agent.py) idéntico, o hay
mismatch train/serve.

`cook_time_remaining` es la única feature cruda que se dispara (~0..20) mientras
el resto vive en [-5, 6]; entra ~4x más grande a un MLP sin VecNormalize. La
escalamos a ~[0,1]. Sus índices en el vector de 96 son constantes (num_pots=2,
2 jugadores): las 2 ollas del ego + las 2 del otro.
"""
import numpy as np

COOK_TIME_IDX = (29, 39, 75, 85)   # p{ego}_pot_{0,1}_cook_time + p{other}_pot_{0,1}_cook_time
# cook_time varía por layout (20 en la mayoría, 45 en counter_circuit). 50 es un bound
# generoso que deja todo en ~[0,1] sin necesitar la mdp en inferencia.
# ponytail: constante fija; si un layout futuro tiene cook_time > 50, pasará de 1.0.
COOK_TIME_SCALE = 50.0


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Devuelve una copia del vector (96,) con cook_time escalado a ~[0,1]."""
    obs = np.asarray(obs, dtype=np.float32).copy()
    obs[..., list(COOK_TIME_IDX)] /= COOK_TIME_SCALE
    return obs


if __name__ == "__main__":
    # Self-check: los índices son cook_time y quedan en [0,1] tras normalizar.
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, SoupState
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    for lay in ["counter_circuit", "forced_coordination"]:
        mdp = OvercookedGridworld.from_layout_name(lay, old_dynamics=True)
        env = OvercookedEnv.from_mdp(mdp, horizon=250)
        st = env.state.deepcopy()
        for p in mdp.get_pot_locations():
            st.add_object(SoupState.get_soup(p, num_onions=3, num_tomatoes=0, cooking_tick=1))
        raw = mdp.featurize_state(st, env.mlam, num_pots=2)[0]
        assert raw[list(COOK_TIME_IDX)].max() >= 15, f"{lay}: cook_time no está en esos índices"
        norm = normalize_obs(raw)
        assert norm[list(COOK_TIME_IDX)].max() <= 1.0, f"{lay}: cook_time no quedó en [0,1]"
        # el resto del vector no cambió
        rest = [i for i in range(96) if i not in COOK_TIME_IDX]
        assert np.array_equal(norm[rest], raw[rest]), "se tocaron features que no eran cook_time"
    print("OK cook_time idx", COOK_TIME_IDX, "-> [0,1], resto intacto")
