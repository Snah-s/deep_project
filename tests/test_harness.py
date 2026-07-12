"""GATE 2 — Harness de score oficial (ver PLAN.md, Etapa 2).

Criterio de avance: el harness reproduce los dos casos de control y su salida incluye
{score_mean, soups_mean, timeouts_total}.

Nota sobre el layout del control greedy+greedy: la greedy del profesor es
"intencionalmente no óptima" y se ATASCA en `cramped_room` (0 sopas con cualquier
avoid_teammate). Sí coopera en otros layouts (coordination_ring=8, counter_circuit=3,
asymmetric_advantages=1). Por eso el control "greedy entrega" se valida en
`coordination_ring`; el control "stay da 0" se valida en `cramped_room`. Ambos casos
de control del GATE 2 (soups>0 y score=0) quedan cubiertos. Ver EXPERIMENTS.md.
"""

from __future__ import annotations

import pytest

from evaluation.harness import (
    DEFAULT_SEEDS,
    evaluate,
    official_score,
    agent_ctor_from_spec,
    spec_from_string,
)


REQUIRED_KEYS = {"score_mean", "soups_mean", "timeouts_total"}


# ------------------------------------------------------------- score unit tests
def test_official_score_zero_soups_is_zero():
    assert official_score(soups=0, t_first_soup=10, t_last_soup=200, timeouts=0, horizon=250) == 0.0


def test_official_score_formula():
    # 2 sopas, primera en t=50, última en t=200, 0 timeouts, horizon 250
    # 10000*2 + 10*(250-200) + (250-50) - 0 = 20000 + 500 + 200 = 20700
    assert official_score(2, 50, 200, 0, 250) == pytest.approx(20700.0)


def test_official_score_timeout_penalty_capped():
    # penalización = min(100*timeouts, 5000); 100 timeouts -> cap 5000
    base = official_score(1, 10, 10, 0, 250)
    capped = official_score(1, 10, 10, 100, 250)
    assert base - capped == pytest.approx(5000.0)


# ----------------------------------------------------------------- control: stay
def test_stay_pair_scores_zero():
    """stay+stay: 0 sopas -> score 0 (cualquier layout)."""
    result = evaluate(
        agent_ctor=agent_ctor_from_spec({"type": "stay"}),
        layout="cramped_room",
        partner_spec={"type": "stay"},
        seeds=[67, 68],
    )
    assert REQUIRED_KEYS <= set(result.keys())
    assert result["soups_mean"] == 0.0
    assert result["score_mean"] == 0.0
    assert result["timeouts_total"] == 0


# --------------------------------------------------------------- control: greedy
def test_greedy_pair_delivers_and_scores():
    """greedy+greedy en coordination_ring entrega sopas -> score > 0."""
    result = evaluate(
        agent_ctor=agent_ctor_from_spec({"type": "greedy"}),
        layout="coordination_ring",
        partner_spec={"type": "greedy"},
        seeds=DEFAULT_SEEDS,
    )
    assert REQUIRED_KEYS <= set(result.keys())
    assert result["soups_mean"] > 0, f"greedy+greedy debe entregar sopas, got {result['soups_mean']}"
    assert result["score_mean"] > 0
    assert result["timeouts_total"] == 0  # agentes scripted no hacen timeout


# ------------------------------------------------------------------- estructura
def test_output_structure_and_role_swap():
    result = evaluate(
        agent_ctor=agent_ctor_from_spec({"type": "greedy"}),
        layout="coordination_ring",
        partner_spec={"type": "greedy"},
        seeds=[67, 68, 69],
    )
    assert result["num_attempts"] == 6  # 3 seeds x 2 roles
    assert len(result["per_seed"]) == 3
    for seed_block in result["per_seed"]:
        assert len(seed_block["attempts"]) == 2
        roles = {a["role_swap"] for a in seed_block["attempts"]}
        assert roles == {False, True}  # ambos roles cubiertos


def test_spec_from_string():
    assert spec_from_string("greedy") == {"type": "greedy"}
    assert spec_from_string("checkpoint:/tmp/m.zip") == {"type": "checkpoint", "path": "/tmp/m.zip"}
