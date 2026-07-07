"""Synthetic-trajectory tests for the trust-ratchet anneal controller.

Each test feeds a hand-built arena-Elo trajectory (with realistic noise:
sigma ~30 Elo at 128 arena games) and asserts the controller is neither too
aggressive (advances through a real regression) nor too regressive (walks
back to full protection on a noisy plateau or a transient dip).

Run:  PYTHONPATH=. python -m pytest jaxarimaa/tests_anneal.py -q
  or  PYTHONPATH=. python jaxarimaa/tests_anneal.py
"""

import numpy as np

from jaxarimaa.anneal import TrustRatchet
from jaxarimaa.config import TrainConfig

SIGMA = 30.0  # arena Elo noise at 128 games


def make_tc(**kw):
    kw.setdefault("anneal_stages", 10)
    return TrainConfig(value_loss_weight=0.25, value_tail_weight=0.0,
                       corpus_mix=0.2, anneal_value_loss_weight=1.0,
                       anneal_value_tail_weight=0.25, anneal_corpus_mix=0.0, **kw)


def test_knob_endpoints_and_midpoint():
    r = TrustRatchet(make_tc())
    assert r.knobs() == (0.25, 0.0, 0.2)
    r.stage = 5
    vw, tw, mix = r.knobs()
    assert np.allclose((vw, tw, mix), (0.625, 0.125, 0.1))
    r.stage = 10
    assert r.knobs() == (1.0, 0.25, 0.0)


def test_disabled_is_inert():
    r = TrustRatchet(make_tc(anneal_stages=0))
    assert r.knobs() == (0.25, 0.0, 0.2)
    assert not r.update(500.0) and r.stage == 0


def test_healthy_climb_reaches_full_without_retreats():
    r, rng = TrustRatchet(make_tc()), np.random.default_rng(0)
    stages = [r.stage]
    for i in range(15):
        r.update(40.0 * i + rng.normal(0, SIGMA))
        stages.append(r.stage)
    assert r.stage == 10, stages
    assert all(b >= a for a, b in zip(stages, stages[1:])), stages  # monotone


def test_cliff_retreats_to_zero_and_holds():
    r = TrustRatchet(make_tc())
    for i in range(6):
        r.update(40.0 * i)
    assert r.stage == 6
    for _ in range(12):
        r.update(-250.0)  # hard, persistent regression (probe-scale collapse)
    assert r.stage == 0
    for _ in range(20):
        r.update(-250.0)
    assert r.stage == 0  # parks at full protection for dozens of rounds
    # ... but a PERMANENT new level is eventually accepted as the baseline
    # (leaky best) and the controller re-probes rather than parking forever
    for _ in range(100):
        r.update(-250.0)
    assert r.stage > 0


def test_transient_dip_retreats_then_recovers():
    r = TrustRatchet(make_tc())
    for i in range(6):
        r.update(40.0 * i)  # healthy to stage 6, best ~ ema of climb
    assert r.stage == 6
    r.update(-100.0)                     # one bad arena: must back off
    assert r.stage == 5
    for i in range(8):
        r.update(220.0 + 10.0 * i)       # recovery past the old level
    assert r.stage == 10


def test_noisy_plateau_drifts_up_not_down():
    """A genuine plateau (elo ~ N(0, sigma) forever) must NOT walk back to full
    protection off a lucky early reading — the EMA-best keeps the baseline
    honest and the controller keeps probing toward full AlphaZero."""
    r, rng = TrustRatchet(make_tc()), np.random.default_rng(1)
    for _ in range(60):
        r.update(rng.normal(0, SIGMA))
    assert r.stage >= 8, r.stage
    # and across many seeds it must never end anywhere near fully regressed
    finals = []
    for seed in range(20):
        r, rng = TrustRatchet(make_tc()), np.random.default_rng(seed)
        for _ in range(60):
            r.update(rng.normal(0, SIGMA))
        finals.append(r.stage)
    assert min(finals) >= 5, finals


def test_raw_best_would_fail_plateau():
    """Documents WHY best is EMA-smoothed: with best = max(raw readings), a
    plateau decays toward full protection (the flaw the EMA fixes). If this
    test ever fails, the raw-max flaw no longer reproduces and the EMA can be
    reconsidered."""
    tc = make_tc()
    rng = np.random.default_rng(1)
    stage, best = 0, float("-inf")
    for _ in range(200):
        elo = rng.normal(0, SIGMA)
        if elo >= best - tc.anneal_hold_band:
            stage = min(stage + 1, tc.anneal_stages)
        elif elo < best - tc.anneal_backoff:
            stage = max(stage - 1, 0)
        best = max(best, elo)  # raw max: ratchets on noise
    assert stage <= 5, stage  # decays; the EMA version stays >= 8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"all {len(fns)} tests passed")
