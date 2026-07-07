"""Trust-ratchet anneal controller (see TrainConfig.anneal_*).

Walks the value-conservative warm-start knobs (value_loss_weight,
value_tail_weight, corpus_mix) from their configured start values toward the
anneal_* finals in `anneal_stages` equal steps, gated on the arena anchor-Elo:
advance one stage per arena round while the Elo is healthy, retreat when it
regresses (probe-and-back-off, TCP-style). The knobs are traced scalars in
train_step, so stage changes never recompile.

Health baseline: `best` is the max of an EMA-smoothed Elo (a single lucky
+3-sigma arena result must not permanently inflate the baseline) and it LEAKS
by `anneal_best_leak` Elo per round (a genuine plateau eventually re-probes
toward full AlphaZero instead of parking regressed forever — TCP slow-start
after timeout). Decisions still compare the RAW current Elo against `best`,
so a real cliff triggers a retreat on the very next reading, and the small
leak keeps it parked at full protection for dozens of rounds.
"""


class TrustRatchet:
    def __init__(self, tc, stage=0):
        self.tc = tc
        self.stage = min(int(stage), tc.anneal_stages) if tc.anneal_stages else 0
        self.best = float("-inf")
        self.ema = None

    def knobs(self):
        """(value_loss_weight, value_tail_weight, corpus_mix) at the current stage."""
        tc = self.tc
        t = self.stage / tc.anneal_stages if tc.anneal_stages else 0.0
        lerp = lambda a, b: a + (b - a) * t
        return (lerp(tc.value_loss_weight, tc.anneal_value_loss_weight),
                lerp(tc.value_tail_weight, tc.anneal_value_tail_weight),
                lerp(tc.corpus_mix, tc.anneal_corpus_mix))

    def update(self, elo):
        """Feed a new arena Elo reading; returns True if the stage changed
        (caller should re-read knobs()). Within hold_band of best -> advance;
        more than backoff below best -> retreat; between -> hold."""
        tc = self.tc
        if not tc.anneal_stages:
            return False
        prev = self.stage
        if elo >= self.best - tc.anneal_hold_band:
            self.stage = min(self.stage + 1, tc.anneal_stages)
        elif elo < self.best - tc.anneal_backoff:
            self.stage = max(self.stage - 1, 0)
        self.ema = elo if self.ema is None else 0.5 * self.ema + 0.5 * elo
        self.best = max(self.ema, self.best - tc.anneal_best_leak)
        return self.stage != prev
