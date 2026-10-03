"""Mechanism check: WHY does the learned policy do worse than LRU after the shift?

For the post-shift phase only (32 frames, same traces as the main experiment) we measure
  premature_rate : share of evictions whose victim is referenced again within 16 accesses
                   (a direct measure of bad eviction decisions)
  stale_protected: mean number of resident pages that the model considers 'likely to be
                   reused' (score >= 0.5) although they have not been touched for > 100
                   accesses (Learned-static only)
Writes results/diagnostics.json
"""
import bisect
import json
from collections import defaultdict

import numpy as np

from osml.policies import LRU, Learned, reuse_score
from osml.sim import simulate, train_static_model
from osml.traces import gen_trace

F, SEEDS, K = 32, range(10), 16


def run(policy, trace, boundary, probe_stale=False):
    log, stale = [], []
    orig = policy.victim

    def wrapped(t, resident):
        if probe_stale and t >= boundary:
            pages = list(resident)
            X = policy.tr.features(t, pages)
            sc = reuse_score(policy.model, X)
            stale.append(int(((X[:, 0] > 100) & (sc >= 0.5)).sum()))
        v = orig(t, resident)
        log.append((t, v))
        return v

    policy.victim = wrapped
    simulate(trace, F, policy)
    pos = defaultdict(list)
    for i, p in enumerate(trace):
        pos[int(p)].append(i)
    ev = [(t, v) for t, v in log if t >= boundary]
    prem = 0
    for t, v in ev:
        j = bisect.bisect_right(pos[v], t)
        if j < len(pos[v]) and pos[v][j] - t <= K:
            prem += 1
    return prem / max(len(ev), 1), (float(np.mean(stale)) if stale else None)


def main():
    model, _ = train_static_model(F)
    res = {"LRU": [], "Learned-static": [], "stale": []}
    for s in SEEDS:
        trace, b = gen_trace(s)
        res["LRU"].append(run(LRU(), trace, b)[0])
        pr, st = run(Learned(model), trace, b, probe_stale=True)
        res["Learned-static"].append(pr)
        res["stale"].append(st)
    out = dict(premature_eviction_rate_after_shift=dict(
        LRU=float(np.mean(res["LRU"])), Learned_static=float(np.mean(res["Learned-static"]))),
        mean_stale_protected_pages_after_shift=float(np.mean(res["stale"])),
        frames=F, window_K=K, seeds=len(list(SEEDS)))
    json.dump(out, open("results/diagnostics.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
