"""Cache simulator and offline training of the learned eviction model."""
import numpy as np

from .policies import _Collector, new_tree
from .traces import gen_trace


def simulate(trace, F, policy):
    """Run `policy` over `trace` with F frames; return a boolean hit array."""
    policy.reset(trace, F)
    resident = set()
    hits = np.zeros(len(trace), dtype=bool)
    for t in range(len(trace)):
        p = int(trace[t])
        if p in resident:
            hits[t] = True
            policy.on_access(t, p, True)
        else:
            if len(resident) >= F:
                v = policy.victim(t, resident)
                resident.remove(v)
            resident.add(p)
            policy.on_access(t, p, False)
    return hits


def phase_metrics(hits, boundary):
    out = {}
    for name, seg in (("before", hits[:boundary]), ("after", hits[boundary:])):
        h = int(seg.sum())
        out[name] = dict(hits=h, faults=len(seg) - h, hit_ratio=h / len(seg))
    return out


def train_static_model(F, H=64, W=256, seeds=(1000, 1001, 1002), n_phase=10_000, V=256):
    """Train the frozen model on PRE-SHIFT traces only (different seeds from
    every evaluation trace)."""
    Xs, ys = [], []
    for s in seeds:
        trace, _ = gen_trace(s, n_phase=n_phase, V=V, shift=False)
        col = _Collector(H=H, W=W)
        from .sim import simulate as _sim
        _sim(trace, F, col)
        Xs += col.Xs
        ys += col.ys
    X, y = np.vstack(Xs), np.concatenate(ys)
    model = new_tree().fit(X, y)
    return model, dict(rows=len(y), positive_rate=float(y.mean()))
