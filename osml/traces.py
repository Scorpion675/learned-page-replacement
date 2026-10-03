"""Synthetic page-reference trace generation with a deliberate mid-trace shift.

Phase 1 (locality-heavy / sequential):
    70% of accesses hit a small Zipf-weighted "hot" set (temporal locality),
    30% are a cyclic sequential scan over the remaining ("cold") pages.
    The scan is the classic pattern that pollutes recency-based policies.

Phase 2 (random / bursty):
    mostly uniform-random accesses over the whole address space, interrupted
    by short bursts that hammer a small region of pages for 10-40 accesses.
    There is no stable hot set any more, so past behaviour is a poor guide.
"""
import numpy as np


def gen_phase1(rng, n, V, hot=20, p_hot=0.7, zipf_s=1.0):
    pages = rng.permutation(V)
    hot_pages = pages[:hot]
    cold_pages = np.sort(pages[hot:])  # scan visits cold pages in address order
    w = 1.0 / np.arange(1, hot + 1) ** zipf_s
    w /= w.sum()
    is_hot = rng.random(n) < p_hot
    hot_draw = rng.choice(hot_pages, size=n, p=w)
    out = np.empty(n, dtype=np.int64)
    ptr = 0
    for i in range(n):
        if is_hot[i]:
            out[i] = hot_draw[i]
        else:
            out[i] = cold_pages[ptr]
            ptr = (ptr + 1) % len(cold_pages)
    return out


def gen_phase2(rng, n, V, burst_prob=0.03, region=8, burst_len=(10, 40)):
    out = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        if rng.random() < burst_prob:
            base = int(rng.integers(0, V - region + 1))
            m = min(int(rng.integers(*burst_len)), n - i)
            out[i:i + m] = base + rng.integers(0, region, size=m)
            i += m
        else:
            out[i] = rng.integers(0, V)
            i += 1
    return out


def gen_trace(seed, n_phase=10_000, V=256, shift=True):
    """Return (trace, boundary). boundary is the index where phase 2 starts."""
    rng = np.random.default_rng(seed)
    p1 = gen_phase1(rng, n_phase, V)
    if not shift:
        return p1, len(p1)
    p2 = gen_phase2(rng, n_phase, V)
    return np.concatenate([p1, p2]), len(p1)
