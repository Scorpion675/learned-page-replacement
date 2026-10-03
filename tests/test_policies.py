"""Sanity tests. Run with:  python -m pytest -q   (or: python tests/test_policies.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osml.policies import FIFO, LRU, OPT, Learned  # noqa: E402
from osml.sim import simulate  # noqa: E402
from osml.traces import gen_trace  # noqa: E402

# Textbook reference string (Silberschatz et al.), 3 frames: FIFO=15, LRU=12, OPT=9 faults
REF = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]


def faults(policy, trace, F):
    return int((~simulate(np.array(trace), F, policy)).sum())


def test_textbook_fault_counts():
    assert faults(FIFO(), REF, 3) == 15
    assert faults(LRU(), REF, 3) == 12
    assert faults(OPT(), REF, 3) == 9


def test_opt_is_lower_bound_on_synthetic_trace():
    trace, _ = gen_trace(seed=123, n_phase=2000)
    f_opt = faults(OPT(), trace, 32)
    for pol in (FIFO(), LRU(), Learned(model=None), Learned(model=None, adaptive=True)):
        assert faults(pol, trace, 32) >= f_opt


def test_learned_without_model_equals_lru():
    trace, _ = gen_trace(seed=5, n_phase=2000)
    assert faults(Learned(model=None), trace, 32) == faults(LRU(), trace, 32)


if __name__ == "__main__":
    test_textbook_fault_counts()
    test_opt_is_lower_bound_on_synthetic_trace()
    test_learned_without_model_equals_lru()
    print("all tests passed")
