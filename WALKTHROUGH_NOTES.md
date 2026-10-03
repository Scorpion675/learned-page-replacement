# 3-5 minute demo: talking points

**0:00 Problem (30 s).** Page replacement picks a victim when memory is full. FIFO and LRU are fixed rules, OPT
needs the future. Question: can a tiny learned model beat LRU, and what happens when the workload changes?

**0:30 Code tour (90 s).**
- `osml/traces.py`: phase 1 = hot set + sequential scan, phase 2 = random + bursts, shift at reference 10,000.
- `osml/policies.py`: FIFO (queue), LRU (ordered dict), OPT (next-use array, evict farthest).
  Learned: 4 features per resident page -> decision tree predicts "reused within 64 accesses?" -> evict lowest probability.
- `osml/sim.py`: one loop for every policy; hits are split into before/after the shift.
- Static model = trained once on pre-shift traces. Adaptive = retrained every 1000 accesses on recent decisions
  (labels only used 64 accesses later, so no cheating).

**2:00 Live run (60 s).** `python tests/test_policies.py` (FIFO 15 / LRU 12 / OPT 9 faults on the textbook string), then
show `results/table_main.md` and `fig_rolling_hit_ratio.png`.

**3:00 Findings (90 s).**
- Before shift: learned 67-69% vs LRU 60.5%. The tree is ~99% a frequency rule, which ignores one-time scan pages.
- After shift: everyone drops, learned drops most (~33 pp) and ends below LRU (34-36% vs 38.4%).
- Why: stale high-frequency pages keep frames (about 12 of 32), premature evictions 10.8% vs 4.9% for LRU.
- Adaptive retraining helps a little (+1.8 pp) but lags; OPT still far ahead -> future knowledge is the real gap.
- Lesson: learned OS heuristics need drift detection and an LRU-style fallback.

## Likely questions - make sure you can answer
1. Why is OPT not implementable? (needs future references)
2. Why do FIFO and LRU end up equal after the shift? (no exploitable recency/locality in random references)
3. Why is the learned label "reused within H=64"? (an approximation of "far next use", cheap to compute from history)
4. Why can't the labels be used immediately in the adaptive version? (the outcome is unknown until 64 accesses later)
5. What would you change with real traces? (replace the generator, keep the simulator; add drift detection)
