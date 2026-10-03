# Learned Page Replacement Under a Workload Shift

CSE-307 Operating Systems, Part B term paper, **Track 1: Learned Page Replacement (Memory Management)**.

A page-replacement simulator with FIFO, LRU and Optimal (Belady), plus a lightweight learned eviction
policy (a depth-4 decision tree on recency / frequency / age / reuse-gap features). The policies are
compared on synthetic traces whose access pattern changes half-way through.

## Repository layout

```
osml/
  traces.py      synthetic trace generator (phase 1: locality + sequential scan, phase 2: random/bursty)
  policies.py    FIFO, LRU, OPT, Learned (static and adaptive), feature tracker
  sim.py         simulator, per-phase metrics, offline training of the static model
tests/           sanity tests (textbook fault counts, OPT is a lower bound)
run_experiments.py   runs everything and writes results/
diagnostics.py       measures *why* the learned policy degrades after the shift
results/         raw_runs.csv, summary.csv/json, table_main.md/.tex, 3 figures, static_tree.txt, diagnostics.json
report/          report.tex (LaTeX source) and report.pdf
```

## Setup

Python 3.10+ is required.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

```bash
python tests/test_policies.py     # quick correctness checks (prints "all tests passed")
python run_experiments.py         # full run, about 3 minutes; use --seeds 3 for a quick run
python diagnostics.py             # optional mechanism check (writes results/diagnostics.json)
cd report && latexmk -pdf report.tex
```

All randomness is seeded, so repeated runs give identical numbers.

## Experimental design

* **Trace:** 20,000 references over 256 virtual pages; the shift happens at reference 10,000.
  * Phase 1: 70% of references go to a 20-page Zipf-weighted hot set, 30% are a cyclic sequential scan of the other pages.
  * Phase 2: uniform-random references with short bursts (10-40 references) on random 8-page regions.
* **Memory:** 32 frames (main result); 16 and 64 frames in the sweep.
* **Learned policy:** the tree predicts "will this page be reused within 64 accesses?"; the page with the lowest
  predicted probability is evicted (ties broken by recency).
  * `Learned-static`: trained offline on three *pre-shift* traces (seeds 1000-1002, never used for testing), then frozen.
  * `Learned-adaptive`: same start, retrained every 1000 accesses on the latest 6000 labelled decisions.
    Labels are only used 64 accesses after the decision was made, so no future information leaks into training.
* **Metrics:** hit ratio and page faults for each half of the trace, mean +/- std over 10 seeds.

## Results summary (32 frames, 10 traces)

| Policy | Hit% before | Hit% after | Drop (pp) | Faults before | Faults after |
|---|---|---|---|---|---|
| FIFO | 52.7 | 38.3 | 14.4 | 4732 | 6171 |
| LRU | 60.5 | 38.4 | 22.1 | 3946 | 6158 |
| OPT | 71.1 | 59.1 | 12.0 | 2890 | 4086 |
| Learned-static | 67.3 | 34.2 | 33.1 | 3267 | 6578 |
| Learned-adaptive | 69.2 | 36.0 | 33.2 | 3080 | 6396 |

Faults are per 10,000 references. The full table with standard deviations is in `results/table_main.md`.

* Before the shift the learned policies beat LRU by 7-9 pp (the tree is almost purely a frequency rule, which
  resists scan pollution).
* After the shift they fall below LRU and lose the most (about 33 pp). Diagnostics show 10.8% premature
  evictions versus 4.9% for LRU, and about 12 of 32 frames held by stale "hot" pages.
* Online retraining recovers about 2 pp but does not beat LRU after the shift.

Figures: `results/fig_phase_hit_ratio.png`, `results/fig_rolling_hit_ratio.png`, `results/fig_frame_sweep.png`.

## Limitations

Synthetic traces with a single, hand-designed shift; a small tree evaluated in a Python simulator (no kernel
overhead model); OPT is an offline reference only. Conclusions about the ranking after the shift depend on how
bursty phase 2 is.

## AI assistance disclosure

An AI assistant (Claude, by Anthropic) was used to help write the simulator code, the experiment and
diagnostics scripts, the tests, this README, and a first draft of the report. The code was run and the numbers
in the report and this README come from the outputs in `results/`. The author is responsible for reviewing all
of it and for being able to explain every part.

## References

Belady (1966); Mattson et al. (1970); Silberschatz, Galvin and Gagne, *Operating System Concepts* (10th ed.);
Vietri et al., LeCaR (HotStorage 2018); scikit-learn (Pedregosa et al., 2011); NumPy (Harris et al., 2020).
Full entries are in `report/report.tex`.
