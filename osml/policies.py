"""Page replacement policies: FIFO, LRU, Optimal (Belady), and a learned policy.

Every policy implements:
    reset(trace, F)            - prepare for a new run
    on_access(t, page, hit)    - called after every reference
    victim(t, resident)        - choose a page to evict; the incoming page is
                                 NOT yet in `resident`; the policy must also
                                 drop the victim from its own bookkeeping.
"""
from collections import OrderedDict, deque

import numpy as np
from sklearn.tree import DecisionTreeClassifier

INF = 1 << 60


def next_use_array(trace):
    """nxt[t] = index of the next reference to trace[t] after t (INF if none)."""
    n = len(trace)
    nxt = np.full(n, INF, dtype=np.int64)
    last = {}
    for t in range(n - 1, -1, -1):
        p = int(trace[t])
        nxt[t] = last.get(p, INF)
        last[p] = t
    return nxt


class Policy:
    name = "base"

    def reset(self, trace, F):
        self.trace = trace
        self.F = F

    def on_access(self, t, p, hit):
        pass

    def victim(self, t, resident):
        raise NotImplementedError


class FIFO(Policy):
    name = "FIFO"

    def reset(self, trace, F):
        super().reset(trace, F)
        self.q = deque()

    def on_access(self, t, p, hit):
        if not hit:
            self.q.append(p)

    def victim(self, t, resident):
        return self.q.popleft()


class LRU(Policy):
    name = "LRU"

    def reset(self, trace, F):
        super().reset(trace, F)
        self.od = OrderedDict()

    def on_access(self, t, p, hit):
        if hit:
            self.od.move_to_end(p)
        else:
            self.od[p] = None

    def victim(self, t, resident):
        v, _ = self.od.popitem(last=False)
        return v


class OPT(Policy):
    """Belady's optimal: evict the resident page whose next use is farthest away."""
    name = "OPT"

    def reset(self, trace, F):
        super().reset(trace, F)
        self.nxt = next_use_array(trace)
        self.nu = {}

    def on_access(self, t, p, hit):
        self.nu[p] = self.nxt[t]

    def victim(self, t, resident):
        v = max(resident, key=self.nu.__getitem__)
        del self.nu[v]
        return v


# --------------------------------------------------------------------------
# Learned policy
# --------------------------------------------------------------------------
FEATURES = ["recency", "frequency", "age_in_memory", "last_gap"]


class FeatureTracker:
    """Per-resident-page metadata an OS could realistically keep (like reference
    bits / timestamps). State is dropped when a page is evicted."""

    def __init__(self, W=256):
        self.W = W
        self.last, self.load, self.gap, self.hist = {}, {}, {}, {}

    def access(self, t, p, hit):
        if hit:
            self.gap[p] = t - self.last[p]
            self.hist[p].append(t)
        else:
            self.gap[p] = 2 * self.W  # sentinel: no previous reuse seen
            self.load[p] = t
            self.hist[p] = deque([t])
        self.last[p] = t

    def evict(self, p):
        for d in (self.last, self.load, self.gap, self.hist):
            d.pop(p, None)

    def features(self, t, pages):
        X = np.empty((len(pages), 4))
        for i, p in enumerate(pages):
            h = self.hist[p]
            while h and t - h[0] >= self.W:
                h.popleft()
            X[i] = (t - self.last[p], len(h), t - self.load[p], self.gap[p])
        return X


def new_tree():
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0)


def reuse_score(model, X):
    """Predicted probability that each page is reused within the horizon.
    With no model, fall back to LRU behaviour (older = lower score)."""
    if model is None:
        return -X[:, 0]
    proba = model.predict_proba(X)
    cls = list(model.classes_)
    return proba[:, cls.index(1)] if 1 in cls else np.zeros(len(X))


class Learned(Policy):
    """Evict the resident page with the lowest predicted reuse probability.

    A page is labelled 'will be reused' (1) if its next reference falls within
    H accesses of the eviction decision. Ties are broken by recency (LRU-like).

    adaptive=False : the model is trained offline on pre-shift traces and frozen.
    adaptive=True  : the model starts from the offline model but is retrained
                     every `retrain_every` accesses on the most recent
                     eviction decisions whose outcome is already known
                     (labels are consumed only H accesses after the decision,
                     so no future information leaks into training).
    """

    def __init__(self, model=None, adaptive=False, H=64, W=256,
                 retrain_every=1000, buf_rows=6000, min_rows=300, name=None):
        self.init_model = model
        self.adaptive = adaptive
        self.H, self.W = H, W
        self.retrain_every, self.buf_rows, self.min_rows = retrain_every, buf_rows, min_rows
        self.name = name or ("Learned-adaptive" if adaptive else "Learned-static")

    def reset(self, trace, F):
        super().reset(trace, F)
        self.tr = FeatureTracker(self.W)
        self.nxt = next_use_array(trace)
        self.nu = {}
        self.model = self.init_model
        self.pending = deque()   # (t_snap, X, nu_array) awaiting labels
        self.blocks = deque()    # labelled (X, y) blocks
        self.rows = 0
        self.retrains = 0

    def _resolve_pending(self, t):
        while self.pending and self.pending[0][0] + self.H <= t:
            ts, X, nu = self.pending.popleft()
            y = ((nu - ts) <= self.H).astype(int)
            self.blocks.append((X, y))
            self.rows += len(y)
        while self.rows > self.buf_rows and len(self.blocks) > 1:
            X, _ = self.blocks.popleft()
            self.rows -= len(X)

    def _retrain(self):
        if self.rows < self.min_rows:
            return
        X = np.vstack([b[0] for b in self.blocks])
        y = np.concatenate([b[1] for b in self.blocks])
        if len(np.unique(y)) < 2:
            return
        self.model = new_tree().fit(X, y)
        self.retrains += 1

    def on_access(self, t, p, hit):
        self.tr.access(t, p, hit)
        self.nu[p] = self.nxt[t]
        if self.adaptive:
            self._resolve_pending(t)
            if t > 0 and t % self.retrain_every == 0:
                self._retrain()

    def victim(self, t, resident):
        pages = list(resident)
        X = self.tr.features(t, pages)
        score = reuse_score(self.model, X)
        i = int(np.lexsort((-X[:, 0], score))[0])  # lowest score, then oldest
        v = pages[i]
        if self.adaptive:
            self.pending.append((t, X, np.array([self.nu[q] for q in pages])))
        self.tr.evict(v)
        del self.nu[v]
        return v


class _Collector(LRU):
    """LRU eviction that also records (features, label) at every eviction.
    Used only to build the offline training set."""

    def __init__(self, H=64, W=256):
        self.H, self.W = H, W

    def reset(self, trace, F):
        super().reset(trace, F)
        self.tr = FeatureTracker(self.W)
        self.nxt = next_use_array(trace)
        self.nu = {}
        self.Xs, self.ys = [], []

    def on_access(self, t, p, hit):
        super().on_access(t, p, hit)
        self.tr.access(t, p, hit)
        self.nu[p] = self.nxt[t]

    def victim(self, t, resident):
        pages = list(resident)
        self.Xs.append(self.tr.features(t, pages))
        self.ys.append(np.array([(self.nu[q] - t) <= self.H for q in pages], dtype=int))
        v = super().victim(t, resident)
        self.tr.evict(v)
        del self.nu[v]
        return v
