import heapq
import numpy as np
from collections import Counter


class BallTree:
    """
    Ball-tree K-NN. Nested hyperspheres + triangle-inequality pruning.
    """

    def __init__(self, k=5, leaf_size=20):
        self.k = k
        self.leaf_size = leaf_size

    # ------------------------------------------------------------------
    # BUILD — flat arrays indexed by node id, mirroring the KDTree layout.
    # Internal nodes have children; leaves carry a bucket of point indices.
    # ------------------------------------------------------------------
    def fit(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y)
        self.n_classes = int(self.y.max()) + 1

        self._center = []      # ball center per node
        self._radius = []      # ball radius per node
        self._left = []        # child ids (-1 = none)
        self._right = []
        self._bucket = []      # leaf: array of point indices; internal: None

        # Build over indices into self.X so leaves can store cheap int ids.
        self._build(np.arange(len(self.X)))

        self._center = np.array(self._center, dtype=np.float32)
        self._radius = np.array(self._radius, dtype=np.float32)
        self._left = np.array(self._left)
        self._right = np.array(self._right)
        return self

    def _build(self, idx):
        """Build a subtree over point indices `idx`; return its node id."""
        pts = self.X[idx]
        center = pts.mean(axis=0)
        # Radius = distance from center to the farthest enclosed point.
        radius = float(np.sqrt(np.max(np.sum((pts - center) ** 2, axis=1))))

        node_id = len(self._center)
        self._center.append(center) # type: ignore
        self._radius.append(radius) # type: ignore
        self._left.append(-1) # type: ignore
        self._right.append(-1) # type: ignore
        self._bucket.append(None)

        if len(idx) <= self.leaf_size:           # leaf: store the bucket
            self._bucket[node_id] = idx
            return node_id

        # Split along the dimension of greatest spread, at its median.
        spread = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spread))
        median = np.median(pts[:, axis])
        mask = pts[:, axis] < median
        # Degenerate guard: if everything lands on one side (e.g. many
        # ties at the median), force a balanced split so we still recurse.
        if mask.all() or not mask.any():
            mask = np.arange(len(idx)) < len(idx) // 2

        self._left[node_id] = self._build(idx[mask])
        self._right[node_id] = self._build(idx[~mask])
        return node_id

    # ------------------------------------------------------------------
    # QUERY — explicit stack, triangle-inequality pruning. Distances kept
    # squared in the heap to avoid sqrt in the hot loop; the prune test is
    # done in real distance because the bound (d_center - radius) isn't
    # linear under squaring.
    # ------------------------------------------------------------------
    def _search(self, q):
        q = np.asarray(q, dtype=np.float32)
        heap = []          # max-heap of (-sqdist, counter, point_index)
        counter = 0
        stack = [0] if len(self._center) else []

        while stack:
            i = stack.pop()
            if i == -1:
                continue

            # Prune: closest possible point in this ball is d_center - radius.
            d_center = np.sqrt(np.sum((q - self._center[i]) ** 2))
            if len(heap) == self.k:
                worst = np.sqrt(-heap[0][0])      # current k-th distance
                if d_center - self._radius[i] >= worst:
                    continue                       # whole ball too far — skip

            if self._bucket[i] is not None:        # leaf: brute-force bucket
                bidx = self._bucket[i]
                d2 = np.sum((self.X[bidx] - q) ** 2, axis=1)
                for j, sq in zip(bidx, d2):
                    sq = float(sq)
                    if len(heap) < self.k:
                        heapq.heappush(heap, (-sq, counter, j)); counter += 1
                    elif sq < -heap[0][0]:
                        heapq.heapreplace(heap, (-sq, counter, j)); counter += 1
            else:                                  # internal: nearer child first
                l, r = self._left[i], self._right[i]
                dl = np.sum((q - self._center[l]) ** 2)
                dr = np.sum((q - self._center[r]) ** 2)
                near, far = (l, r) if dl < dr else (r, l)
                # LIFO stack: push FAR first so NEAR is popped/explored first.
                stack.append(far)
                stack.append(near)

        return heap

    def kneighbor_dists(self, q):
        """Sorted distances to the k nearest — for correctness checks
        against brute force (the true invariant; labels can differ on ties)."""
        heap = self._search(q)
        return np.sqrt(np.sort([-h[0] for h in heap]))

    def kneighbors(self, q):
        heap = self._search(q)
        order = sorted(heap, reverse=True)         # nearest first
        return [self.y[j] for _, _, j in order]

    def predict(self, Q):
        out = []
        for q in np.asarray(Q, dtype=np.float32):
            labels = self.kneighbors(q)
            out.append(Counter(labels).most_common(1)[0][0])
        return np.array(out)