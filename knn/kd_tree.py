import heapq
import numpy as np
from collections import Counter


class KDTree:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.d = X.shape[1]

        # Flat storage, appended to as we build. Index = node id.
        self._pts = []      # the splitting point at each node
        self._lbl = []      # its label
        self._axis = []     # split axis at each node
        self._left = []     # child node ids (-1 = none)
        self._right = []

        self._build(list(zip(X, y)), depth=0)

        # Freeze into arrays. Everything stays NumPy: traversal is host-side
        # branchy Python, so there's nothing to gain from putting these on
        # the device (and a device round-trip per node would be pure cost).
        self._pts_np = np.array(self._pts, dtype=np.float32)
        self.lbl = np.array(self._lbl)
        self.axis = np.array(self._axis)
        self.left = np.array(self._left)
        self.right = np.array(self._right)
        self.n_classes = int(y.max()) + 1
        return self

    def _build(self, points, depth):
        """Returns the node id of the subtree root, or -1 if empty."""
        if not points:
            return -1
        axis = depth % self.d
        points.sort(key=lambda p: p[0][axis])   # sort by current axis
        mid = len(points) // 2                    # median split

        # Reserve this node's id now, before recursing, so children can
        # refer back. We append placeholders and patch the child links
        # after the recursive calls return.
        node_id = len(self._pts)
        self._pts.append(points[mid][0])
        self._lbl.append(points[mid][1])
        self._axis.append(axis)
        self._left.append(-1)                     # placeholder
        self._right.append(-1)

        l = self._build(points[:mid], depth + 1)
        r = self._build(points[mid + 1:], depth + 1)
        self._left[node_id] = l                   # patch real child ids
        self._right[node_id] = r
        return node_id

    def kneighbors(self, q):
        q = np.asarray(q, dtype=np.float32)
        pts = self._pts_np                       # host-side NumPy view
        heap = []          # max-heap of (-sqdist, counter, label)
        counter = 0

        stack = [0] if len(self.left) else []
        while stack:
            i = stack.pop()
            if i == -1:
                continue

            sq = float(np.sum((q - pts[i]) ** 2))   # squared L2, host-side
            if len(heap) < self.k:
                heapq.heappush(heap, (-sq, counter, self.lbl[i]))
                counter += 1
            elif sq < -heap[0][0]:
                heapq.heapreplace(heap, (-sq, counter, self.lbl[i]))
                counter += 1

            ax = self.axis[i]
            diff = float(q[ax]) - float(pts[i][ax])
            near = self.left[i] if diff < 0 else self.right[i]
            far = self.right[i] if diff < 0 else self.left[i]

            # Stack is LIFO, so push FAR first and NEAR last — that way
            # NEAR is popped and explored first, matching the recursive
            # "search near side, then maybe far side" ordering.
            # Push far only if it can't be pruned: the splitting plane
            # (perpendicular distance diff) must be closer than our
            # current worst neighbor. Compared in squared space.
            if len(heap) < self.k or diff * diff < -heap[0][0]:
                stack.append(far)
            stack.append(near)

        return [lbl for _, _, lbl in sorted(heap, reverse=True)]

    def kneighbor_dists(self, q):
        q = np.asarray(q, dtype=np.float32)
        pts = self._pts_np
        heap = []
        counter = 0
        stack = [0] if len(self.left) else []
        while stack:
            i = stack.pop()
            if i == -1:
                continue
            sq = float(np.sum((q - pts[i]) ** 2))
            if len(heap) < self.k:
                heapq.heappush(heap, (-sq, counter, i)); counter += 1
            elif sq < -heap[0][0]:
                heapq.heapreplace(heap, (-sq, counter, i)); counter += 1
            ax = self.axis[i]
            diff = float(q[ax]) - float(pts[i][ax])
            near = self.left[i] if diff < 0 else self.right[i]
            far = self.right[i] if diff < 0 else self.left[i]
            if len(heap) < self.k or diff * diff < -heap[0][0]:
                stack.append(far)
            stack.append(near)
        return np.sqrt(np.sort([-h[0] for h in heap]))

    def predict(self, Q):
        out = []
        for q in np.asarray(Q, dtype=np.float32):
            labels = self.kneighbors(q)
            out.append(Counter(labels).most_common(1)[0][0])
        return np.array(out)