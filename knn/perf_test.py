import time
import numpy as np

try:
    import jax
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from knn import KNN
from kd_tree import KDTree
from ball_tree import BallTree


def make_data(n_train, n_query, d, n_classes=3):
    """Deterministic dataset so bench() and verify() see identical inputs."""
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(n_train, d)).astype(np.float32)
    ytr = rng.integers(0, n_classes, size=n_train)
    Xte = rng.normal(size=(n_query, d)).astype(np.float32)
    return Xtr, ytr, Xte


def _make_model(algo, k):
    if algo == 'KDTree':
        return KDTree(k=k)
    if algo == 'BallTree':
        return BallTree(k=k)
    return KNN(k=k)


def verify(n_train, n_query, d, k=5, n_classes=3, algo='KDTree'):
    """Confirm a tree finds the same nearest neighbors as brute force.

    We compare neighbor *distances*, not predicted labels. Labels can
    legitimately differ on a tied class vote (e.g. neighbors split 2-2-1),
    where the two methods break the tie toward different classes — both
    correct. Distances are the true algorithmic invariant: if the tree
    pruned wrongly, a returned distance would be larger than brute force's.
    """
    Xtr, ytr, Xte = make_data(n_train, n_query, d, n_classes)

    bf = KNN(k=k).fit(Xtr, ytr)
    tree = _make_model(algo, k).fit(Xtr, ytr)

    bad = 0
    for q in Xte:
        d_bf = np.asarray(bf.kneighbors(q)[0])      # sorted distances (k,)
        d_tree = tree.kneighbor_dists(q)             # sorted distances (k,)
        if not np.allclose(d_bf, d_tree, atol=1e-4):
            bad += 1

    status = "OK " if bad == 0 else "BAD"
    print(f"{status} {algo:<8} n={n_train:>7} d={d:>4} "
          f"wrong_neighbors={bad:>4}/{n_query}")
    return bad


def _sync(out):
    """JAX dispatches async; block to time real compute. KD-tree returns a
    NumPy array (no such method) — nothing to wait on there, so just skip."""
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    return out


def bench(n_train, n_query, d, k=5, n_classes=3, repeats=5, algo='KNN'):
    Xtr, ytr, Xte = make_data(n_train, n_query, d, n_classes)

    model = _make_model(algo, k).fit(Xtr, ytr)

    _sync(model.predict(Xte))                     # warm-up (JIT compile), untimed

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = model.predict(Xte)
        _sync(out)
        times.append(time.perf_counter() - t0)

    best = min(times)
    return best, n_query / best


if __name__ == "__main__":
    if _HAS_JAX:
        print(f"device: {jax.devices()[0]}\n")
    else:
        print("device: jax not installed — brute force uses NumPy fallback\n")

    print("-" * 48)
    print("correctness: trees find same neighbors as brute force")
    print("-" * 48)
    total_bad = 0
    for algo in ('KDTree', 'BallTree'):
        for n in (500, 2_000):
            total_bad += verify(n_train=n, n_query=200, d=20, algo=algo)
        for d in (2, 20, 100):
            total_bad += verify(n_train=2_000, n_query=200, d=d, algo=algo)
    if total_bad:
        print(f"\nERROR: {total_bad} queries with wrong neighbors — "
              f"tree pruning bug.\n")
    else:
        print("\nall tree neighbors identical to brute force\n")

    print(f"{'n_train':>8} {'n_query':>8} {'d':>4} {'best(ms)':>10} {'queries/s':>12}")
    print("-" * 48)
    print("KNN brute force")
    print("-" * 48)

    for n in (500, 2_000, 10_000):
        sec, qps = bench(n_train=n, n_query=200, d=20)
        print(f"{n:>8} {200:>8} {20:>4} {sec*1e3:>10.3f} {qps:>12,.0f}")

    print()
    for d in (2, 20, 100):
        sec, qps = bench(n_train=2_000, n_query=200, d=d)
        print(f"{2000:>8} {200:>8} {d:>4} {sec*1e3:>10.3f} {qps:>12,.0f}")

    print()
    print("-" * 48)
    print("KNN KD Tree")
    print("-" * 48)

    for n in (500, 2_000, 10_000):
        sec, qps = bench(n_train=n, n_query=200, d=20, algo='KDTree')
        print(f"{n:>8} {200:>8} {20:>4} {sec*1e3:>10.3f} {qps:>12,.0f}")

    print()
    for d in (2, 20, 100):
        sec, qps = bench(n_train=2_000, n_query=200, d=d, algo='KDTree')
        print(f"{2000:>8} {200:>8} {d:>4} {sec*1e3:>10.3f} {qps:>12,.0f}")

    print()
    print("-" * 48)
    print("KNN Ball Tree")
    print("-" * 48)

    for n in (500, 2_000, 10_000):
        sec, qps = bench(n_train=n, n_query=200, d=20, algo='BallTree')
        print(f"{n:>8} {200:>8} {20:>4} {sec*1e3:>10.3f} {qps:>12,.0f}")

    print()
    for d in (2, 20, 100):
        sec, qps = bench(n_train=2_000, n_query=200, d=d, algo='BallTree')
        print(f"{2000:>8} {200:>8} {d:>4} {sec*1e3:>10.3f} {qps:>12,.0f}")