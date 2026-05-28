"""Skip-gram with negative sampling (SGNS).

Trains two embedding matrices V and U by binary classification on word pairs:
  positive pairs (w, c) drawn from a sliding window over the corpus,
  negative pairs (w, c_i) drawn from the unigram^0.75 noise distribution.

Loss per pair (maximize):
    L = log sigma(u_c · v_w) + sum_i log sigma(-u_{c_i} · v_w)
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial


# -------------------------------------------------------------------------
# CORPUS PREP — text -> integer ids -> (center, positive) pairs
# -------------------------------------------------------------------------
def build_vocab(sentences, min_count=1):
    """Map each unique word to an integer id. Returns (word2id, id2word, freq)."""
    from collections import Counter
    counts = Counter(w for s in sentences for w in s)
    words = [w for w, c in counts.most_common() if c >= min_count]
    word2id = {w: i for i, w in enumerate(words)}
    id2word = words
    freq = np.array([counts[w] for w in words], dtype=np.float64)
    return word2id, id2word, freq


def make_pairs(sentences, word2id, window=2):
    """Slide a window; emit (center, context) index pairs."""
    pairs = []
    for s in sentences:
        ids = [word2id[w] for w in s if w in word2id]
        for i, c in enumerate(ids):
            lo, hi = max(0, i - window), min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((c, ids[j]))
    return np.array(pairs, dtype=np.int32)            # (N_pairs, 2)


def noise_distribution(freq, power=0.75):
    """Unigram^0.75 distribution for sampling negatives."""
    p = freq ** power
    return p / p.sum()


# -------------------------------------------------------------------------
# MODEL — two embedding matrices, score via dot product
# -------------------------------------------------------------------------
def init_params(key, V_size, d):
    """Initialize input (V) and output (U) embedding matrices."""
    k1, k2 = jax.random.split(key)
    # Small random init keeps initial dot products near 0 (sigma ~ 0.5).
    V = jax.random.normal(k1, (V_size, d)) * 0.5 / d ** 0.5
    U = jax.random.normal(k2, (V_size, d)) * 0.5 / d ** 0.5
    return {"V": V, "U": U}


def loss_fn(params, centers, positives, negatives):
    """Negative log-likelihood for a batch of (center, positive, negatives).

    Shapes:
      centers:   (B,)        center word ids
      positives: (B,)        true context ids
      negatives: (B, K)      sampled negative ids
    """
    V, U = params["V"], params["U"]
    v_w = V[centers]                                   # (B, d)
    u_pos = U[positives]                               # (B, d)
    u_neg = U[negatives]                               # (B, K, d)

    # Positive scores: u_pos . v_w   -> (B,)
    pos_score = jnp.sum(u_pos * v_w, axis=-1)

    # Negative scores: u_neg . v_w   -> (B, K). einsum is the cleanest way to
    # contract the last (d) axis of u_neg with v_w.
    neg_score = jnp.einsum("bkd,bd->bk", u_neg, v_w)

    # log sigma is more stable than log(sigmoid(...)); use the softplus form:
    # log sigma(x) = -softplus(-x).  Negatives flip sign: log sigma(-x).
    log_sig_pos = -jax.nn.softplus(-pos_score)         # (B,)
    log_sig_neg = -jax.nn.softplus(neg_score)          # (B, K)  note: +neg_score

    # We MAXIMIZE the per-pair likelihood, so MINIMIZE its negative.
    per_pair = log_sig_pos + log_sig_neg.sum(axis=-1)  # (B,)
    return -per_pair.mean()


# -------------------------------------------------------------------------
# TRAINING LOOP — sample a batch, take a gradient step
# -------------------------------------------------------------------------
def train(pairs, freq, V_size, d=50, K=5, lr=0.025,
          steps=5000, batch=512, seed=0):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    params = init_params(key, V_size, d)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    noise_p = noise_distribution(freq)

    @jax.jit
    def step(params, opt_state, centers, positives, negatives):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, centers, positives, negatives)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    N = len(pairs)
    for s in range(steps):
        # Sample a minibatch of (center, positive) pairs uniformly with
        # replacement — simplest unbiased estimator of the corpus.
        idx = rng.integers(0, N, size=batch)
        centers = pairs[idx, 0]
        positives = pairs[idx, 1]
        # Negatives: K per row, drawn from unigram^0.75. NB this can
        # occasionally draw the positive itself as a "negative"; with V large
        # the bias is negligible. For a strict version, rejection-resample.
        negatives = rng.choice(V_size, size=(batch, K), p=noise_p)

        params, opt_state, loss = step(
            params, opt_state,
            jnp.asarray(centers), jnp.asarray(positives), jnp.asarray(negatives))

        if s % 500 == 0 or s == steps - 1:
            print(f"step {s:>5}  loss {float(loss):.4f}")

    return params


# -------------------------------------------------------------------------
# DEMO — tiny corpus showing the model learns semantic clusters
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Two-cluster toy corpus: royalty/people and animals/actions.
    sentences = [
        "king queen man woman".split(),
        "man king woman queen".split(),
        "king man queen woman king".split(),
        "queen king woman man queen".split(),
        "cat dog run walk".split(),
        "dog cat walk run dog".split(),
        "cat run dog walk cat".split(),
        "walk dog run cat walk".split(),
    ] * 30   # repeat so we get enough pairs

    word2id, id2word, freq = build_vocab(sentences)
    V_size = len(id2word)
    print(f"vocab ({V_size}):", id2word)

    pairs = make_pairs(sentences, word2id, window=2)
    print(f"training pairs: {len(pairs)}")

    params = train(pairs, freq, V_size,
                   d=8, K=5, lr=0.05, steps=3000, batch=256)

    # Inspect what was learned: cosine similarity between input embeddings.
    V = np.asarray(params["V"])
    V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    sim = V_norm @ V_norm.T                            # (V_size, V_size)

    print("\nnearest neighbors by cosine similarity:")
    for i, w in enumerate(id2word):
        order = np.argsort(-sim[i])
        neighbors = [(id2word[j], round(float(sim[i, j]), 2))
                     for j in order if j != i][:3]
        print(f"  {w:>6}: {neighbors}")
