import jax
import jax.numpy as jnp
from functools import partial

class KNN:
    def __init__(self, k = 5):
        self.k = k

    def fit(self, X, y):
        self.X = jnp.asarray(X, dtype=jnp.float32) #(n, d)
        self.y = jnp.asarray(y) #(n)

        # Number of classes; needed for a static bincount width under JIT.
        self.n_classes = int(self.y.max()) + 1
        return self
    
    @partial(jax.jit, static_argnums=0)
    def _kneighbors(self, q):
        # Squared Euclidean is enough for ranking; skip the sqrt.
        diff = self.X - q # (n, d)
        d2 = jnp.einsum('ij,ij->i', diff, diff) # (n,)
        # top_k on the negative gives the k smallest; returns sorted order.
        neg_d, idx = jax.lax.top_k(-d2, self.k)
        return jnp.sqrt(-neg_d), idx # (k,), (k,)
    
    def kneighbors(self, q):
        return self._kneighbors(jnp.asarray(q, dtype=jnp.float32))
    
    @partial(jax.jit, static_argnums=0)
    def _predict_one(self, q):
        _, idx = self._kneighbors(q)
        votes = self.y[idx]
        counts = jnp.bincount(votes, length=self.n_classes)
        return jnp.argmax(counts) # majority label
    
    @partial(jax.jit, static_argnums=0)
    def _predict_batch(self, Q):
        # vmap turns the single-query function into a batched one.
        # Memory note: vmap stacks the per-query (n_train, d) diff into
        # (n_query, n_train, d). Fine at these sizes, but for large n_train
        # x n_query this is the ceiling — chunk Q with jax.lax.map if it bites.
        return jax.vmap(self._predict_one)(Q)

    def predict(self, Q):
        return self._predict_batch(jnp.asarray(Q, dtype=jnp.float32))