"""
Natural Gradient Descent vs Vanilla Gradient Descent
====================================================

Model: Gaussian N(mu, sigma^2) parameterized as theta = (mu, rho)
where rho = log(sigma). This non-trivial parameterization makes the
FIM strongly anisotropic and theta-dependent — perfect for showing
where natural gradient shines.

Loss: Negative log-likelihood on observed data x_1,...,x_n drawn
from a target distribution.

We compute:
- analytic gradient
- analytic FIM
and run both VGD and NGD from the same starting point.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import os

# ----------------------------------------------------------------------
# Model: N(mu, sigma^2) with theta = (mu, rho), rho = log(sigma)
# ----------------------------------------------------------------------

def log_p(x, theta):
    """log p(x | theta) for theta = (mu, rho)."""
    mu, rho = theta
    sigma = np.exp(rho)
    return -0.5 * np.log(2 * np.pi) - rho - 0.5 * ((x - mu) / sigma) ** 2


def score(x, theta):
    """Score s(x, theta) = grad_theta log p(x | theta).

    Returns shape (2,) for scalar x, or (n, 2) for x of shape (n,).
    """
    mu, rho = theta
    sigma = np.exp(rho)
    # d log p / d mu = (x - mu) / sigma^2
    # d log p / d rho: log p = -rho - 0.5 (x-mu)^2 exp(-2 rho) + const
    #               = -1 + (x - mu)^2 / sigma^2
    d_mu = (x - mu) / sigma ** 2
    d_rho = -1.0 + ((x - mu) / sigma) ** 2
    if np.isscalar(x):
        return np.array([d_mu, d_rho])
    return np.stack([d_mu, d_rho], axis=-1)


def loss(theta, data):
    """Negative log-likelihood averaged over data."""
    return -np.mean(log_p(data, theta))


def grad_loss(theta, data):
    """Gradient of the average negative log-likelihood."""
    # grad of -log p averaged over data = - mean of score
    return -np.mean(score(data, theta), axis=0)


def empirical_fim(theta, data):
    """Estimate FIM from data using the score-covariance form.

    F = E[s s^T] where the expectation is over x sampled from p_theta.
    Here we approximate by averaging over the observed data, which is
    standard (this is the 'empirical' FIM).
    """
    s = score(data, theta)        # (n, 2)
    return (s.T @ s) / len(data)  # (2, 2)


def analytic_fim(theta):
    """Closed-form FIM for N(mu, sigma^2) with theta = (mu, rho).

    For mu: F_mu = 1/sigma^2
    For rho: F_rho = 2 (using log-scale)
    Off-diagonal: 0 (mu and rho are orthogonal parameters)
    """
    _, rho = theta
    sigma = np.exp(rho)
    return np.array([[1.0 / sigma ** 2, 0.0],
                     [0.0, 2.0]])


# ----------------------------------------------------------------------
# Optimization
# ----------------------------------------------------------------------

def run_vgd(theta0, data, lr, n_steps):
    """Vanilla gradient descent."""
    theta = np.array(theta0, dtype=float)
    history = [theta.copy()]
    for _ in range(n_steps):
        g = grad_loss(theta, data)
        theta = theta - lr * g
        history.append(theta.copy())
    return np.array(history)


def run_ngd(theta0, data, lr, n_steps, damping=1e-4, use_analytic_fim=True):
    """Natural gradient descent."""
    theta = np.array(theta0, dtype=float)
    history = [theta.copy()]
    for _ in range(n_steps):
        g = grad_loss(theta, data)
        if use_analytic_fim:
            F = analytic_fim(theta)
        else:
            F = empirical_fim(theta, data)
        F_reg = F + damping * np.eye(2)            # numerical stability
        d = np.linalg.solve(F_reg, g)
        theta = theta - lr * d
        history.append(theta.copy())
    return np.array(history)


# ----------------------------------------------------------------------
# Setup the problem
# ----------------------------------------------------------------------

np.random.seed(42)
TRUE_MU = 2.0
TRUE_SIGMA = 0.5
TRUE_RHO = np.log(TRUE_SIGMA)
N_DATA = 200
data = np.random.normal(TRUE_MU, TRUE_SIGMA, size=N_DATA)

theta0 = np.array([-1.0, 1.5])    # start far away, with wrong scale (sigma ~ 4.5)
n_steps = 80

# Pick learning rates that give each method its best fair chance.
# Vanilla GD is sensitive to ill-conditioning; we use a small LR.
# NGD can handle a larger LR thanks to the preconditioner.
lr_vgd = 0.04
lr_ngd = 0.5

vgd_hist = run_vgd(theta0, data, lr=lr_vgd, n_steps=n_steps)
ngd_hist = run_ngd(theta0, data, lr=lr_ngd, n_steps=n_steps)

# ----------------------------------------------------------------------
# Build the loss surface for plotting
# ----------------------------------------------------------------------

mu_grid = np.linspace(-1.5, 3.0, 80)
rho_grid = np.linspace(-1.5, 2.0, 80)
MU, RHO = np.meshgrid(mu_grid, rho_grid)
LOSS = np.zeros_like(MU)
for i in range(MU.shape[0]):
    for j in range(MU.shape[1]):
        LOSS[i, j] = loss((MU[i, j], RHO[i, j]), data)

# Compute gradient field on a coarser grid
mu_q = np.linspace(-1.3, 2.8, 14)
rho_q = np.linspace(-1.3, 1.8, 12)
MUq, RHOq = np.meshgrid(mu_q, rho_q)
VGD_U = np.zeros_like(MUq)
VGD_V = np.zeros_like(MUq)
NGD_U = np.zeros_like(MUq)
NGD_V = np.zeros_like(MUq)

for i in range(MUq.shape[0]):
    for j in range(MUq.shape[1]):
        t = np.array([MUq[i, j], RHOq[i, j]])
        g = grad_loss(t, data)
        F = analytic_fim(t)
        Finv_g = np.linalg.solve(F + 1e-4 * np.eye(2), g)
        # Negate to show descent direction
        VGD_U[i, j] = -g[0]
        VGD_V[i, j] = -g[1]
        NGD_U[i, j] = -Finv_g[0]
        NGD_V[i, j] = -Finv_g[1]

# Normalize for cleaner arrow display (preserve direction, equal length)
def normalize_field(U, V):
    mag = np.sqrt(U ** 2 + V ** 2)
    mag[mag == 0] = 1
    return U / mag, V / mag

VGD_Un, VGD_Vn = normalize_field(VGD_U, VGD_V)
NGD_Un, NGD_Vn = normalize_field(NGD_U, NGD_V)

# ----------------------------------------------------------------------
# Plot 1: trajectories on loss surface
# ----------------------------------------------------------------------

os.makedirs('outputs', exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(9, 7))
levels = np.linspace(LOSS.min(), np.percentile(LOSS, 92), 25)
cs = ax.contourf(MU, RHO, LOSS, levels=levels, cmap='Blues_r', alpha=0.8)
ax.contour(MU, RHO, LOSS, levels=levels, colors='white', linewidths=0.4, alpha=0.5)
plt.colorbar(cs, ax=ax, label='Negative log-likelihood', shrink=0.85)

# True optimum
ax.plot(TRUE_MU, TRUE_RHO, marker='*', color='gold',
        markersize=22, markeredgecolor='black', markeredgewidth=1.0,
        zorder=10, label='True optimum')

# Start point
ax.plot(theta0[0], theta0[1], marker='o', color='black',
        markersize=10, zorder=10, label='Start')

# VGD trajectory
ax.plot(vgd_hist[:, 0], vgd_hist[:, 1], '-o', color='#185FA5',
        markersize=3, linewidth=1.5, label=f'Vanilla GD (lr={lr_vgd})', zorder=8)

# NGD trajectory
ax.plot(ngd_hist[:, 0], ngd_hist[:, 1], '-o', color='#D85A30',
        markersize=3, linewidth=1.5, label=f'Natural GD (lr={lr_ngd})', zorder=9)

ax.set_xlabel(r'$\mu$', fontsize=13)
ax.set_ylabel(r'$\rho = \log\sigma$', fontsize=13)
ax.set_title('Optimization trajectories on the NLL surface\n'
             r'Model: $\mathcal{N}(\mu, \sigma^2)$, parameterization $\theta=(\mu, \log\sigma)$',
             fontsize=12)
ax.legend(loc='upper right', framealpha=0.95)
plt.tight_layout()
plt.savefig('outputs/trajectories.png', dpi=130, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# Plot 2: gradient fields side by side
# ----------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, U, V, name, color in [
    (axes[0], VGD_Un, VGD_Vn, 'Vanilla GD: $-\\nabla L$', '#185FA5'),
    (axes[1], NGD_Un, NGD_Vn, 'Natural GD: $-F^{-1}\\nabla L$', '#D85A30'),
]:
    cs = ax.contourf(MU, RHO, LOSS, levels=levels, cmap='Blues_r', alpha=0.7)
    ax.contour(MU, RHO, LOSS, levels=levels, colors='white', linewidths=0.4, alpha=0.5)
    ax.quiver(MUq, RHOq, U, V, color=color, scale=22, width=0.0045,
              headwidth=4.5, alpha=0.95)
    ax.plot(TRUE_MU, TRUE_RHO, marker='*', color='gold',
            markersize=20, markeredgecolor='black', markeredgewidth=1.0)
    ax.plot(theta0[0], theta0[1], marker='o', color='black', markersize=9)
    ax.set_xlabel(r'$\mu$', fontsize=13)
    ax.set_ylabel(r'$\rho = \log\sigma$', fontsize=13)
    ax.set_title(name, fontsize=12)

plt.suptitle('Descent direction fields: vanilla vs natural gradient',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('outputs/gradient_fields.png', dpi=130, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# Plot 3: convergence of loss & KL to true distribution
# ----------------------------------------------------------------------

def kl_gaussian(theta, true_mu=TRUE_MU, true_sigma=TRUE_SIGMA):
    """KL(N(true_mu, true_sigma^2) || N(mu, sigma^2))."""
    mu, rho = theta
    sigma = np.exp(rho)
    return (np.log(sigma / true_sigma)
            + (true_sigma ** 2 + (true_mu - mu) ** 2) / (2 * sigma ** 2)
            - 0.5)

vgd_losses = [loss(t, data) for t in vgd_hist]
ngd_losses = [loss(t, data) for t in ngd_hist]
vgd_kls = [kl_gaussian(t) for t in vgd_hist]
ngd_kls = [kl_gaussian(t) for t in ngd_hist]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(vgd_losses, color='#185FA5', label='Vanilla GD', linewidth=1.8)
axes[0].plot(ngd_losses, color='#D85A30', label='Natural GD', linewidth=1.8)
axes[0].set_xlabel('Iteration', fontsize=12)
axes[0].set_ylabel('NLL', fontsize=12)
axes[0].set_title('Loss vs iteration', fontsize=12)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].semilogy(vgd_kls, color='#185FA5', label='Vanilla GD', linewidth=1.8)
axes[1].semilogy(ngd_kls, color='#D85A30', label='Natural GD', linewidth=1.8)
axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel(r'$D_{KL}(p_{true} \| p_\theta)$', fontsize=12)
axes[1].set_title('KL to true distribution (log scale)', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('outputs/convergence.png', dpi=130, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# Plot 4: how the FIM (metric tensor) varies across parameter space
# ----------------------------------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(9, 7))
cs = ax.contourf(MU, RHO, LOSS, levels=levels, cmap='Blues_r', alpha=0.45)

# Sample a grid and draw FIM ellipses at each point.
# Each ellipse shows the set of delta with delta^T F delta = c (constant KL level).
# Where the ellipse is LARGE: parameter changes don't move the distribution much
# (low information). Where SMALL: parameter changes move distribution a lot.
mu_e = np.linspace(-0.8, 2.6, 5)
rho_e = np.linspace(-1.0, 1.6, 6)
c = 0.05  # constant KL level — small enough for clean non-overlapping ellipses

for me in mu_e:
    for re in rho_e:
        F = analytic_fim((me, re))
        eigvals, eigvecs = np.linalg.eigh(F)
        rx = np.sqrt(c / eigvals[0])
        ry = np.sqrt(c / eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse((me, re), 2 * rx, 2 * ry, angle=angle,
                      facecolor='#D85A30', edgecolor='#712B13',
                      linewidth=1.0, alpha=0.45)
        ax.add_patch(ell)
        # Add a dot at the center for clarity
        ax.plot(me, re, '.', color='#712B13', markersize=2.5)

# Restrict to original loss-surface range to avoid showing ellipses outside
ax.set_xlim(mu_grid.min(), mu_grid.max())
ax.set_ylim(rho_grid.min(), rho_grid.max())

ax.plot(TRUE_MU, TRUE_RHO, marker='*', color='gold',
        markersize=20, markeredgecolor='black', markeredgewidth=1.0,
        zorder=10, label='True optimum')

ax.set_xlabel(r'$\mu$', fontsize=13)
ax.set_ylabel(r'$\rho = \log\sigma$', fontsize=13)
ax.set_title('FIM ellipses: each ellipse = set of $\\delta$ with $\\delta^\\top F \\delta = $ const\n'
             '(Larger ellipse $\\Rightarrow$ larger $\\delta$ allowed for same KL $\\Rightarrow$ less sensitive direction)',
             fontsize=11)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('outputs/fim_ellipses.png', dpi=130, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# Print summary
# ----------------------------------------------------------------------

print("=" * 60)
print("Final results")
print("=" * 60)
print(f"True parameters:     mu = {TRUE_MU:.3f}, rho = {TRUE_RHO:.3f} (sigma = {TRUE_SIGMA:.3f})")
print(f"Starting point:      mu = {theta0[0]:.3f}, rho = {theta0[1]:.3f}")
print()
print(f"VGD after {n_steps} steps:")
print(f"  theta = ({vgd_hist[-1, 0]:.3f}, {vgd_hist[-1, 1]:.3f})")
print(f"  NLL  = {vgd_losses[-1]:.4f}")
print(f"  KL   = {vgd_kls[-1]:.6f}")
print()
print(f"NGD after {n_steps} steps:")
print(f"  theta = ({ngd_hist[-1, 0]:.3f}, {ngd_hist[-1, 1]:.3f})")
print(f"  NLL  = {ngd_losses[-1]:.4f}")
print(f"  KL   = {ngd_kls[-1]:.6f}")
print()
print(f"NGD reached KL < 0.01 at iteration: ", end="")
ngd_arr = np.array(ngd_kls)
idx = np.where(ngd_arr < 0.01)[0]
print(idx[0] if len(idx) else "never")
print(f"VGD reached KL < 0.01 at iteration: ", end="")
vgd_arr = np.array(vgd_kls)
idx = np.where(vgd_arr < 0.01)[0]
print(idx[0] if len(idx) else "never")


