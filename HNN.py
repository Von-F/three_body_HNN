#!/usr/bin/env python3
"""
JAX port of the Hamiltonian Neural Network for the Sun-Earth-Jupiter
Three-Body Problem.

Key differences from the PyTorch version:
  • jax.jit + jax.grad  →  compiled gradient (much faster than eager autograd)
  • optax Adam          →  JAX-native optimizer
  • Explicit param dicts (PyTree) instead of nn.Module
  • All data as numpy arrays, JAX ops via jax.numpy

Usage:  python hnn_jax.py   (requires: jax, optax, numpy, matplotlib)
"""

# %% [markdown]
# # 1. Imports & Device Setup

# %%
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

print(f"JAX  {jax.__version__}")
print(f"Devices: {jax.devices()}")

# %% [markdown]
# # 2. Physical Constants

# %%
G_si  = 6.673e-11
Me_si = 6e24
Ms_si = 2e30
Mj_si = 1.9e27

RR = 1.496e11
MM = 6e24
TT = 365 * 24 * 60 * 60.0

GG = (G_si * MM * TT ** 2) / (RR ** 3)

Me = Me_si / MM                   # = 1.0
Ms = Ms_si / MM                   # ≈ 3.33×10⁵
Mj = 500 * Mj_si / MM             # Super‑Jupiter

print(f"GG = {GG:.6e}   Me = {Me:.2f}   Ms = {Ms:.2f}   Mj = {Mj:.2f}")
print(f"Ms·GG = {Ms * GG:.4f}   ≈ 4π² = {4*np.pi**2:.4f}")

# %% [markdown]
# # 3. Data Generation (numpy — identical to PyTorch version)

# %%
def compute_derivatives_np(r_e, v_e, r_j, v_j):
    d_e  = np.linalg.norm(r_e,          axis=1, keepdims=True) + 1e-20
    d_j  = np.linalg.norm(r_j,          axis=1, keepdims=True) + 1e-20
    d_ej = np.linalg.norm(r_e - r_j,    axis=1, keepdims=True) + 1e-20

    F_se = -GG * Ms * Me * r_e         / (d_e  ** 3)
    F_sj = -GG * Ms * Mj * r_j         / (d_j  ** 3)
    r_ej = r_e - r_j
    F_ej = -GG * Me * Mj * r_ej        / (d_ej ** 3)

    F_earth = F_se + F_ej
    F_jup   = F_sj - F_ej

    dq_dt = np.hstack([v_e, v_j])
    dp_dt = np.hstack([F_earth, F_jup])
    return dq_dt, dp_dt


def rk4_step(r_e, v_e, r_j, v_j, h):
    def derivs(re, ve, rj, vj):
        dq, dp = compute_derivatives_np(
            re[np.newaxis, :], ve[np.newaxis, :],
            rj[np.newaxis, :], vj[np.newaxis, :])
        return dq[0], dp[0]

    k1_dq, k1_dp = derivs(r_e, v_e, r_j, v_j)
    k1_dq_e, k1_dq_j = k1_dq[:2], k1_dq[2:]
    k1_dp_e, k1_dp_j = k1_dp[:2], k1_dp[2:]

    k2_dq, k2_dp = derivs(r_e + 0.5*h*k1_dq_e, v_e + 0.5*h*k1_dp_e/Me,
                           r_j + 0.5*h*k1_dq_j, v_j + 0.5*h*k1_dp_j/Mj)
    k2_dq_e, k2_dq_j = k2_dq[:2], k2_dq[2:]
    k2_dp_e, k2_dp_j = k2_dp[:2], k2_dp[2:]

    k3_dq, k3_dp = derivs(r_e + 0.5*h*k2_dq_e, v_e + 0.5*h*k2_dp_e/Me,
                           r_j + 0.5*h*k2_dq_j, v_j + 0.5*h*k2_dp_j/Mj)
    k3_dq_e, k3_dq_j = k3_dq[:2], k3_dq[2:]
    k3_dp_e, k3_dp_j = k3_dp[:2], k3_dp[2:]

    k4_dq, k4_dp = derivs(r_e + h*k3_dq_e, v_e + h*k3_dp_e/Me,
                           r_j + h*k3_dq_j, v_j + h*k3_dp_j/Mj)
    k4_dq_e, k4_dq_j = k4_dq[:2], k4_dq[2:]
    k4_dp_e, k4_dp_j = k4_dp[:2], k4_dp[2:]

    r_e_new = r_e + h*(k1_dq_e + 2*k2_dq_e + 2*k3_dq_e + k4_dq_e)/6.0
    v_e_new = v_e + h*(k1_dp_e/Me + 2*k2_dp_e/Me + 2*k3_dp_e/Me + k4_dp_e/Me)/6.0
    r_j_new = r_j + h*(k1_dq_j + 2*k2_dq_j + 2*k3_dq_j + k4_dq_j)/6.0
    v_j_new = v_j + h*(k1_dp_j/Mj + 2*k2_dp_j/Mj + 2*k3_dp_j/Mj + k4_dp_j/Mj)/6.0

    return r_e_new, v_e_new, r_j_new, v_j_new


def generate_trajectory(years=200, pts_per_year=100,
                        r_e0=None, v_e0=None, r_j0=None, v_j0=None):
    N = int(years * pts_per_year) + 1
    t = np.linspace(0, years, N)
    h = t[1] - t[0]

    if r_e0 is None:  r_e0 = np.array([1.0, 0.0])
    if r_j0 is None:  r_j0 = np.array([5.2, 0.0])
    if v_e0 is None:
        vv = np.sqrt(Ms * GG / r_e0[0])
        v_e0 = np.array([0.0, vv])
    if v_j0 is None:
        vvj = 13.06e3 * TT / RR
        v_j0 = np.array([0.0, vvj])

    r_e = np.zeros((N, 2));  v_e = np.zeros((N, 2))
    r_j = np.zeros((N, 2));  v_j = np.zeros((N, 2))
    r_e[0] = r_e0;  v_e[0] = v_e0
    r_j[0] = r_j0;  v_j[0] = v_j0

    for i in range(N - 1):
        r_e[i+1], v_e[i+1], r_j[i+1], v_j[i+1] = \
            rk4_step(r_e[i], v_e[i], r_j[i], v_j[i], h)

    return {"t": t, "r_e": r_e, "v_e": v_e, "r_j": r_j, "v_j": v_j}


def generate_diverse_trajectories(n_traj, years, pts_per_year=400, seed=42):
    rng = np.random.RandomState(seed)
    trajectories = []
    for _ in range(n_traj):
        r_e_r = rng.uniform(0.25, 2.0)
        r_e_a = rng.uniform(0, 2 * np.pi)
        r_e0 = np.array([r_e_r * np.cos(r_e_a), r_e_r * np.sin(r_e_a)])

        v_circ_e = np.sqrt(Ms * GG / r_e_r)
        v_e_s = rng.uniform(0.6, 1.4)
        v_e0 = np.array([-v_circ_e * v_e_s * np.sin(r_e_a),
                          v_circ_e * v_e_s * np.cos(r_e_a)])

        r_j_r = rng.uniform(4.8, 5.6)
        r_j_a = rng.uniform(0, 2 * np.pi)
        r_j0 = np.array([r_j_r * np.cos(r_j_a), r_j_r * np.sin(r_j_a)])

        v_circ_j = np.sqrt(Ms * GG / r_j_r)
        v_j_s = rng.uniform(0.9, 1.1)
        v_j0 = np.array([-v_circ_j * v_j_s * np.sin(r_j_a),
                          v_circ_j * v_j_s * np.cos(r_j_a)])

        traj = generate_trajectory(years=years, pts_per_year=pts_per_year,
                                    r_e0=r_e0, v_e0=v_e0,
                                    r_j0=r_j0, v_j0=v_j0)
        trajectories.append(traj)
    return trajectories

# %% [markdown]
# # 4. Generate Training & Test Data

# %%
print("Generating 100 training trajectories (2 years each, 400 pts/yr) …")
tic = time.time()
traj_train = generate_diverse_trajectories(n_traj=100, years=2, pts_per_year=400,
                                            seed=42)
n_train_pts = sum(len(t["t"]) for t in traj_train)
print(f"  done in {time.time()-tic:.1f}s → {len(traj_train)} trajectories, "
      f"{n_train_pts} total points")

print("Generating test trajectory (400 pts/yr) …")
vv = np.sqrt(Ms * GG / 1.0)
data_test = generate_trajectory(
    years=30, pts_per_year=400,
    v_e0=np.array([0.0, vv * 1.02]),
)
print(f"  done → {len(data_test['t'])} time steps")

# %% [markdown]
# # 5. Build Dataset

# %%
def build_dataset(data):
    if isinstance(data, list):
        q_parts, p_parts, dq_parts, dp_parts = [], [], [], []
        for d in data:
            q, p, dq, dp = build_dataset(d)
            q_parts.append(q); p_parts.append(p)
            dq_parts.append(dq); dp_parts.append(dp)
        return (np.concatenate(q_parts), np.concatenate(p_parts),
                np.concatenate(dq_parts), np.concatenate(dp_parts))
    r_e = data["r_e"]; v_e = data["v_e"]
    r_j = data["r_j"]; v_j = data["v_j"]
    q = np.hstack([r_e, r_j])
    p = np.hstack([Me * v_e, Mj * v_j])
    dq_dt, dp_dt = compute_derivatives_np(r_e, v_e, r_j, v_j)
    return q, p, dq_dt, dp_dt


q_train, p_train, dq_dt_train, dp_dt_train = build_dataset(traj_train)
q_test,  p_test,  dq_dt_test,  dp_dt_test  = build_dataset(data_test)

# inv_d statistics
inv_d_e_tr  = 1.0 / (np.linalg.norm(q_train[:, :2], axis=1) + 1e-12)
inv_d_j_tr  = 1.0 / (np.linalg.norm(q_train[:, 2:], axis=1) + 1e-12)
inv_d_ej_tr = 1.0 / (np.linalg.norm(q_train[:, :2] - q_train[:, 2:], axis=1) + 1e-12)
inv_d_mean = np.array([inv_d_e_tr.mean(), inv_d_j_tr.mean(), inv_d_ej_tr.mean()])
inv_d_std  = np.array([inv_d_e_tr.std(),  inv_d_j_tr.std(),  inv_d_ej_tr.std()]) + 1e-10

# V_scale init
f_e_rms  = float(np.sqrt(np.mean(dp_dt_train[:, :2] ** 2)))
f_j_rms  = float(np.sqrt(np.mean(dp_dt_train[:, 2:] ** 2)))
f_ej_rms = float(np.sqrt(np.mean(
    (dp_dt_train[:, :2] + GG * Ms * Me * q_train[:, :2] /
     (np.linalg.norm(q_train[:, :2], axis=1, keepdims=True) ** 3 + 1e-20)) ** 2
)))
v_se_init = f_e_rms  * inv_d_std[0]
v_sj_init = f_j_rms  * inv_d_std[1]
v_ej_init = f_ej_rms * inv_d_std[2]

print(f"Train samples: {q_train.shape[0]},  Test samples: {q_test.shape[0]}")
print(f"inv_d_mean: {inv_d_mean},  inv_d_std: {inv_d_std}")
print(f"V init — se: {v_se_init:.2f},  sj: {v_sj_init:.1f},  ej: {v_ej_init:.3f}")

# %% [markdown]
# # 6. Model Definition (JAX — functional, no nn.Module)

# %%
def init_mlp(key, in_dim, hidden_dims):
    """
    MLP with linear skip connection (Patch 1).
    skip(x) = x·w  (pure linear, learns Newton's 1/r instantly).
    mlp(x) adds small nonlinear corrections (gain=0.01 init).
    """
    keys = jrandom.split(key, len(hidden_dims) + 2)
    mlp_layers = []
    prev = in_dim
    for i, h in enumerate(hidden_dims):
        w = jrandom.normal(keys[i], (prev, h)) * jnp.sqrt(2.0 / prev) * 0.01
        b = jnp.zeros(h)
        mlp_layers.append((w, b))
        prev = h
    # Output layer
    w = jrandom.normal(keys[len(hidden_dims)], (prev, 1)) * 0.01
    b = jnp.zeros(1)
    mlp_layers.append((w, b))
    # Skip weight = 1.0 (identity-like)
    skip_w = jnp.ones((in_dim, 1))
    return {'mlp': mlp_layers, 'skip_w': skip_w}


def mlp_forward(params, x):
    """params: {'mlp': [(w,b),...], 'skip_w': array}.  x: (1,) → y: (1,)"""
    y = x
    for w, b in params['mlp'][:-1]:
        y = jax.nn.softplus(y @ w + b)
    w_last, b_last = params['mlp'][-1]
    y = y @ w_last + b_last
    y = y + x @ params['skip_w']
    return y


def hnn_forward(params, q, p):
    """
    H(q, p) = T(p) + V_se(1/|q_e|) + V_sj(1/|q_j|) + V_ej(1/|q_e−q_j|)

    q, p: (4,) arrays (single sample).
    Returns scalar H.
    """
    q_e, q_j = q[:2], q[2:]
    p_e, p_j = p[:2], p[2:]

    # inv_d
    inv_d_e  = jax.lax.rsqrt(jnp.sum(q_e ** 2) + 1e-12)
    inv_d_j  = jax.lax.rsqrt(jnp.sum(q_j ** 2) + 1e-12)
    inv_d_ej = jax.lax.rsqrt(jnp.sum((q_e - q_j) ** 2) + 1e-12)

    # Normalise
    im = params['inv_d_mean'];  ist = params['inv_d_std']
    inv_d_e_n  = (inv_d_e  - im[0]) / ist[0]
    inv_d_j_n  = (inv_d_j  - im[1]) / ist[1]
    inv_d_ej_n = (inv_d_ej - im[2]) / ist[2]

    # Kinetic energy (exact)
    T = jnp.sum(p_e ** 2) / (2 * Me) + jnp.sum(p_j ** 2) / (2 * Mj)

    # Potential energy
    V_se = params['V_se_scale'] * mlp_forward(params['V_se'], inv_d_e_n.reshape(1))[0]
    V_sj = params['V_sj_scale'] * mlp_forward(params['V_sj'], inv_d_j_n.reshape(1))[0]
    V_ej = params['V_ej_scale'] * mlp_forward(params['V_ej'], inv_d_ej_n.reshape(1))[0]

    return T + V_se + V_sj + V_ej


# Vectorise over batch
hnn_batch = jax.vmap(hnn_forward, in_axes=(None, 0, 0))

# Gradient ∂H/∂q for single sample
_dH_dq_fn = jax.grad(hnn_forward, argnums=1)

# Batched gradient ∂H/∂q → (B, 4)
dH_dq_batch = jax.vmap(_dH_dq_fn, in_axes=(None, 0, 0))

# %% [markdown]
# # 7. Loss & Training

# %%
def loss_fn(params, q_batch, p_batch, dp_true_batch):
    """
    Relative error loss (Patch 2): each body contributes equally
    regardless of force magnitude (~40 vs ~232k).
    """
    dH_dq = dH_dq_batch(params, q_batch, p_batch)          # (B, 4)
    dp_pred = -dH_dq                                         # (B, 4)

    F_norm_e = jnp.sum(dp_true_batch[:, :2] ** 2, axis=1, keepdims=True) + 1e-4
    F_norm_j = jnp.sum(dp_true_batch[:, 2:] ** 2, axis=1, keepdims=True) + 1e-4

    sq_err_e = jnp.sum((dp_pred[:, :2] - dp_true_batch[:, :2]) ** 2,
                        axis=1, keepdims=True)
    sq_err_j = jnp.sum((dp_pred[:, 2:] - dp_true_batch[:, 2:]) ** 2,
                        axis=1, keepdims=True)

    loss = jnp.mean(sq_err_e / F_norm_e) + jnp.mean(sq_err_j / F_norm_j)
    return loss


# JIT-compiled loss + gradient
loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))


def init_params(key):
    """Initialise full HNN parameter PyTree."""
    keys = jrandom.split(key, 3)
    return {
        'V_se': init_mlp(keys[0], 1, (64, 64, 32)),
        'V_sj': init_mlp(keys[1], 1, (64, 64, 32)),
        'V_ej': init_mlp(keys[2], 1, (64, 64, 32)),
        'V_se_scale': jnp.array(v_se_init),
        'V_sj_scale': jnp.array(v_sj_init),
        'V_ej_scale': jnp.array(v_ej_init),
        'inv_d_mean': jnp.array(inv_d_mean),
        'inv_d_std':  jnp.array(inv_d_std),
    }


def train(params, q_train, p_train, dp_train,
          q_val, p_val, dp_val, epochs=800, lr=1e-3, patience=100):
    """Training loop with optax + early stopping."""
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def update(params, opt_state, q_b, p_b, dp_b):
        loss, grads = loss_and_grad(params, q_b, p_b, dp_b)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    best_val = float('inf')
    best_params = params
    no_improve = 0
    history = {'train': [], 'val': []}

    t0 = time.time()
    for ep in range(epochs):
        # Shuffle + minibatch
        perm = np.random.permutation(q_train.shape[0])
        ep_loss = 0.0; n_batch = 0
        for start in range(0, q_train.shape[0], 512):
            idx = perm[start:start + 512]
            params, opt_state, loss = update(
                params, opt_state,
                jnp.array(q_train[idx]), jnp.array(p_train[idx]),
                jnp.array(dp_train[idx]))
            ep_loss += float(loss)
            n_batch += 1

        # Validation
        val_loss = float(loss_fn(params,
                                  jnp.array(q_val), jnp.array(p_val),
                                  jnp.array(dp_val)))
        history['train'].append(ep_loss / n_batch)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_params = jax.tree.map(lambda x: x.copy(), params)
            no_improve = 0
        else:
            no_improve += 1

        if (ep + 1) % 80 == 0:
            print(f"  epoch {ep+1:4d} | train {ep_loss/n_batch:.2e} | "
                  f"val {val_loss:.2e} | lr {lr:.2e}")

        if no_improve >= patience:
            print(f"  early stopping at epoch {ep+1}")
            break

    print(f"Training done in {(time.time()-t0)/60:.1f} min | "
          f"best val = {best_val:.2e}")
    return best_params, history

# %% [markdown]
# # 8. Train

# %%
key = jrandom.PRNGKey(42)
n_total = q_train.shape[0]
n_val = int(n_total * 0.1)
idx = np.random.permutation(n_total)
idx_tr, idx_val = idx[:n_total - n_val], idx[n_total - n_val:]

params = init_params(key)
n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"HNN parameters: {n_params:,}")

params, history = train(
    params,
    q_train[idx_tr], p_train[idx_tr], dp_dt_train[idx_tr],
    q_train[idx_val], p_train[idx_val], dp_dt_train[idx_val],
    epochs=800, lr=1e-3, patience=120)

# Test loss
test_loss = float(loss_fn(params,
                           jnp.array(q_test), jnp.array(p_test),
                           jnp.array(dp_dt_test)))
print(f"Test loss (relative): {test_loss:.4e}")

# %% [markdown]
# # 9. Rollout — JIT‑compiled force, in‑JAX integration

# %%
# JIT‑compile the force — single call to jitted function per evaluation
_force_jit = jax.jit(lambda params, q, p: -_dH_dq_fn(params, q, p))


def _integrator_loop(params, q0, p0, t_span, steps_per_unit, step_fn):
    """
    Generic integration loop: JAX for force, numpy for storage.
    Avoids JAX immutable-array O(N²) copy overhead.
    """
    h = 1.0 / steps_per_unit
    N = int(t_span * steps_per_unit) + 1
    t = np.linspace(0, t_span, N)
    q = np.zeros((N, 4)); p = np.zeros((N, 4))
    q[0] = q0.copy(); p[0] = p0.copy()

    for i in range(N - 1):
        qi_j = jnp.array(q[i]); pi_j = jnp.array(p[i])
        q_next, p_next = step_fn(params, qi_j, pi_j, h, _force_jit, Me, Mj)
        q[i+1] = np.array(q_next); p[i+1] = np.array(p_next)
    return t, q, p


def _verlet_step(params, qi, pi, h, force_fn, Me, Mj):
    dp = force_fn(params, qi, pi)
    p_half = pi + 0.5 * h * dp
    q_next = qi.at[:2].add(h * p_half[:2] / Me)
    q_next = q_next.at[2:].add(h * p_half[2:] / Mj)
    dp_next = force_fn(params, q_next, p_half)
    p_next = p_half + 0.5 * h * dp_next
    return q_next, p_next


def _yoshida_step(params, qi, pi, h, force_fn, Me, Mj):
    w0 = w2 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w1 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))
    for w in (w0, w1, w2):
        hw = w * h
        dp = force_fn(params, qi, pi)
        p_half = pi + 0.5 * hw * dp
        qi = qi.at[:2].add(hw * p_half[:2] / Me)
        qi = qi.at[2:].add(hw * p_half[2:] / Mj)
        dp_next = force_fn(params, qi, p_half)
        pi = p_half + 0.5 * hw * dp_next
    return qi, pi


def _rk4_step(params, qi, pi, h, force_fn, Me, Mj):
    k1_q = jnp.concatenate([pi[:2] / Me, pi[2:] / Mj])
    k1_p = force_fn(params, qi, pi)
    q2 = qi + 0.5 * h * k1_q; p2 = pi + 0.5 * h * k1_p
    k2_q = jnp.concatenate([p2[:2] / Me, p2[2:] / Mj])
    k2_p = force_fn(params, q2, p2)
    q3 = qi + 0.5 * h * k2_q; p3 = pi + 0.5 * h * k2_p
    k3_q = jnp.concatenate([p3[:2] / Me, p3[2:] / Mj])
    k3_p = force_fn(params, q3, p3)
    q4 = qi + h * k3_q; p4 = pi + h * k3_p
    k4_q = jnp.concatenate([p4[:2] / Me, p4[2:] / Mj])
    k4_p = force_fn(params, q4, p4)
    q_next = qi + h * (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6.0
    p_next = pi + h * (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6.0
    return q_next, p_next


def verlet_rollout(params, q0, p0, t_span, steps_per_unit=400):
    return _integrator_loop(params, q0, p0, t_span, steps_per_unit, _verlet_step)


def yoshida_rollout(params, q0, p0, t_span, steps_per_unit=400):
    return _integrator_loop(params, q0, p0, t_span, steps_per_unit, _yoshida_step)


def rk4_rollout(params, q0, p0, t_span, steps_per_unit=400):
    return _integrator_loop(params, q0, p0, t_span, steps_per_unit, _rk4_step)


def true_rollout(data, start_idx, length, steps_per_unit=100):
    pts_per_year = int(len(data["t"]) / (data["t"][-1] - data["t"][0]))
    step = max(1, pts_per_year // steps_per_unit)
    end_idx = min(start_idx + length * pts_per_year + 1, len(data["t"]))
    indices = np.arange(start_idx, end_idx, step)
    r_e = data["r_e"][indices]; r_j = data["r_j"][indices]
    v_e = data["v_e"][indices]; v_j = data["v_j"][indices]
    q = np.hstack([r_e, r_j])
    p = np.hstack([Me * v_e, Mj * v_j])
    t = data["t"][indices]
    return t, q, p


print("\nRunning HNN rollout — Verlet / Yoshida / RK4 …")
r_e0 = data_test["r_e"][0]; r_j0 = data_test["r_j"][0]
q0 = np.concatenate([r_e0, r_j0])
p0 = np.concatenate([Me * data_test["v_e"][0], Mj * data_test["v_j"][0]])

SPU = 400
tic = time.time()
t_v2, q_v2, p_v2 = verlet_rollout(params, q0, p0, 20, steps_per_unit=SPU)
dt_v2 = time.time() - tic
print(f"  Verlet (2nd, sympl):  {dt_v2:.1f}s")

tic = time.time()
t_y4, q_y4, p_y4 = yoshida_rollout(params, q0, p0, 20, steps_per_unit=SPU)
dt_y4 = time.time() - tic
print(f"  Yoshida (4th, sympl): {dt_y4:.1f}s")

tic = time.time()
t_rk4, q_rk4, p_rk4 = rk4_rollout(params, q0, p0, 20, steps_per_unit=SPU)
dt_rk4 = time.time() - tic
print(f"  RK4 (4th):            {dt_rk4:.1f}s")

t_true, q_true, p_true = true_rollout(data_test, 0, 20, steps_per_unit=SPU)

# %% [markdown]
# # 10. Energy & Error Evaluation

# %%
def physical_energy(q, p):
    r_e = q[:, :2]; r_j = q[:, 2:]
    p_e = p[:, :2]; p_j = p[:, 2:]
    KE = (p_e**2).sum(axis=1)/(2*Me) + (p_j**2).sum(axis=1)/(2*Mj)
    d_e  = np.linalg.norm(r_e, axis=1) + 1e-20
    d_j  = np.linalg.norm(r_j, axis=1) + 1e-20
    d_ej = np.linalg.norm(r_e - r_j, axis=1) + 1e-20
    PE = -GG*Ms*Me/d_e - GG*Ms*Mj/d_j - GG*Me*Mj/d_ej
    return KE + PE


E_true = physical_energy(q_true, p_true)
E_v2 = physical_energy(q_v2, p_v2)
E_y4 = physical_energy(q_y4, p_y4)
E_rk4 = physical_energy(q_rk4, p_rk4)

err_v2  = np.linalg.norm(q_v2  - q_true, axis=1)
err_y4  = np.linalg.norm(q_y4  - q_true, axis=1)
err_rk4 = np.linalg.norm(q_rk4 - q_true, axis=1)

print(f"\n{'':>22} {'True':>12} {'Verlet(2)':>12} {'Yoshida(4)':>12} {'RK4':>12}")
print(f"{'E phys mean':>22} {E_true.mean():12.2f} "
      f"{E_v2.mean():12.2f} {E_y4.mean():12.2f} {E_rk4.mean():12.2f}")
print(f"{'E phys std':>22} {E_true.std():12.4f} "
      f"{E_v2.std():12.4f} {E_y4.std():12.4f} {E_rk4.std():12.4f}")
print(f"{'|q_err| mean':>22} {'—':>12} "
      f"{err_v2.mean():12.4f} {err_y4.mean():12.4f} {err_rk4.mean():12.4f}")
print(f"{'|q_err| final':>22} {'—':>12} "
      f"{err_v2[-1]:12.4f} {err_y4[-1]:12.4f} {err_rk4[-1]:12.4f}")

# Default to Yoshida for visualisation
t_hnn, q_hnn, p_hnn = t_y4, q_y4, p_y4

# %% [markdown]
# # 11. Visualisation

# %%
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.35)

for (ax, q, title) in [
    (fig.add_subplot(gs[0, 0]), q_true, "Ground Truth (RK4)"),
    (fig.add_subplot(gs[0, 1]), q_hnn,  "HNN (Yoshida 4th)"),
]:
    ax.plot(0, 0, "o", markersize=8, color="#FDB813", markeredgecolor="#FD7813",
            label="Sun")
    ax.plot(q[:, 0], q[:, 1], lw=0.8, color="#0077BE", label="Earth")
    ax.plot(q[:, 2], q[:, 3], lw=0.8, color="#f66338", label="Jupiter")
    ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
    ax.set_title(title); ax.axis("equal")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax_en = fig.add_subplot(gs[1, :])
ax_en.plot(t_true, E_true, lw=1.5, color="black", label="E(q,p) on true traj")
ax_en.plot(t_hnn,  E_y4,   lw=1.5, color="red", alpha=0.7,
           label="E(q,p) on HNN traj")
ax_en.set_xlabel("Time (years)"); ax_en.set_ylabel("Physical Energy")
ax_en.set_title("Physical Energy Conservation")
ax_en.legend(); ax_en.grid(True, alpha=0.3)

ax_tr = fig.add_subplot(gs[2, 0])
ax_tr.semilogy(history["train"], lw=1.0, alpha=0.7, label="train")
ax_tr.semilogy(history["val"],   lw=1.5, label="validation")
ax_tr.set_xlabel("epoch"); ax_tr.set_ylabel("Relative loss")
ax_tr.set_title("Training Loss"); ax_tr.legend(); ax_tr.grid(True, alpha=0.3)

ax_err = fig.add_subplot(gs[2, 1])
min_len = min(len(t_true), len(t_hnn))
err = np.linalg.norm(q_hnn[:min_len] - q_true[:min_len], axis=1)
ax_err.semilogy(t_hnn[:min_len], err, lw=1.0, color="red")
ax_err.set_xlabel("Time (years)")
ax_err.set_ylabel(r"$\|q_{\rm HNN} - q_{\rm true}\|$ (AU)")
ax_err.set_title("Position Error Growth")
ax_err.grid(True, alpha=0.3)

fig.suptitle("JAX HNN:  H(q,p) = T(p) + V(1/d)  (skip-connected)", fontsize=13, y=0.98)
plt.show()

print(f"\nMean  |q_err|: {err.mean():.4f} AU")
print(f"Final |q_err|: {err[-1]:.4f} AU")
print(f"Max   |q_err|: {err.max():.4f} AU")
