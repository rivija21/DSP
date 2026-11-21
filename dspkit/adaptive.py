"""
dspkit.adaptive -- adaptive FIR filters (LMS family and RLS).

Common structure (transversal filter, M taps):
    regressor   u[n] = [x[n], x[n-1], ..., x[n-M+1]]^T
    output      y[n] = w[n]^T u[n]
    error       e[n] = d[n] - y[n]           (d = desired / reference signal)
    update      w[n+1] = w[n] + (gain vector) * e[n]

Typical configurations
    System identification : x -> unknown plant -> d ; w converges to plant taps
    Noise cancellation    : d = signal + noise filtered by path P, x = noise ref ;
                            e[n] converges to the clean signal
    Line enhancement      : x = delayed d ; y keeps the predictable (narrowband) part

Algorithms (cost per sample)
    LMS  : w += mu e u                        O(M)    stable if 0 < mu < 2/(M * Px)
    NLMS : w += mu e u / (eps + u^T u)        O(M)    stable for 0 < mu < 2, scale-free
    RLS  : Kalman-like gain from inverse correlation P   O(M^2), fastest convergence
"""
from __future__ import annotations

import numpy as np

__all__ = ["lms", "nlms", "rls", "regressor_matrix"]


def regressor_matrix(x, M):
    """U[n] = [x[n], x[n-1], ..., x[n-M+1]] with zeros before n = 0. Shape (len(x), M)."""
    x = np.asarray(x, float)
    xp = np.concatenate([np.zeros(M - 1), x])
    # sliding_window_view gives oldest..newest; reverse to newest-first
    return np.lib.stride_tricks.sliding_window_view(xp, M)[:, ::-1]


def lms(x, d, M, mu, w0=None, return_history=False):
    """Least-mean-squares: stochastic gradient descent on E{e^2}.
    Gradient estimate of e^2 w.r.t. w is -2 e u -> step w += mu e u (factor 2 absorbed)."""
    U = regressor_matrix(x, M)
    d = np.asarray(d, float)
    w = np.zeros(M) if w0 is None else np.array(w0, float)
    y = np.empty(len(d))
    e = np.empty(len(d))
    W = np.empty((len(d), M)) if return_history else None
    for n in range(len(d)):
        u = U[n]
        y[n] = w @ u
        e[n] = d[n] - y[n]
        w = w + mu * e[n] * u
        if return_history:
            W[n] = w
    return (y, e, w, W) if return_history else (y, e, w)


def nlms(x, d, M, mu=0.5, eps=1e-8, w0=None, return_history=False):
    """Normalised LMS: step divided by the instantaneous input power ||u||^2,
    which makes convergence speed independent of the input level."""
    U = regressor_matrix(x, M)
    d = np.asarray(d, float)
    w = np.zeros(M) if w0 is None else np.array(w0, float)
    y = np.empty(len(d))
    e = np.empty(len(d))
    W = np.empty((len(d), M)) if return_history else None
    for n in range(len(d)):
        u = U[n]
        y[n] = w @ u
        e[n] = d[n] - y[n]
        w = w + (mu / (eps + u @ u)) * e[n] * u
        if return_history:
            W[n] = w
    return (y, e, w, W) if return_history else (y, e, w)


def rls(x, d, M, lam=0.999, delta=100.0, w0=None, return_history=False):
    """Exponentially weighted recursive least squares.

    Minimises sum_i lam^(n-i) e[i]^2 exactly at every n. With P = inverse of the
    weighted input autocorrelation matrix (initialised to delta * I):
        k  = P u / (lam + u^T P u)          gain vector
        e  = d - w^T u                      a-priori error
        w += k e
        P  = (P - k u^T P) / lam            matrix inversion lemma
    lam in (0.95, 1]: effective memory ~ 1/(1 - lam) samples.
    """
    U = regressor_matrix(x, M)
    d = np.asarray(d, float)
    w = np.zeros(M) if w0 is None else np.array(w0, float)
    P = delta * np.eye(M)
    y = np.empty(len(d))
    e = np.empty(len(d))
    W = np.empty((len(d), M)) if return_history else None
    for n in range(len(d)):
        u = U[n]
        Pu = P @ u
        k = Pu / (lam + u @ Pu)
        y[n] = w @ u
        e[n] = d[n] - y[n]
        w = w + k * e[n]
        P = (P - np.outer(k, Pu)) / lam
        if return_history:
            W[n] = w
    return (y, e, w, W) if return_history else (y, e, w)
