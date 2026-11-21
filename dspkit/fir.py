"""
dspkit.fir -- linear-phase FIR design from first principles.

Methods
-------
1. Window method      : truncate the ideal (sinc) response, taper with a window.
2. Kaiser formulas    : pick (numtaps, beta) directly from a ripple/transition spec.
3. Frequency sampling : specify N samples of the amplitude response, inverse DFT.
4. Least squares      : minimise integral of weighted squared error (closed form).
5. Parks-McClellan    : minimise the maximum weighted error (Remez exchange) --
                        the optimal equiripple filter.

All functions return the impulse response h[0..N-1] (= FIR taps b).
Band edges are given in the units of `fs` (default fs=2 -> Nyquist = 1).

Linear-phase cheat sheet (N taps, symmetric h[n] = h[N-1-n])
-----------------------------------------------------------
  Type I  : N odd,  symmetric    -> any response.          A(w) = sum a_k cos(kw)
  Type II : N even, symmetric    -> H(pi) = 0  (no highpass)
  Type III: N odd,  antisymmetric-> H(0)=H(pi)=0 (differentiator / Hilbert)
  Type IV : N even, antisymmetric-> H(0) = 0
The LS and Remez designers here are Type I only (keeps the maths minimal).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "bessel_i0", "get_window", "firwin", "kaiser_beta", "kaiserord",
    "freq_sampling", "firls", "firls_grid", "remez",
]


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def bessel_i0(x, terms=40):
    """Modified Bessel function of the first kind, order 0, by power series:
        I0(x) = sum_{k>=0} ((x/2)^k / k!)^2
    Converges fast for the x range used by Kaiser windows (x < ~50).
    """
    x = np.asarray(x, dtype=float)
    term = np.ones_like(x)
    acc = np.ones_like(x)
    q = (x / 2.0) ** 2
    for k in range(1, terms):
        term = term * q / (k * k)
        acc = acc + term
    return acc


def get_window(name, N, beta=None):
    """Symmetric length-N window (as used for filter design).

    n runs 0..N-1 and the 'phase' variable is t = n/(N-1) in [0, 1].
      rect     : 1
      hann     : 0.5 - 0.5 cos(2 pi t)
      hamming  : 0.54 - 0.46 cos(2 pi t)
      blackman : 0.42 - 0.5 cos(2 pi t) + 0.08 cos(4 pi t)
      kaiser   : I0(beta sqrt(1 - (2t-1)^2)) / I0(beta)
    """
    if N == 1:
        return np.ones(1)
    t = np.arange(N) / (N - 1)
    name = name.lower()
    if name in ("rect", "rectangular", "boxcar"):
        return np.ones(N)
    if name in ("hann", "hanning"):
        return 0.5 - 0.5 * np.cos(2 * np.pi * t)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * t)
    if name == "blackman":
        return 0.42 - 0.5 * np.cos(2 * np.pi * t) + 0.08 * np.cos(4 * np.pi * t)
    if name == "kaiser":
        if beta is None:
            raise ValueError("kaiser window needs beta")
        r = 2 * t - 1
        return bessel_i0(beta * np.sqrt(np.clip(1 - r * r, 0, None))) / bessel_i0(beta)
    raise ValueError(f"unknown window {name!r}")


# --------------------------------------------------------------------------- #
# 1. Window method
# --------------------------------------------------------------------------- #
def firwin(numtaps, cutoff, window="hamming", pass_zero=True, fs=2.0, beta=None, scale=True):
    """Windowed-sinc FIR design (lowpass / highpass / bandpass / bandstop / multiband).

    The ideal response is a sum of rectangular passbands [l, r] (normalised to
    Nyquist = 1). Each passband contributes
        h_ideal[m] = r * sinc(r m) - l * sinc(l m),  m = n - (N-1)/2
    because the inverse DTFT of a lowpass with edge w_c = pi*r is r*sinc(r m)
    (np.sinc(x) = sin(pi x)/(pi x)).

    Parameters
    ----------
    cutoff    : scalar or increasing list of band edges (units of fs).
    pass_zero : True  -> first band [0, cutoff[0]] is a passband (lowpass/bandstop)
                False -> first band is a stopband (highpass/bandpass)
    scale     : normalise gain to exactly 1 at the centre of the first passband
                (DC for lowpass, Nyquist for highpass, centre otherwise).
    """
    cutoff = np.atleast_1d(np.asarray(cutoff, dtype=float)) / (fs / 2)
    if np.any(cutoff <= 0) or np.any(cutoff >= 1):
        raise ValueError("cutoff must be strictly between 0 and fs/2")
    # Build the list of passband edges [0, c0, c1, ..., 1] and pick alternate bands.
    edges = np.concatenate([[0.0], cutoff, [1.0]])
    first = 0 if pass_zero else 1
    bands = [(edges[i], edges[i + 1]) for i in range(first, len(edges) - 1, 2)]
    if bands[-1][1] == 1.0 and numtaps % 2 == 0:
        raise ValueError("passband at Nyquist requires odd numtaps (Type I)")

    m = np.arange(numtaps) - (numtaps - 1) / 2
    h = np.zeros(numtaps)
    for l, r in bands:
        h += r * np.sinc(r * m) - l * np.sinc(l * m)

    h *= get_window(window, numtaps, beta)

    if scale:
        l, r = bands[0]
        f0 = 0.0 if l == 0 else (1.0 if r == 1.0 else (l + r) / 2)
        g = np.sum(h * np.cos(np.pi * m * f0))      # |H| at f0 for a linear-phase FIR
        h /= g
    return h


# --------------------------------------------------------------------------- #
# 2. Kaiser design formulas
# --------------------------------------------------------------------------- #
def kaiser_beta(atten_db):
    """Kaiser's empirical beta for a stopband attenuation A (dB)."""
    A = atten_db
    if A > 50:
        return 0.1102 * (A - 8.7)
    if A > 21:
        return 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    return 0.0


def kaiserord(atten_db, width, fs=2.0):
    """(numtaps, beta) meeting attenuation A dB with transition width `width`.

    N - 1 = (A - 7.95) / (2.285 * dw),  dw = transition width in rad/sample.
    `width` is in units of fs (default: fraction of Nyquist).
    """
    dw = np.pi * width / (fs / 2)
    numtaps = int(np.ceil((atten_db - 7.95) / (2.285 * dw) + 1))
    return numtaps, kaiser_beta(atten_db)


# --------------------------------------------------------------------------- #
# 3. Frequency sampling
# --------------------------------------------------------------------------- #
def freq_sampling(A):
    """Type I frequency-sampling design.

    A : amplitude samples A[k] at w_k = 2 pi k / N, k = 0..L, with N = 2L+1.
    The linear-phase DFT is H[k] = A[k] e^{-j w_k L}; its inverse DFT is real:

        h[n] = (1/N) [ A[0] + 2 sum_{k=1}^{L} A[k] cos(w_k (n - L)) ]

    The response interpolates A[k] exactly at w_k; between samples it ripples.
    Setting transition samples to intermediate values (e.g. 0.4) reduces the
    ripple -- the classic optimisation exercise.
    """
    A = np.asarray(A, dtype=float)
    L = len(A) - 1
    N = 2 * L + 1
    n = np.arange(N)
    k = np.arange(1, L + 1)
    wk = 2 * np.pi * k / N
    h = (A[0] + 2 * np.cos(np.outer(n - L, wk)) @ A[1:]) / N
    return h


# --------------------------------------------------------------------------- #
# 4. Least squares (Type I)
# --------------------------------------------------------------------------- #
def firls_grid(numtaps, w, D, W=None):
    """Weighted least-squares Type I FIR on an arbitrary frequency grid.

    Amplitude model: A(w) = sum_{k=0}^{L} a_k cos(k w),  L = (N-1)/2.
    Minimise   sum_i W_i (A(w_i) - D_i)^2   ->  normal equations  (C^T W C) a = C^T W D,
    where C[i, k] = cos(k w_i). Solved with lstsq on sqrt(W)-scaled rows.

    w : grid in rad/sample (0..pi), D : desired amplitude, W : weights (>=0).
    Returns taps h with h[L] = a_0 and h[L +- k] = a_k / 2.
    """
    if numtaps % 2 == 0:
        raise ValueError("firls_grid implements Type I: numtaps must be odd")
    L = (numtaps - 1) // 2
    w = np.asarray(w, float)
    D = np.asarray(D, float)
    W = np.ones_like(w) if W is None else np.asarray(W, float)
    C = np.cos(np.outer(w, np.arange(L + 1)))
    sw = np.sqrt(W)
    a, *_ = np.linalg.lstsq(C * sw[:, None], D * sw, rcond=None)
    return _cos_coeffs_to_taps(a)


def _cos_coeffs_to_taps(a):
    L = len(a) - 1
    h = np.zeros(2 * L + 1)
    h[L] = a[0]
    h[L + 1:] = a[1:] / 2
    h[:L] = a[1:][::-1] / 2
    return h


def _band_grid(numtaps, bands, desired, weight, fs, density):
    """Dense grid over the bands. Returns w (rad/s), D, W and band index per point.
    `desired` gives the amplitude at each band edge (linear in between)."""
    bands = np.asarray(bands, float) / (fs / 2)          # -> 0..1
    desired = np.asarray(desired, float)
    nb = len(bands) // 2
    weight = np.ones(nb) if weight is None else np.asarray(weight, float)
    df = 1.0 / (density * numtaps)                       # grid step (Nyquist units)
    w, D, W, bid = [], [], [], []
    for i in range(nb):
        f0, f1 = bands[2 * i], bands[2 * i + 1]
        n = max(int(np.ceil((f1 - f0) / df)) + 1, 2)
        f = np.linspace(f0, f1, n)
        w.append(np.pi * f)
        D.append(np.interp(f, [f0, f1], desired[2 * i:2 * i + 2]))
        W.append(np.full(n, weight[i]))
        bid.append(np.full(n, i))
    return (np.concatenate(w), np.concatenate(D),
            np.concatenate(W), np.concatenate(bid))


def firls(numtaps, bands, desired, weight=None, fs=2.0, density=32):
    """Least-squares multiband FIR (Type I).

    bands   : [f0, f1, f2, f3, ...] pairs of band edges (units of fs).
    desired : amplitude at each band edge (same length as bands).
    weight  : one weight per band.
    The integral of squared error is approximated by a dense uniform grid
    (Riemann sum), so results match scipy.signal.firls closely.
    """
    w, D, W, _ = _band_grid(numtaps, bands, desired, weight, fs, density)
    return firls_grid(numtaps, w, D, W)


# --------------------------------------------------------------------------- #
# 5. Parks-McClellan / Remez exchange (Type I)
# --------------------------------------------------------------------------- #
def _bary_weights(x):
    """Barycentric weights w_k = 1 / prod_{i!=k} (x_k - x_i).
    Products are computed after scaling to avoid overflow/underflow."""
    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)
    # scale each factor by 2 (x in [-1,1] -> differences in [-2,2]) for stability
    return 1.0 / np.prod(diff * 2.0, axis=1)


def remez(numtaps, bands, desired, weight=None, fs=2.0, density=16,
          maxiter=40, tol=1e-9, return_info=False):
    """Parks-McClellan equiripple FIR (Type I, piecewise-constant bands).

    Problem: choose a_0..a_L so that  E(w) = W(w) [D(w) - A(w)]  has the smallest
    possible max |E| over the bands. Chebyshev alternation theorem: the optimum
    has >= L+2 extremal frequencies where E alternates sign with equal |E| = delta.

    Remez exchange loop
      1. Guess L+2 extremal frequencies w_k (on a dense grid).
      2. Solve for delta that makes the error exactly +-delta (alternating) there:
             delta = sum b_k D_k / sum b_k (-1)^k / W_k,   b_k = barycentric weights
         (closed form of the (L+2)x(L+2) linear system).
      3. Interpolate A(w) through the L+1 points C_k = D_k - (-1)^k delta / W_k
         (barycentric Lagrange in x = cos w -- A is a polynomial in cos w).
      4. Evaluate E on the grid, pick the L+2 largest alternating local extrema
         as the new extremal set.
      5. Stop when max|E| ~= |delta| (the error is equiripple).
    Finally sample A(w) and inverse-DFT to obtain h[n].

    Parameters mirror scipy.signal.remez (bands in units of fs, one desired
    value and one weight per band).
    """
    if numtaps % 2 == 0:
        raise ValueError("this remez implements Type I: numtaps must be odd")
    L = (numtaps - 1) // 2
    r = L + 2                                         # number of extremals
    desired = np.asarray(desired, float)
    desired_edges = np.repeat(desired, 2)             # constant per band
    w, D, W, bid = _band_grid(numtaps, bands, desired_edges, weight, fs, density)
    x = np.cos(w)
    ng = len(w)
    if ng < r:
        raise ValueError("grid too coarse; increase density")

    ext = np.round(np.linspace(0, ng - 1, r)).astype(int)     # 1. initial guess
    delta = 0.0
    for it in range(maxiter):
        xk = x[ext]
        bk = _bary_weights(xk)
        sgn = (-1.0) ** np.arange(r)
        delta = np.dot(bk, D[ext]) / np.dot(bk, sgn / W[ext])  # 2.
        Ck = D[ext] - sgn * delta / W[ext]

        # 3. interpolate through first L+1 points
        xi, Ci = xk[:-1], Ck[:-1]
        bi = _bary_weights(xi)
        A = _bary_eval(x, xi, Ci, bi)
        E = W * (D - A)

        # 4. new extremal set
        new_ext = _find_extrema(E, bid, r)
        if new_ext is None:
            break
        Emax = np.max(np.abs(E[new_ext]))
        converged = (Emax - abs(delta)) <= 1e-6 * Emax or np.array_equal(new_ext, ext)
        ext = new_ext
        if converged:
            break

    # Final interpolation and conversion to taps: sample A at w_m = 2 pi m / N
    xk = x[ext]
    sgn = (-1.0) ** np.arange(r)
    bk = _bary_weights(xk)
    delta = np.dot(bk, D[ext]) / np.dot(bk, sgn / W[ext])
    Ck = D[ext] - sgn * delta / W[ext]
    xi, Ci = xk[:-1], Ck[:-1]
    bi = _bary_weights(xi)
    wm = 2 * np.pi * np.arange(L + 1) / numtaps
    Am = _bary_eval(np.cos(wm), xi, Ci, bi)
    h = freq_sampling(Am)
    if return_info:
        return h, {"delta": abs(delta), "iterations": it + 1, "extremal_w": w[ext]}
    return h


def _bary_eval(x, xi, Ci, bi):
    """Barycentric Lagrange interpolation p(x) = sum (b_i C_i/(x-x_i)) / sum (b_i/(x-x_i))."""
    d = x[:, None] - xi[None, :]
    exact = np.isclose(d, 0.0, atol=1e-14)
    d[exact] = 1.0
    t = bi / d
    p = (t @ Ci) / t.sum(axis=1)
    rows, cols = np.nonzero(exact)
    p[rows] = Ci[cols]
    return p


def _find_extrema(E, bid, r):
    """Indices of alternating local extrema of E (band edges included), reduced to r."""
    idx = []
    n = len(E)
    for i in range(n):
        left = E[i - 1] if i > 0 and bid[i - 1] == bid[i] else None
        right = E[i + 1] if i < n - 1 and bid[i + 1] == bid[i] else None
        e = E[i]
        if e >= 0:
            ok = (left is None or e >= left) and (right is None or e > right)
        else:
            ok = (left is None or e <= left) and (right is None or e < right)
        if ok:
            idx.append(i)
    idx = np.array(idx)
    if len(idx) == 0:
        return None
    # enforce alternation: among consecutive same-sign extrema keep the largest |E|
    keep = [idx[0]]
    for i in idx[1:]:
        if np.sign(E[i]) == np.sign(E[keep[-1]]):
            if abs(E[i]) > abs(E[keep[-1]]):
                keep[-1] = i
        else:
            keep.append(i)
    keep = list(keep)
    # too many: drop the smaller of the two ends (keeps alternation intact)
    while len(keep) > r:
        if abs(E[keep[0]]) < abs(E[keep[-1]]):
            keep.pop(0)
        else:
            keep.pop()
    if len(keep) < r:
        return None
    return np.array(keep)
