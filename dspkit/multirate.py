"""
dspkit.multirate -- decimation, interpolation, polyphase structures and CIC filters.

Key identities
--------------
Noble identities (move the rate changer across a filter):
    [H(z^M)] -> (down M)   ==   (down M) -> [H(z)]
    (up L)   -> [H(z^L)]   ==   [H(z)]   -> (up L)

Type-1 polyphase decomposition of an FIR h[n] into M branches:
    H(z) = sum_{k=0}^{M-1} z^-k E_k(z^M),   E_k[n] = h[nM + k]
Applying the noble identity puts every E_k AFTER the downsampler, so each branch
runs at the LOW rate: M times fewer multiplies than filter-then-discard.

CIC (cascaded integrator-comb, Hogenauer 1981):
    H(z) = [ (1 - z^{-RM}) / (1 - z^-1) ]^N      (N stages, rate R, diff. delay M)
         = boxcar of length RM, convolved N times. Zero multipliers.
    Integrators run at the high rate, combs at the low rate. Integrators overflow
    for any DC input, but with two's-complement WRAP arithmetic and register width
        B_out = B_in + ceil(N log2(R M))
    the output is still exact (all arithmetic is mod 2^B and the true output fits).
"""
from __future__ import annotations

import numpy as np

from .core import convolve
from .fir import firwin, firls_grid
from .fixed import QFormat, handle_overflow

__all__ = [
    "downsample", "upsample", "decimate_direct", "interpolate_direct",
    "polyphase_components", "polyphase_decimate", "polyphase_interpolate",
    "resample_poly", "antialias_filter",
    "cic_register_width", "cic_decimate", "cic_interpolate", "cic_response",
    "cic_impulse", "cic_compensator",
]


# --------------------------------------------------------------------------- #
def downsample(x, M, phase=0):
    """y[m] = x[mM + phase]."""
    return np.asarray(x)[phase::M]


def upsample(x, L):
    """Zero stuffing: y[nL] = x[n], zeros elsewhere. Creates L-1 spectral images."""
    x = np.asarray(x)
    y = np.zeros(len(x) * L, dtype=x.dtype)
    y[::L] = x
    return y


def antialias_filter(M, numtaps=None, window="kaiser", beta=8.0):
    """Lowpass with cutoff fs_in/(2M) (Nyquist of the low rate)."""
    numtaps = numtaps or 16 * M + 1
    return firwin(numtaps, 1.0 / M, window=window, beta=beta)


def decimate_direct(x, h, M):
    """Reference: full-rate filtering, then discard M-1 of every M outputs.
    Output length = ceil((len(x)+len(h)-1)/M)."""
    return downsample(convolve(x, h), M)


def interpolate_direct(x, h, L):
    """Reference: zero-stuff, then filter at the high rate (most products are x 0).
    h should have passband gain L to restore amplitude."""
    return convolve(upsample(x, L)[: (len(x) - 1) * L + 1], h)


# --------------------------------------------------------------------------- #
def polyphase_components(h, M):
    """E[k, n] = h[nM + k], shape (M, ceil(len(h)/M)) -- zero-padded."""
    h = np.asarray(h, float)
    P = int(np.ceil(len(h) / M))
    hp = np.pad(h, (0, P * M - len(h)))
    return hp.reshape(P, M).T


def polyphase_decimate(x, h, M):
    """Polyphase decimator ('input commutator').

        y[m] = sum_k (E_k * x_k)[m],   x_k[m] = x[mM - k]   (x[<0] = 0)

    Branch k sees every M-th input sample, offset by k. Every multiply produces a
    sample that is kept. Matches decimate_direct exactly.
    """
    x = np.asarray(x, float)
    E = polyphase_components(h, M)
    n_out = int(np.ceil((len(x) + len(h) - 1) / M))
    y = np.zeros(n_out + E.shape[1])
    for k in range(M):
        # x_k[m] = x[mM - k]: prepend k zeros, then take every M-th sample
        xk = np.concatenate([np.zeros(k), x])[::M]
        yk = convolve(xk, E[k])
        y[:len(yk)] += yk
    return y[:n_out]


def polyphase_interpolate(x, h, L):
    """Polyphase interpolator ('output commutator').

        y[nL + k] = (E_k * x)[n],   k = 0..L-1

    Each of the L branches filters the LOW-rate input; outputs are interleaved.
    No multiplications by stuffed zeros. Matches interpolate_direct exactly.
    """
    x = np.asarray(x, float)
    E = polyphase_components(h, L)
    branch = [convolve(x, E[k]) for k in range(L)]
    n = len(branch[0])
    y = np.zeros(n * L)
    for k in range(L):
        y[k::L] = branch[k]
    n_ref = (len(x) - 1) * L + 1 + len(h) - 1
    return y[:n_ref]


def resample_poly(x, L, M, h=None):
    """Rational rate change L/M: up L -> lowpass at min(pi/L, pi/M) -> down M.
    (Didactic version: polyphase interpolation, then plain downsampling. The fully
    efficient form computes only the kept outputs -- left as an exercise.)"""
    g = np.gcd(L, M)
    L, M = L // g, M // g
    if h is None:
        K = max(L, M)
        h = L * firwin(20 * K + 1, 1.0 / K, window="kaiser", beta=8.0)
    y = polyphase_interpolate(x, h, L)
    return downsample(y, M)


# --------------------------------------------------------------------------- #
# CIC
# --------------------------------------------------------------------------- #
def cic_register_width(in_bits, R, N, M=1):
    """Hogenauer: B_out = B_in + ceil(N log2(RM)) bits (all stages, no pruning)."""
    return int(in_bits + np.ceil(N * np.log2(R * M)))


def cic_decimate(x_codes, R, N, M=1, in_bits=16):
    """Bit-true CIC decimator on integer input codes.

    N integrators at the input rate  : v <- v + x        (wrap to B bits)
    downsample by R
    N combs at the output rate       : u[m] = v[m] - v[m-M] (wrap to B bits)
    Output gain = (RM)^N, i.e. log2((RM)^N) extra bits (the bit growth).
    """
    B = cic_register_width(in_bits, R, N, M)
    q = QFormat(B, 0)
    v = np.asarray(x_codes, dtype=np.int64)
    for _ in range(N):
        v = handle_overflow(np.cumsum(v), q, "wrap")      # integrator (int64 wraps mod 2^64 first: still consistent mod 2^B)
    v = v[::R]
    for _ in range(N):
        delayed = np.concatenate([np.zeros(M, dtype=np.int64), v[:-M]])
        v = handle_overflow(v - delayed, q, "wrap")       # comb
    return v


def cic_interpolate(x_codes, R, N, M=1, in_bits=16):
    """Bit-true CIC interpolator: N combs (low rate) -> zero-stuff by R ->
    N integrators (high rate). Gain = (RM)^N / R."""
    B = int(in_bits + np.ceil(N * np.log2(R * M) - np.log2(R)))
    q = QFormat(B + 1, 0)
    v = np.asarray(x_codes, dtype=np.int64)
    for _ in range(N):
        delayed = np.concatenate([np.zeros(M, dtype=np.int64), v[:-M]])
        v = handle_overflow(v - delayed, q, "wrap")
    v = upsample(v, R)
    for _ in range(N):
        v = handle_overflow(np.cumsum(v), q, "wrap")
    return v


def cic_impulse(R, N, M=1):
    """Equivalent FIR of the CIC: boxcar(RM) convolved N times (length N(RM-1)+1)."""
    h = np.ones(1)
    for _ in range(N):
        h = convolve(h, np.ones(R * M))
    return h


def cic_response(f, R, N, M=1):
    """Normalised magnitude |H(f)| / (RM)^N, f in cycles per INPUT sample:
        | sin(pi R M f) / (R M sin(pi f)) |^N
    Nulls at multiples of f = 1/(RM): these fold onto DC at the output rate,
    giving the CIC its anti-aliasing property around each alias band."""
    f = np.asarray(f, float)
    num = np.sin(np.pi * R * M * f)
    den = R * M * np.sin(np.pi * f)
    with np.errstate(invalid="ignore", divide="ignore"):
        H = np.where(np.abs(den) < 1e-12, 1.0, num / den)
    return np.abs(H) ** N


def cic_compensator(numtaps, R, N, M=1, f_pass=0.4, f_stop=0.6, stop_weight=10.0,
                    density=32):
    """FIR at the OUTPUT rate that flattens the CIC passband droop.

    Frequencies f_pass, f_stop are normalised to the OUTPUT Nyquist (1.0).
    Output frequency f (Nyquist units) corresponds to f/(2R) cycles per input sample.
    Desired: 1/|H_cic| on [0, f_pass], 0 on [f_stop, 1]  -> weighted least squares.
    """
    df = 1.0 / (density * numtaps)
    fp = np.arange(0, f_pass + df / 2, df)
    fsb = np.arange(f_stop, 1 + df / 2, df)
    Dp = 1.0 / cic_response(fp / (2 * R), R, N, M)
    w = np.pi * np.concatenate([fp, fsb])
    D = np.concatenate([Dp, np.zeros(len(fsb))])
    W = np.concatenate([np.ones(len(fp)), np.full(len(fsb), stop_weight)])
    return firls_grid(numtaps, w, D, W)
