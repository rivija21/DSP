"""
dspkit.core -- filtering primitives and frequency-domain analysis.

Conventions used across the whole package
-----------------------------------------
* Transfer function H(z) = B(z)/A(z) with coefficients in ASCENDING powers of z^-1:
      b = [b0, b1, ..., bM]   ->  B(z) = b0 + b1 z^-1 + ... + bM z^-M
      a = [a0, a1, ..., aN]   ->  A(z) = a0 + a1 z^-1 + ... + aN z^-N
  (identical to scipy.signal / MATLAB).
* Frequencies are given with an explicit sample rate `fs`. Default fs = 2.0, so
  frequencies are normalised to Nyquist = 1.0 (MATLAB convention).
* Filter state `zi` / `zf` is the Direct-Form-II-Transposed delay-line content,
  length max(len(a), len(b)) - 1. Passing zf of one block as zi of the next makes
  block processing identical to processing the whole signal at once.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "lfilter", "fir_filter", "convolve", "impulse_response",
    "freqz", "sosfreqz", "group_delay", "mag_db",
    "tf2zpk", "zpk2tf", "normalize_tf",
]


# --------------------------------------------------------------------------- #
# Time-domain filtering
# --------------------------------------------------------------------------- #
def normalize_tf(b, a):
    """Return (b, a) as float arrays, both divided by a[0] so that a[0] == 1."""
    b = np.atleast_1d(np.asarray(b, dtype=float))
    a = np.atleast_1d(np.asarray(a, dtype=float))
    if a[0] == 0:
        raise ValueError("a[0] must be non-zero")
    return b / a[0], a / a[0]


def convolve(x, h):
    """Full linear convolution y[n] = sum_k h[k] x[n-k], len = len(x)+len(h)-1.

    Written as an explicit 'shift-and-accumulate' over the SHORTER sequence so
    the cost is O(len(x) * len(h)) with only len(h) vectorised numpy ops.
    This is exactly the structure of a transposed-form FIR in hardware.
    """
    x = np.asarray(x)
    h = np.asarray(h)
    if len(h) > len(x):
        x, h = h, x
    dtype = np.result_type(x, h)
    y = np.zeros(len(x) + len(h) - 1, dtype=dtype)
    for k, hk in enumerate(h):           # one tap at a time
        y[k:k + len(x)] += hk * x        # x delayed by k, scaled by h[k]
    return y


def fir_filter(b, x, zi=None):
    """Causal FIR filter, streaming-capable.

    Parameters
    ----------
    b  : (M+1,) taps.
    x  : (L,) input block.
    zi : (M,) previous M input samples, newest first (x[-1], x[-2], ...).
         None -> zeros (filter starts at rest).

    Returns
    -------
    y  : (L,) output, same length as x.
    zf : (M,) state to pass as `zi` for the next block.
    """
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    M = len(b) - 1
    hist = np.zeros(M) if zi is None else np.asarray(zi, dtype=float)
    # Prepend history (oldest first) so that xe[M + n] == x[n].
    xe = np.concatenate([hist[::-1], x])
    y = np.zeros(len(x))
    for k in range(M + 1):
        y += b[k] * xe[M - k:M - k + len(x)]
    zf = xe[::-1][:M].copy()                 # newest first
    return y, zf


def lfilter(b, a, x, zi=None):
    """General IIR/FIR filter, Direct Form II Transposed (DF2T), sample by sample.

    Difference equation (a[0] normalised to 1):
        y[n] = sum_{k=0}^{M} b[k] x[n-k] - sum_{k=1}^{N} a[k] y[n-k]

    DF2T realisation with K = max(M, N) state registers s[0..K-1]:
        y    = b0*x + s0
        s[i] = b[i+1]*x - a[i+1]*y + s[i+1]     (i = 0..K-2)
        s[K-1] = b[K]*x - a[K]*y

    The loop is deliberately explicit -- it is the reference model you will later
    translate to C / HDL. It is slow in Python (~1 us/sample/section);
    use scipy.signal.lfilter for production throughput.

    Returns
    -------
    y  : output, len(x)
    zf : final state, length K. Feed back as `zi` for block-wise streaming.
    """
    b, a = normalize_tf(b, a)
    K = max(len(a), len(b)) - 1
    b = np.pad(b, (0, K + 1 - len(b)))
    a = np.pad(a, (0, K + 1 - len(a)))
    s = np.zeros(K) if zi is None else np.array(zi, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    for n, xn in enumerate(x):
        yn = b[0] * xn + (s[0] if K else 0.0)
        if K:
            # vectorised state update: s[0:K-1] = b[1:K]x - a[1:K]y + s[1:K]
            s[:-1] = b[1:K] * xn - a[1:K] * yn + s[1:]
            s[-1] = b[K] * xn - a[K] * yn
        y[n] = yn
    return y, s


def impulse_response(b, a=1.0, n=64):
    """First n samples of h[n] = IDTFT{H}, obtained by filtering a unit impulse."""
    d = np.zeros(n)
    d[0] = 1.0
    return lfilter(b, a, d)[0]


# --------------------------------------------------------------------------- #
# Frequency-domain analysis
# --------------------------------------------------------------------------- #
def _freq_grid(worN, whole, fs):
    if np.isscalar(worN):
        n = int(worN)
        stop = fs if whole else fs / 2
        f = np.arange(n) * stop / n          # endpoint excluded (scipy behaviour)
    else:
        f = np.asarray(worN, dtype=float)
    return f


def freqz(b, a=1.0, worN=512, whole=False, fs=2.0):
    """Frequency response H(e^{jw}) evaluated directly on the unit circle.

    H(e^{jw}) = sum_k b[k] e^{-jwk} / sum_k a[k] e^{-jwk}

    Parameters
    ----------
    worN : int (number of points on [0, fs/2) or [0, fs)) or array of frequencies.
    fs   : sample rate in the same units as the returned frequency axis.

    Returns
    -------
    f : frequencies (units of fs)
    H : complex response
    """
    b = np.atleast_1d(np.asarray(b, dtype=complex))
    a = np.atleast_1d(np.asarray(a, dtype=complex))
    f = _freq_grid(worN, whole, fs)
    w = 2 * np.pi * f / fs                         # rad/sample
    E = np.exp(-1j * np.outer(w, np.arange(max(len(a), len(b)))))
    num = E[:, :len(b)] @ b                        # polynomial in z^-1
    den = E[:, :len(a)] @ a
    return f, num / den


def sosfreqz(sos, worN=512, whole=False, fs=2.0):
    """Response of a cascade of second-order sections = product of section responses."""
    sos = np.atleast_2d(sos)
    f = _freq_grid(worN, whole, fs)
    H = np.ones(len(f), dtype=complex)
    for sec in sos:
        H *= freqz(sec[:3], sec[3:], f, fs=fs)[1]
    return f, H


def group_delay(b, a=1.0, worN=512, fs=2.0):
    """Group delay tau(w) = -d(arg H)/dw in samples.

    Uses the exact identity for a polynomial P(z) = sum p[k] z^-k:
        tau_P(w) = Re{ sum k p[k] e^{-jwk} / sum p[k] e^{-jwk} }
    and tau_H = tau_B - tau_A. (No numerical differentiation of phase,
    which would suffer from phase-wrapping.)
    """
    b = np.atleast_1d(np.asarray(b, dtype=float))
    a = np.atleast_1d(np.asarray(a, dtype=float))
    f = _freq_grid(worN, False, fs)

    def tau(p):
        k = np.arange(len(p))
        _, P = freqz(p, 1.0, f, fs=fs)
        _, Pr = freqz(k * p, 1.0, f, fs=fs)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.real(Pr / P)

    return f, tau(b) - tau(a)


def mag_db(H, floor=-300.0):
    """20 log10 |H| with a floor to avoid -inf."""
    return np.maximum(20 * np.log10(np.abs(H) + 1e-300), floor)


# --------------------------------------------------------------------------- #
# Representations
# --------------------------------------------------------------------------- #
def tf2zpk(b, a):
    """(b, a) in z^-1 powers -> zeros, poles, gain.

    Pad b and a to a common length n+1; then
        H(z) = (b0 z^n + b1 z^(n-1) + ...) / (z^n + a1 z^(n-1) + ...)
    so poles = roots(a) and zeros = roots(b) after stripping leading zeros of b
    (each leading zero lowers the numerator degree by one).
    Trailing zeros produced by padding a correctly give poles at z = 0.
    """
    b, a = normalize_tf(b, a)
    n = max(len(a), len(b))
    b = np.pad(b, (0, n - len(b)))
    a = np.pad(a, (0, n - len(a)))
    lead = int(np.argmax(b != 0))
    k = b[lead]
    z = np.roots(b[lead:])
    p = np.roots(a)
    return z, p, k


def zpk2tf(z, p, k):
    """zeros, poles, gain -> (b, a) in z^-1 powers.

    H(z) = k prod(z - z_i) / prod(z - p_i). np.poly builds the monic polynomial
    from its roots. If there are fewer zeros than poles, the numerator is
    prefixed with (len(p) - len(z)) zeros, i.e. the exact z^-d delay is kept.
    (scipy.signal.zpk2tf omits this prefix -- same magnitude, different phase.)
    """
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    b = k * np.poly(z) if len(z) else np.atleast_1d(float(np.real(k)))
    a = np.poly(p) if len(p) else np.array([1.0])
    if len(p) > len(z):
        b = np.concatenate([np.zeros(len(p) - len(z)), b])
    # Real-coefficient filters have conjugate-symmetric roots: drop rounding imag.
    return np.real(b).astype(float), np.real(a).astype(float)
