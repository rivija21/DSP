"""
dspkit.iir -- IIR design by the analog-prototype / bilinear-transform route.

Pipeline (all in zero-pole-gain form, which is numerically robust):

    analog lowpass prototype (wc = 1 rad/s)      buttap / cheb1ap / cheb2ap
            |
    pre-warp digital edges:  W = 2 fs' tan(pi f / fs)
            |
    analog frequency transform                    lp2lp / lp2hp / lp2bp / lp2bs
            |
    bilinear transform  s = 2 fs' (z-1)/(z+1)    bilinear_zpk
            |
    pair poles/zeros into 2nd-order sections      zpk2sos
            |
    run as cascade of DF2T biquads                sosfilt

Why zpk and SOS: polynomial (b, a) coefficients of high-order narrow-band
filters are extremely sensitive -- tiny coefficient errors move poles a lot
(see examples/02_iir_design.py). Roots are kept explicitly until the end and
the filter is realised as a cascade of biquads whose 2nd-order polynomials are
well conditioned.

SOS array layout: shape (n_sections, 6), each row [b0, b1, b2, a0=1, a1, a2].
"""
from __future__ import annotations

import numpy as np

from .core import zpk2tf

__all__ = [
    "buttap", "cheb1ap", "cheb2ap",
    "lp2lp_zpk", "lp2hp_zpk", "lp2bp_zpk", "lp2bs_zpk", "bilinear_zpk",
    "iirfilter", "butter", "cheby1", "cheby2",
    "zpk2sos", "sos2zpk", "sosfilt", "buttord",
]


# --------------------------------------------------------------------------- #
# Analog lowpass prototypes, cutoff 1 rad/s
# --------------------------------------------------------------------------- #
def buttap(N):
    """Butterworth: |H(jW)|^2 = 1 / (1 + W^(2N)).
    Poles equally spaced on the unit circle in the left half plane:
        p_k = exp(j pi (2k + N - 1) / (2N)),  k = 1..N
    """
    k = np.arange(1, N + 1)
    p = np.exp(1j * np.pi * (2 * k + N - 1) / (2 * N))
    return np.array([]), p, 1.0


def cheb1ap(N, rp):
    """Chebyshev I: equiripple passband (rp dB), monotone stopband.
    |H|^2 = 1 / (1 + eps^2 T_N^2(W)),  eps^2 = 10^(rp/10) - 1.
    Poles lie on an ellipse:
        p_k = -sinh(mu) sin(th_k) + j cosh(mu) cos(th_k),
        mu = asinh(1/eps)/N,  th_k = (2k-1) pi / (2N)
    Gain: |H(0)| = 1 for odd N, 1/sqrt(1+eps^2) (bottom of ripple) for even N.
    """
    eps = np.sqrt(10 ** (0.1 * rp) - 1)
    mu = np.arcsinh(1 / eps) / N
    th = (2 * np.arange(1, N + 1) - 1) * np.pi / (2 * N)
    p = -np.sinh(mu) * np.sin(th) + 1j * np.cosh(mu) * np.cos(th)
    k = np.real(np.prod(-p))
    if N % 2 == 0:
        k /= np.sqrt(1 + eps ** 2)
    return np.array([]), p, k


def cheb2ap(N, rs):
    """Chebyshev II (inverse Chebyshev): monotone passband, equiripple stopband
    with minimum attenuation rs dB; the stopband edge is at 1 rad/s.
    Obtained from Chebyshev I by W -> 1/W:
        zeros  z_k = j / cos(th_k)      (on the jW axis; none for cos(th_k)=0)
        poles  p_k = 1 / q_k, q_k = Chebyshev-I-like poles with eps = 1/sqrt(10^(rs/10)-1)
    """
    de = 1.0 / np.sqrt(10 ** (0.1 * rs) - 1)
    mu = np.arcsinh(1 / de) / N
    th = (2 * np.arange(1, N + 1) - 1) * np.pi / (2 * N)
    q = -np.sinh(mu) * np.sin(th) + 1j * np.cosh(mu) * np.cos(th)
    p = 1.0 / q
    c = np.cos(th)
    c = c[np.abs(c) > 1e-12]
    z = 1j / c
    k = np.real(np.prod(-p) / np.prod(-z))
    return z, p, k


# --------------------------------------------------------------------------- #
# Analog frequency transforms (zpk), s-plane
# --------------------------------------------------------------------------- #
def _degree(z, p):
    d = len(p) - len(z)
    if d < 0:
        raise ValueError("improper transfer function (more zeros than poles)")
    return d


def lp2lp_zpk(z, p, k, wo=1.0):
    """s -> s/wo : every root scales by wo; gain by wo^(#poles - #zeros)."""
    d = _degree(z, p)
    return wo * np.asarray(z), wo * np.asarray(p), k * wo ** d


def lp2hp_zpk(z, p, k, wo=1.0):
    """s -> wo/s : roots invert (r -> wo/r); the d excess poles create d zeros at s=0."""
    z, p = np.asarray(z, complex), np.asarray(p, complex)
    d = _degree(z, p)
    z_hp = np.concatenate([wo / z, np.zeros(d)])
    p_hp = wo / p
    k_hp = k * np.real(np.prod(-z) / np.prod(-p))
    return z_hp, p_hp, k_hp


def lp2bp_zpk(z, p, k, wo=1.0, bw=1.0):
    """s -> (s^2 + wo^2)/(bw s): each root r maps to the two roots of
    s^2 - r bw s + wo^2 = 0  ->  s = r bw/2 +- sqrt((r bw/2)^2 - wo^2).
    d excess poles give d zeros at s = 0 (and d at infinity)."""
    z, p = np.asarray(z, complex), np.asarray(p, complex)
    d = _degree(z, p)
    zl, pl = z * bw / 2, p * bw / 2
    z_bp = np.concatenate([zl + np.sqrt(zl ** 2 - wo ** 2), zl - np.sqrt(zl ** 2 - wo ** 2), np.zeros(d)])
    p_bp = np.concatenate([pl + np.sqrt(pl ** 2 - wo ** 2), pl - np.sqrt(pl ** 2 - wo ** 2)])
    return z_bp, p_bp, k * bw ** d


def lp2bs_zpk(z, p, k, wo=1.0, bw=1.0):
    """s -> (bw s)/(s^2 + wo^2): each root r maps to roots of r s^2 - bw s + r wo^2;
    the d excess poles become d pairs of zeros at s = +-j wo (the notch)."""
    z, p = np.asarray(z, complex), np.asarray(p, complex)
    d = _degree(z, p)
    zh, ph = (bw / 2) / z, (bw / 2) / p
    z_bs = np.concatenate([zh + np.sqrt(zh ** 2 - wo ** 2), zh - np.sqrt(zh ** 2 - wo ** 2),
                           np.full(d, 1j * wo), np.full(d, -1j * wo)])
    p_bs = np.concatenate([ph + np.sqrt(ph ** 2 - wo ** 2), ph - np.sqrt(ph ** 2 - wo ** 2)])
    k_bs = k * np.real(np.prod(-z) / np.prod(-p))
    return z_bs, p_bs, k_bs


# --------------------------------------------------------------------------- #
# Bilinear transform
# --------------------------------------------------------------------------- #
def bilinear_zpk(z, p, k, fs):
    """s = 2 fs (z - 1)/(z + 1)  <=>  z = (2 fs + s)/(2 fs - s).

    * Left-half-plane s maps inside the unit circle (stability preserved).
    * The jW axis maps onto the unit circle with  W = 2 fs tan(w/2)
      (frequency warping -- hence the pre-warp step).
    * Zeros at s = infinity (d = #p - #z of them) land at z = -1 (Nyquist).
    * Gain: k_d = k * prod(2fs - z) / prod(2fs - p).
    """
    z, p = np.asarray(z, complex), np.asarray(p, complex)
    d = _degree(z, p)
    fs2 = 2.0 * fs
    z_d = np.concatenate([(fs2 + z) / (fs2 - z), -np.ones(d)])
    p_d = (fs2 + p) / (fs2 - p)
    k_d = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))
    return z_d, p_d, k_d


# --------------------------------------------------------------------------- #
# Complete designer
# --------------------------------------------------------------------------- #
def iirfilter(N, Wn, rp=None, rs=None, btype="lowpass", ftype="butter",
              output="sos", fs=2.0):
    """Digital IIR design, same calling convention as scipy.signal.iirfilter.

    N     : prototype order (bandpass/bandstop filters end up with order 2N).
    Wn    : edge frequency (scalar) or [low, high] (units of fs).
    rp/rs : passband ripple / stopband attenuation in dB (Chebyshev).
    output: 'sos' | 'zpk' | 'ba'
    """
    Wn = np.atleast_1d(np.asarray(Wn, float)) / (fs / 2)       # normalise to Nyquist
    if np.any(Wn <= 0) or np.any(Wn >= 1):
        raise ValueError("edges must be in (0, fs/2)")

    if ftype == "butter":
        z, p, k = buttap(N)
    elif ftype == "cheby1":
        z, p, k = cheb1ap(N, rp)
    elif ftype == "cheby2":
        z, p, k = cheb2ap(N, rs)
    else:
        raise ValueError(ftype)

    # Pre-warp with an internal bilinear rate fs' = 2 (so Nyquist = 1):
    #   W_analog = 2 fs' tan(pi f / 2)  with f normalised to Nyquist.
    fsb = 2.0
    warped = 2 * fsb * np.tan(np.pi * Wn / fsb)

    if btype in ("lowpass", "low"):
        z, p, k = lp2lp_zpk(z, p, k, warped[0])
    elif btype in ("highpass", "high"):
        z, p, k = lp2hp_zpk(z, p, k, warped[0])
    elif btype in ("bandpass", "band"):
        bw = warped[1] - warped[0]
        wo = np.sqrt(warped[0] * warped[1])                      # geometric centre
        z, p, k = lp2bp_zpk(z, p, k, wo, bw)
    elif btype in ("bandstop", "stop"):
        bw = warped[1] - warped[0]
        wo = np.sqrt(warped[0] * warped[1])
        z, p, k = lp2bs_zpk(z, p, k, wo, bw)
    else:
        raise ValueError(btype)

    z, p, k = bilinear_zpk(z, p, k, fsb)

    if output == "zpk":
        return z, p, k
    if output == "ba":
        return zpk2tf(z, p, k)
    return zpk2sos(z, p, k)


def butter(N, Wn, btype="lowpass", output="sos", fs=2.0):
    return iirfilter(N, Wn, btype=btype, ftype="butter", output=output, fs=fs)


def cheby1(N, rp, Wn, btype="lowpass", output="sos", fs=2.0):
    return iirfilter(N, Wn, rp=rp, btype=btype, ftype="cheby1", output=output, fs=fs)


def cheby2(N, rs, Wn, btype="lowpass", output="sos", fs=2.0):
    return iirfilter(N, Wn, rs=rs, btype=btype, ftype="cheby2", output=output, fs=fs)


def buttord(wp, ws, gpass, gstop, fs=2.0):
    """Minimum Butterworth order for a lowpass spec (passband edge wp with at most
    gpass dB loss, stopband edge ws with at least gstop dB), using pre-warped edges:
        N >= log10[(10^(gstop/10)-1)/(10^(gpass/10)-1)] / (2 log10(Ws/Wp))
    Returns (N, Wn) where Wn is the -3 dB frequency that meets the passband exactly.
    """
    Wp = np.tan(np.pi * wp / fs)
    Ws = np.tan(np.pi * ws / fs)
    gp, gs = 10 ** (0.1 * gpass) - 1, 10 ** (0.1 * gstop) - 1
    N = int(np.ceil(np.log10(gs / gp) / (2 * np.log10(Ws / Wp))))
    Wc = Wp / gp ** (1 / (2 * N))
    Wn = np.arctan(Wc) * fs / np.pi
    return N, Wn


# --------------------------------------------------------------------------- #
# Second-order sections
# --------------------------------------------------------------------------- #
def _split_pairs(r, tol=1e-9):
    """Group roots into conjugate pairs and real roots.
    Returns list of complex roots with Im>0 (each stands for a pair) and array of reals."""
    r = np.asarray(r, complex)
    real_mask = np.abs(r.imag) <= tol * np.maximum(1, np.abs(r))
    reals = np.sort(r[real_mask].real)
    cplx = r[~real_mask]
    upper = cplx[cplx.imag > 0]
    if len(upper) * 2 != len(cplx):
        raise ValueError("complex roots are not in conjugate pairs")
    return list(upper), list(reals)


def zpk2sos(z, p, k):
    """Pair poles and zeros into biquads.

    Rule (Jackson / scipy 'nearest'): process pole pairs from the one CLOSEST to the
    unit circle (highest Q, most dominant) and give each the zero pair nearest to it,
    so large peaks are partially cancelled inside one section; then order sections so
    the highest-Q section comes LAST (minimises overflow / noise amplification in a
    fixed-point cascade). Overall gain k goes into the first section.

    Odd counts are completed with a root at the origin (b2 = 0 or a2 = 0).
    """
    z = np.asarray(z, complex)
    p = np.asarray(p, complex)
    # pad zeros with origin so that #z == #p (only needed for analog-like inputs)
    if len(z) < len(p):
        z = np.concatenate([z, np.zeros(len(p) - len(z))])

    def to_groups(r):
        up, re = _split_pairs(r)
        groups = [(c, np.conj(c)) for c in up]
        re = list(re)
        while len(re) >= 2:
            groups.append((re.pop(0), re.pop(0)))
        if re:
            groups.append((re.pop(0), 0.0))      # lone real root + root at origin
        return groups

    pg = to_groups(p)
    zg = to_groups(z)
    while len(zg) < len(pg):
        zg.append((0.0, 0.0))

    # most dominant pole pair first
    pg.sort(key=lambda g: -max(abs(g[0]), abs(g[1])))
    sections = []
    for pp in pg:
        ref = pp[0] if abs(pp[0]) >= abs(pp[1]) else pp[1]
        j = int(np.argmin([min(abs(zz[0] - ref), abs(zz[1] - ref)) for zz in zg]))
        zz = zg.pop(j)
        b = np.real(np.poly(zz))
        a = np.real(np.poly(pp))
        sections.append(np.concatenate([b, a]))
    sos = np.array(sections[::-1])                # dominant section last
    sos[0, :3] *= k
    return sos


def sos2zpk(sos):
    """Concatenate the roots of every section (zeros/poles at origin included)."""
    sos = np.atleast_2d(sos)
    z = np.concatenate([np.roots(s[:3]) if s[0] != 0 else np.roots(s[1:3]) for s in sos])
    p = np.concatenate([np.roots(s[3:]) for s in sos])
    k = np.prod(sos[:, 0])
    return z, p, k


def sosfilt(sos, x, zi=None):
    """Cascade of DF2T biquads. zi: (n_sections, 2) state array or None.

    Per section and sample (a0 = 1):
        y  = b0 x + s1
        s1 = b1 x - a1 y + s2
        s2 = b2 x - a2 y
    Output of section i is the input of section i+1.
    Returns (y, zf) with zf of shape (n_sections, 2).
    """
    sos = np.atleast_2d(np.asarray(sos, float))
    ns = len(sos)
    s = np.zeros((ns, 2)) if zi is None else np.array(zi, float)
    y = np.array(x, dtype=float, copy=True)
    for i in range(ns):
        b0, b1, b2, a0, a1, a2 = sos[i] / sos[i, 3]
        s1, s2 = s[i]
        out = np.empty_like(y)
        for n, xn in enumerate(y):
            yn = b0 * xn + s1
            s1 = b1 * xn - a1 * yn + s2
            s2 = b2 * xn - a2 * yn
            out[n] = yn
        s[i] = s1, s2
        y = out
    return y, s
