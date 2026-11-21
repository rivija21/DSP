"""
dspkit.fft -- DFT/FFT algorithms and fast convolution.

  dft             O(N^2) matrix form (reference)
  fft_recursive   radix-2 decimation-in-time, recursive (the textbook derivation)
  fft             radix-2 DIT, iterative, in-place: bit reversal + log2(N) butterfly
                  stages (the structure used in hardware / embedded code)
  ifft            via the conjugation identity
  rfft_packed     N-point real FFT from one N/2-point complex FFT
  goertzel        single DFT bin with a 2nd-order IIR (DTMF detection)
  fft_convolve, overlap_add, overlap_save, stft

DFT definition (numpy sign convention):
    X[k] = sum_{n=0}^{N-1} x[n] e^{-j 2 pi k n / N}
    x[n] = (1/N) sum_k X[k] e^{+j 2 pi k n / N}
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "dft", "fft_recursive", "fft", "ifft", "bit_reverse_indices",
    "rfft_packed", "goertzel", "next_pow2",
    "fft_convolve", "overlap_add", "overlap_save", "stft",
]


def next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(n, 1))))


def _check_pow2(N):
    if N < 1 or (N & (N - 1)):
        raise ValueError(f"length {N} is not a power of 2")


# --------------------------------------------------------------------------- #
def dft(x):
    """Direct DFT: X = F x with F[k, n] = W_N^{kn}, W_N = e^{-j 2pi/N}. O(N^2)."""
    x = np.asarray(x, complex)
    N = len(x)
    n = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(n, n) / N)
    return F @ x


def fft_recursive(x):
    """Radix-2 DIT, recursive.

    Split into even/odd samples:  X[k] = E[k] + W_N^k O[k]
                                  X[k + N/2] = E[k] - W_N^k O[k]     (k < N/2)
    where E, O are the N/2-point DFTs of x[0::2], x[1::2].
    """
    x = np.asarray(x, complex)
    N = len(x)
    _check_pow2(N)
    if N == 1:
        return x.copy()
    E = fft_recursive(x[0::2])
    O = fft_recursive(x[1::2])
    W = np.exp(-2j * np.pi * np.arange(N // 2) / N)       # twiddle factors
    t = W * O
    return np.concatenate([E + t, E - t])


def bit_reverse_indices(N):
    """Permutation p with p[i] = bit-reversal of i over log2(N) bits.
    Built iteratively: rev(2i) = rev(i) >> 1, rev(2i+1) = rev(2i) | N/2."""
    _check_pow2(N)
    bits = N.bit_length() - 1
    rev = np.zeros(N, dtype=int)
    for i in range(1, N):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bits - 1))
    return rev


def fft(x):
    """Iterative in-place radix-2 DIT FFT.

    1. Reorder input in bit-reversed order (DIT input ordering).
    2. For stage s = 1..log2(N): span m = 2^s, half = m/2,
       twiddles W_m^j = e^{-j 2 pi j / m}, j = 0..half-1.
       For each group start g (0, m, 2m, ...) and j:
            a = X[g+j], b = X[g+j+half] * W_m^j
            X[g+j] = a + b ;  X[g+j+half] = a - b          <- one butterfly
    Each stage is vectorised over all groups (reshape to (N/m, m)).
    Operation count: (N/2) log2 N complex multiplies.
    """
    X = np.asarray(x, complex)[bit_reverse_indices(len(x))].copy()
    N = len(X)
    m = 2
    while m <= N:
        half = m // 2
        W = np.exp(-2j * np.pi * np.arange(half) / m)
        Xg = X.reshape(-1, m)                   # rows = butterfly groups (a view)
        a = Xg[:, :half].copy()
        b = Xg[:, half:] * W
        Xg[:, :half] = a + b
        Xg[:, half:] = a - b
        m *= 2
    return X


def ifft(X):
    """x = conj(FFT(conj(X))) / N -- reuse the forward FFT, no separate kernel."""
    X = np.asarray(X, complex)
    return np.conj(fft(np.conj(X))) / len(X)


def rfft_packed(x):
    """FFT of a REAL length-N sequence using one N/2-point complex FFT.

    Pack z[n] = x[2n] + j x[2n+1]; Z = FFT_{N/2}(z). Then
        E[k] = (Z[k] + conj(Z[N/2-k])) / 2           (DFT of even samples)
        O[k] = (Z[k] - conj(Z[N/2-k])) / (2j)        (DFT of odd samples)
        X[k] = E[k] + W_N^k O[k],  k = 0..N/2
    Returns X[0..N/2] (the non-redundant half, like numpy.fft.rfft).
    """
    x = np.asarray(x, float)
    N = len(x)
    _check_pow2(N)
    M = N // 2
    Z = fft(x[0::2] + 1j * x[1::2])
    k = np.arange(M + 1)
    Zk = Z[k % M]
    Zc = np.conj(Z[(-k) % M])
    E = (Zk + Zc) / 2
    O = (Zk - Zc) / 2j
    return E + np.exp(-2j * np.pi * k / N) * O


def goertzel(x, k):
    """Single DFT bin X[k] (k may be non-integer -> any frequency f = k fs / N).

    Equivalent filter: H(z) = 1 / (1 - e^{jw} z^-1), w = 2 pi k / N, so
        y[N-1] = sum_m x[m] e^{jw(N-1-m)} = e^{jw(N-1)} X[k].
    Multiply numerator and denominator by (1 - e^{-jw} z^-1) to get a REAL
    feedback recursion (poles at e^{+-jw}) followed by a one-off complex FIR step:
        s[n]   = x[n] + 2 cos(w) s[n-1] - s[n-2]          (per sample, real)
        y[N-1] = s[N-1] - e^{-jw} s[N-2]                    (once, at the end)
        X[k]   = e^{-jw(N-1)} y[N-1]
    Power only: |X[k]|^2 = s1^2 + s2^2 - 2 cos(w) s1 s2 (no complex arithmetic).

    Cost: 1 real multiply per sample per bin -> cheaper than an FFT when only
    a few bins (< log2 N) are needed, e.g. the 8 DTMF frequencies.
    """
    x = np.asarray(x, float)
    N = len(x)
    w = 2 * np.pi * k / N
    c = 2 * np.cos(w)
    s1 = s2 = 0.0
    for xn in x:
        s0 = xn + c * s1 - s2
        s2, s1 = s1, s0
    y = s1 - np.exp(-1j * w) * s2
    return y * np.exp(-1j * w * (N - 1))


# --------------------------------------------------------------------------- #
# Fast convolution
# --------------------------------------------------------------------------- #
def fft_convolve(x, h):
    """Linear convolution through the DFT.
    Circular convolution of length L equals linear convolution iff
    L >= len(x) + len(h) - 1, so zero-pad both to the next power of 2 >= that."""
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    Ly = len(x) + len(h) - 1
    L = next_pow2(Ly)
    X = fft(np.pad(x, (0, L - len(x))))
    H = fft(np.pad(h, (0, L - len(h))))
    return np.real(ifft(X * H))[:Ly]


def overlap_add(x, h, nfft=None):
    """Block convolution, overlap-add.

    Split x into non-overlapping blocks of length B = nfft - M + 1 (M = len(h)).
    Each block convolved with h gives B + M - 1 = nfft samples (no aliasing);
    consecutive results overlap by M-1 samples and are ADDED.
    """
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    M = len(h)
    nfft = nfft or next_pow2(2 * M)
    B = nfft - M + 1
    if B < 1:
        raise ValueError("nfft must be >= len(h)")
    H = fft(np.pad(h, (0, nfft - M)))
    y = np.zeros(len(x) + M - 1 + nfft)
    for start in range(0, len(x), B):
        blk = x[start:start + B]
        yb = np.real(ifft(fft(np.pad(blk, (0, nfft - len(blk)))) * H))
        y[start:start + nfft] += yb
    return y[:len(x) + M - 1]


def overlap_save(x, h, nfft=None):
    """Block convolution, overlap-save.

    Take INPUT blocks of length nfft that overlap by M-1 samples. The circular
    convolution of each block with h is correct except for its first M-1 samples
    (wrap-around aliasing) -> discard them and keep B = nfft - M + 1 outputs.
    No additions needed; the input buffer is a sliding window (streaming-friendly).
    """
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    M = len(h)
    nfft = nfft or next_pow2(2 * M)
    B = nfft - M + 1
    H = fft(np.pad(h, (0, nfft - M)))
    Ly = len(x) + M - 1
    xp = np.concatenate([np.zeros(M - 1), x, np.zeros(nfft)])   # M-1 zeros of history
    y = []
    for start in range(0, Ly, B):
        blk = xp[start:start + nfft]
        yb = np.real(ifft(fft(blk) * H))
        y.append(yb[M - 1:])                                      # discard aliased part
    return np.concatenate(y)[:Ly]


def stft(x, nperseg=256, hop=None, window="hann", nfft=None):
    """Short-time Fourier transform (frames along axis 0).
    Returns S of shape (n_frames, nfft//2 + 1) and frame start indices."""
    from .fir import get_window
    x = np.asarray(x, float)
    hop = hop or nperseg // 2
    nfft = nfft or next_pow2(nperseg)
    # periodic window (length N+1 symmetric, drop last) for COLA-friendly STFT
    w = get_window(window, nperseg + 1)[:-1]
    starts = np.arange(0, len(x) - nperseg + 1, hop)
    S = np.empty((len(starts), nfft // 2 + 1), complex)
    for i, s in enumerate(starts):
        frame = np.pad(x[s:s + nperseg] * w, (0, nfft - nperseg))
        S[i] = rfft_packed(frame)
    return S, starts
