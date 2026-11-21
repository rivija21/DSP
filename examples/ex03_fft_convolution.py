"""
Example 03 -- FFT algorithms, Goertzel DTMF detection, block convolution.

A: correctness and operation count/timing of DFT vs radix-2 FFT vs numpy.
B: Goertzel bank detecting a DTMF digit in noise (8 bins instead of an FFT).
C: overlap-add / overlap-save streaming convolution of a long signal.
"""
import time

import numpy as np

from _common import plt, save
from dspkit import fft as F
from dspkit import fir

rng = np.random.default_rng(0)

# --- A. timing
Ns = [64, 128, 256, 512, 1024, 2048, 4096]
t_dft, t_rec, t_it, t_np = [], [], [], []
for N in Ns:
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    for fn, store, reps in [(F.dft, t_dft, 2), (F.fft_recursive, t_rec, 3), (F.fft, t_it, 20),
                            (np.fft.fft, t_np, 200)]:
        if fn is F.dft and N > 2048:
            store.append(np.nan)
            continue
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(x)
        store.append((time.perf_counter() - t0) / reps)
    assert np.allclose(F.fft(x), np.fft.fft(x))
fig, ax = plt.subplots(figsize=(5, 3.4))
for t, lab in [(t_dft, "dft  O(N^2)"), (t_rec, "fft_recursive"), (t_it, "fft (iterative, vectorised stages)"),
               (t_np, "numpy.fft (C, pocketfft)")]:
    ax.loglog(Ns, t, "o-", ms=3, label=lab)
ax.set(xlabel="N", ylabel="seconds per transform", title="Same O(N log N), very different constants")
ax.legend(fontsize=7)
save(fig, "03_fft_timing.png")

# --- B. DTMF with Goertzel
fs, N = 8000, 205                        # classic choice: bins land near the 8 tones
low = [697, 770, 852, 941]
high = [1209, 1336, 1477, 1633]
keys = ["123A", "456B", "789C", "*0#D"]
digit = "7"
r = [i for i, row in enumerate(keys) if digit in row][0]
c = keys[r].index(digit)
n = np.arange(N)
x = np.sin(2 * np.pi * low[r] * n / fs) + np.sin(2 * np.pi * high[c] * n / fs) + 0.5 * rng.standard_normal(N)
freqs = low + high
P = [abs(F.goertzel(x, f * N / fs)) ** 2 for f in freqs]    # non-integer k allowed
det_r = int(np.argmax(P[:4]))
det_c = int(np.argmax(P[4:]))
print("DTMF detected:", keys[det_r][det_c], "(sent", digit + ")")
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar([str(f) for f in freqs], P, color=["#1f2a44"] * 4 + ["#c0392b"] * 4)
ax.set(title=f"Goertzel bank, N={N}: digit '{keys[det_r][det_c]}'", ylabel="|X|^2", xlabel="Hz")
ax.tick_params(axis="x", labelrotation=45)
save(fig, "03_dtmf_goertzel.png")

# --- C. Block convolution: identical output, different memory/latency behaviour
h = fir.firwin(129, 0.1)
x = rng.standard_normal(20000)
y_ref = np.convolve(x, h)
for fn in (F.overlap_add, F.overlap_save):
    y = fn(x, h, nfft=512)
    print(f"{fn.__name__:13s} nfft=512  max err = {np.max(np.abs(y - y_ref)):.2e}")
# multiplies per output sample: direct M vs FFT-based ~ (2 * (N/2) log2 N + N) / B (complex)
M = len(h)
for nfft in (256, 512, 1024, 4096):
    B = nfft - M + 1
    cm = (2 * (nfft / 2) * np.log2(nfft) + nfft) / B
    print(f"nfft={nfft:5d}: block {B:5d}, ~{4 * cm:6.1f} real mult/sample vs direct {M}")
