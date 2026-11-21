"""
Example 05 -- Multirate: aliasing, polyphase efficiency, CIC + compensator.

A: decimate by 4 with and without anti-alias filtering (spectra).
B: polyphase decimator == filter-then-discard, at 1/M of the multiplies.
C: CIC (R=16, N=4) response, passband droop, compensating FIR at the output rate.
"""
import time

import numpy as np

from _common import plt, save
from dspkit import core, fft as F
from dspkit import multirate as mr

rng = np.random.default_rng(0)

# --- A. aliasing
fs, M = 48_000, 4
n = np.arange(8192)
x = np.sin(2 * np.pi * 1_000 * n / fs) + 0.5 * np.sin(2 * np.pi * 15_000 * n / fs)   # 15 kHz > 6 kHz (new Nyquist)
h = mr.antialias_filter(M, 129)
y_bad = mr.downsample(x, M)
y_good = mr.polyphase_decimate(x, h, M)
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for a, sig, rate, title in [(ax[0], x, fs, "input @ 48 kHz"),
                            (ax[1], y_bad, fs / M, "downsample only @ 12 kHz"),
                            (ax[2], y_good[100:2148], fs / M, "anti-alias + downsample")]:
    N = F.next_pow2(len(sig)) // 2
    S = np.abs(F.rfft_packed(sig[:N] * np.hanning(N)))
    a.plot(np.arange(len(S)) * rate / N / 1e3, 20 * np.log10(S / S.max() + 1e-12))
    a.set(title=title, xlabel="kHz", ylim=(-100, 5))
ax[0].set_ylabel("dB")
ax[1].annotate("15 kHz aliased to 3 kHz", (3, -6), (3.5, -40), fontsize=7, arrowprops=dict(arrowstyle="->"))
save(fig, "05_aliasing.png")

# --- B. polyphase cost
M = 8
h = mr.antialias_filter(M, 16 * M + 1)
x = rng.standard_normal(40_000)
t0 = time.perf_counter(); y1 = mr.decimate_direct(x, h, M); t1 = time.perf_counter()
y2 = mr.polyphase_decimate(x, h, M); t2 = time.perf_counter()
print(f"max |direct - polyphase| = {np.max(np.abs(y1 - y2)):.2e}")
print(f"multiplies per OUTPUT sample: direct {M * len(h)}, polyphase {len(h)}")
print(f"python time: direct {1e3 * (t1 - t0):.1f} ms, polyphase {1e3 * (t2 - t1):.1f} ms")

# --- C. CIC + compensator
R, N = 16, 4
f_in = np.linspace(0, 0.5, 4000)                        # cycles per input sample
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(f_in * R, 20 * np.log10(mr.cic_response(f_in, R, N) + 1e-12))
for k in range(1, 9):
    ax[0].axvspan(k - 0.1, k + 0.1, color="0.88")
ax[0].set(xlabel="frequency (multiples of output rate fs/R)", ylabel="dB", ylim=(-140, 5),
          title=f"CIC R={R}, N={N}: nulls at k fs/R (shaded = alias bands)")
h_c = mr.cic_compensator(31, R, N, f_pass=0.4, f_stop=0.7)
f_out = np.linspace(0, 1, 2000)                          # output Nyquist units
_, Hc = core.freqz(h_c, 1, f_out)
Hcic = mr.cic_response(f_out / (2 * R), R, N)
ax[1].plot(f_out, 20 * np.log10(Hcic), label="CIC droop")
ax[1].plot(f_out, 20 * np.log10(np.abs(Hc)), label="31-tap compensator")
ax[1].plot(f_out, 20 * np.log10(np.abs(Hc) * Hcic + 1e-12), lw=2, label="cascade")
ax[1].axvline(0.4, ls=":", c="k")
ax[1].set(xlabel="f / output Nyquist", ylim=(-60, 10), title="Compensation at the output rate")
ax[1].legend(fontsize=7)
save(fig, "05_cic.png")
pb = 20 * np.log10(np.abs(Hc[f_out <= 0.4]) * Hcic[f_out <= 0.4])
print(f"passband flatness after compensation: {np.ptp(pb):.3f} dB (CIC alone {-20*np.log10(Hcic[f_out<=0.4].min()):.2f} dB droop)")
print("CIC register width for 16-bit input:", mr.cic_register_width(16, R, N), "bits")

# bit-true run on integer data
xi = (2 ** 14 * np.sin(2 * np.pi * 0.002 * np.arange(8000))).astype(np.int64)
yi = mr.cic_decimate(xi, R, N, in_bits=16)
print("CIC output peak / (input peak * R^N) =", round(np.max(np.abs(yi[20:])) / (2 ** 14 * R ** N), 4))
