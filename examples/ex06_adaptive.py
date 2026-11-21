"""
Example 06 -- Adaptive filters.

A: system identification, learning curves of LMS / NLMS / RLS (ensemble of 30 runs),
   with a coloured (lowpass) input -- where plain LMS slows down.
B: adaptive noise cancellation: remove 50 Hz mains hum (+ harmonics) that reached
   the sensor through an unknown path, using a clean hum reference.
"""
import numpy as np

from _common import plt, save
from dspkit import adaptive as ad, iir

rng = np.random.default_rng(0)
h_true = np.array([0.5, -0.4, 0.3, 0.2, -0.1, 0.05, 0.02, -0.01])
M, n = len(h_true), 3000
col = iir.butter(2, 0.2)                          # colouring filter -> eigenvalue spread

curves = {"LMS mu=0.02": [], "NLMS mu=0.5": [], "RLS lam=0.995": []}
for run in range(30):
    x = iir.sosfilt(col, rng.standard_normal(n))[0]
    x /= x.std()
    d = np.convolve(x, h_true)[:n] + 0.01 * rng.standard_normal(n)
    curves["LMS mu=0.02"].append(ad.lms(x, d, M, 0.02)[1] ** 2)
    curves["NLMS mu=0.5"].append(ad.nlms(x, d, M, 0.5)[1] ** 2)
    curves["RLS lam=0.995"].append(ad.rls(x, d, M, 0.995)[1] ** 2)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for lab, c in curves.items():
    mse = np.mean(c, axis=0)
    sm = np.convolve(mse, np.ones(20) / 20, "valid")          # 20-sample moving average
    ax[0].plot(np.arange(len(sm)) + 10, 10 * np.log10(sm), label=lab)
ax[0].axhline(20 * np.log10(0.01), ls="--", c="k", lw=0.8, label="noise floor")
ax[0].set(xlabel="iteration", ylabel="MSE (dB)", title="Learning curves, coloured input (30-run average)")
ax[0].legend(fontsize=7)

# --- B. hum cancellation
fs = 8000
t = np.arange(4 * fs) / fs
s = 0.5 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))   # wanted signal
ref = sum(np.sin(2 * np.pi * 50 * k * t + k) / k for k in (1, 3, 5))           # hum at the mains socket
path = np.array([0.0, 0.8, 0.3, -0.2])                                            # unknown coupling path
d = s + np.convolve(ref, path)[:len(t)]
_, e, w = ad.nlms(ref, d, 16, mu=0.05)
seg = slice(3 * fs, 3 * fs + 400)
ax[1].plot(t[seg] * 1e3, d[seg], label="sensor (signal + hum)", lw=0.9)
ax[1].plot(t[seg] * 1e3, e[seg], label="NLMS output e[n]", lw=1.2)
ax[1].plot(t[seg] * 1e3, s[seg], "--", label="clean signal", lw=0.9)
ax[1].set(xlabel="ms", title="Adaptive noise cancellation (after 3 s)")
ax[1].legend(fontsize=7, loc="lower right")
save(fig, "06_adaptive.png")
err_db = 10 * np.log10(np.mean((e[3 * fs:] - s[3 * fs:]) ** 2) / np.mean((d[3 * fs:] - s[3 * fs:]) ** 2))
print(f"hum power reduced by {-err_db:.1f} dB")
