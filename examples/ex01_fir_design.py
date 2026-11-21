"""
Example 01 -- FIR design: window method vs Kaiser vs least squares vs Parks-McClellan.

Spec (fs = 48 kHz): passband 0..6 kHz, stopband 8..24 kHz, >= 60 dB attenuation.
Question answered: how many taps does each method need for the same spec?
"""
import numpy as np

from _common import plt, save
from dspkit import core, fir

fs = 48_000
fp, fst, A = 6_000, 8_000, 60
fc = (fp + fst) / 2

# --- 1. Window method: same length, different windows (shows the trade-off)
N = 81
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for w in ["rect", "hann", "hamming", "blackman"]:
    h = fir.firwin(N, fc, window=w, fs=fs)
    f, H = core.freqz(h, 1, 4096, fs=fs)
    ax[0].plot(f / 1e3, core.mag_db(H), label=w)
ax[0].set(title=f"Windowed sinc, N={N}", xlabel="kHz", ylabel="dB", ylim=(-120, 5))
ax[0].legend()
for w in ["rect", "hann", "hamming", "blackman"]:
    ax[1].plot(fir.get_window(w, N), label=w)
ax[1].set(title="Window shapes", xlabel="n")
save(fig, "01_windows.png")

# --- 2. Kaiser: order from formula
numtaps, beta = fir.kaiserord(A, (fst - fp), fs=fs)
if numtaps % 2 == 0:
    numtaps += 1
h_kai = fir.firwin(numtaps, fc, window="kaiser", beta=beta, fs=fs)

# --- 3./4. LS and Remez: search for the minimum odd length meeting the spec
def meets(h):
    f, H = core.freqz(h, 1, 8192, fs=fs)
    m = core.mag_db(H)
    pr = np.max(np.abs(m[f <= fp]))
    return np.max(m[f >= fst]) <= -A and pr <= 0.1, pr


def min_taps(design):
    for n in range(11, 401, 2):
        h = design(n)
        ok, _ = meets(h)
        if ok:
            return n, h
    raise RuntimeError


n_ls, h_ls = min_taps(lambda n: fir.firls(n, [0, fp, fst, fs / 2], [1, 1, 0, 0], [1, 30], fs=fs))
n_pm, h_pm = min_taps(lambda n: fir.remez(n, [0, fp, fst, fs / 2], [1, 0], [1, 30], fs=fs))

f, H = core.freqz(h_kai, 1, 8192, fs=fs)
att = -np.max(core.mag_db(H)[f >= fst])
# Kaiser's formula is empirical: always verify, bump N if it falls short.
print(f"Kaiser   : {numtaps} taps (beta={beta:.2f})  achieved {att:.1f} dB")
print(f"LS       : {n_ls} taps")
print(f"Remez    : {n_pm} taps")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for h, lab in [(h_kai, f"Kaiser N={numtaps}"), (h_ls, f"LS N={n_ls}"), (h_pm, f"Remez N={n_pm}")]:
    f, H = core.freqz(h, 1, 8192, fs=fs)
    ax[0].plot(f / 1e3, core.mag_db(H), label=lab)
    ax[1].plot(f / 1e3, core.mag_db(H), label=lab)
for a in ax:
    a.axvspan(fp / 1e3, fst / 1e3, color="0.9")
ax[0].axhline(-A, ls="--", c="k", lw=0.8)
ax[0].set(title="Same spec: Kaiser vs LS vs Remez", xlabel="kHz", ylabel="dB", ylim=(-110, 5))
ax[0].legend()
ax[1].set(title="Passband zoom", xlim=(0, 7), ylim=(-0.12, 0.12), xlabel="kHz")
save(fig, "01_methods.png")

# --- 5. Frequency sampling with and without a transition sample
L = 20
A0 = np.r_[np.ones(6), np.zeros(L + 1 - 6)]
A1 = A0.copy()
A1[6] = 0.39                                # one optimised transition sample
fig, ax = plt.subplots(figsize=(5, 3.2))
for Ak, lab in [(A0, "no transition sample"), (A1, "T1 = 0.39")]:
    h = fir.freq_sampling(Ak)
    f, H = core.freqz(h, 1, 4096)
    ax.plot(f, core.mag_db(H), label=lab)
ax.plot(2 * np.arange(L + 1) / (2 * L + 1), 20 * np.log10(np.maximum(A1, 1e-6)), "ko", ms=3, label="samples")
ax.set(title="Frequency sampling, N = 41", xlabel="f / Nyquist", ylabel="dB", ylim=(-90, 5))
ax.legend()
save(fig, "01_freq_sampling.png")

# --- group delay of a linear-phase FIR is constant (N-1)/2
f, g = core.group_delay(h_pm, 1, 512, fs=fs)
print(f"Remez group delay = {np.median(g):.1f} samples (expected {(n_pm - 1) / 2})")
