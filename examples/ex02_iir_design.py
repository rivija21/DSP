"""
Example 02 -- IIR design: prototypes, pole-zero maps, group delay, and why SOS.

Part A: Butterworth vs Chebyshev I vs Chebyshev II, same order and edge.
Part B: an 8th-order narrow bandpass realised as ONE polynomial (b, a) vs as
        4 biquads, with coefficients rounded to 16 fractional bits. The direct
        form loses its passband (poles move); the SOS cascade does not.
"""
import numpy as np

from _common import plt, save
from dspkit import core, iir

N, fc = 5, 0.25
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
designs = {
    "Butterworth": iir.butter(N, fc, output="zpk"),
    "Chebyshev I (1 dB)": iir.cheby1(N, 1, fc, output="zpk"),
    "Chebyshev II (40 dB)": iir.cheby2(N, 40, fc, output="zpk"),
}
for (name, (z, p, k)), mk in zip(designs.items(), ["o", "s", "^"]):
    sos = iir.zpk2sos(z, p, k)
    f, H = core.sosfreqz(sos, 2048)
    ax[0].plot(f, core.mag_db(H), label=name)
    b, a = core.zpk2tf(z, p, k)
    f, g = core.group_delay(b, a, 2048)
    m = f < 0.97                        # Butterworth zeros at z=-1 make tau 0/0 at Nyquist
    ax[1].plot(f[m], g[m], label=name)
    ax[2].plot(p.real, p.imag, "x", ms=7, label=f"{name} poles")
    ax[2].plot(z.real, z.imag, mk, mfc="none", ms=6)
t = np.linspace(0, 2 * np.pi, 400)
ax[2].plot(np.cos(t), np.sin(t), "k", lw=0.6)
ax[2].set(aspect="equal", title="Pole-zero map (x poles, o/s/^ zeros)", xlim=(-1.3, 1.3), ylim=(-1.3, 1.3))
ax[2].legend(fontsize=6, loc="lower left")
ax[0].set(title=f"Magnitude, order {N}, fc = {fc}", xlabel="f / Nyquist", ylabel="dB", ylim=(-80, 3))
ax[0].legend(fontsize=7)
ax[1].set(title="Group delay (samples)", xlabel="f / Nyquist", ylim=(0, 30))
save(fig, "02_prototypes.png")

# --- Part B: coefficient sensitivity
Wn = [0.10, 0.12]
z, p, k = iir.butter(4, Wn, "bandpass", output="zpk")        # order 8
b, a = core.zpk2tf(z, p, k)
sos = iir.zpk2sos(z, p, k)
q = 2.0 ** -16


def rnd(c):
    return np.round(np.asarray(c) / q) * q


# Round only the FEEDBACK (a) coefficients -- they decide pole positions.
# (Rounding b as well would zero it: k ~ 1e-8 here, a separate scaling lesson.)
b_q, a_q = b, rnd(a)
sos_q = sos.copy()
sos_q[:, 3:] = rnd(sos[:, 3:])
p_q = np.roots(a_q)
print("max |pole| ideal      :", np.max(np.abs(p)))
print("max |pole| direct  Q16:", np.max(np.abs(p_q)), "(>1 = unstable)" if np.max(np.abs(p_q)) >= 1 else "")
print("max |pole| SOS     Q16:", max(np.max(np.abs(np.roots(s[3:]))) for s in sos_q))
print("dynamic range of a[]  :", np.max(np.abs(a)), "vs smallest", np.min(np.abs(a)))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
f, H = core.sosfreqz(sos, 4096)
ax[0].plot(f, core.mag_db(H), label="ideal")
f, H = core.freqz(b_q, a_q, 4096)
ax[0].plot(f, core.mag_db(H), label="direct form, a[] rounded to 2^-16")
f, H = core.sosfreqz(sos_q, 4096)
ax[0].plot(f[::60], core.mag_db(H)[::60], "o", ms=3, mfc="none", label="SOS, a1/a2 rounded to 2^-16")
# NB: with a pole outside |z|=1 the DTFT curve is not what the filter does -- its
# impulse response diverges. The curve only shows how far the design moved.
ax[0].set(xlim=(0, 0.3), ylim=(-80, 10), xlabel="f / Nyquist", ylabel="dB", title="8th-order bandpass 0.10-0.12")
ax[0].legend(fontsize=7)
ax[1].plot(p.real, p.imag, "kx", label="ideal poles")
ax[1].plot(p_q.real, p_q.imag, "o", mfc="none", label="direct-form quantised")
ax[1].plot(np.cos(t), np.sin(t), "k", lw=0.6)
ax[1].set(aspect="equal", xlim=(0.6, 1.1), ylim=(0, 0.6), title="Pole displacement (zoom)")
ax[1].legend(fontsize=7)
save(fig, "02_sos_vs_direct.png")

# --- Bilinear warping: with fs' = 2, digital w = pi f maps to analog W = 2 fs' tan(w/2)
fd = np.linspace(0.01, 0.95, 200)
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.plot(fd, 4 * np.tan(np.pi * fd / 2), label="bilinear: W = 4 tan(pi f / 2)")
ax.plot(fd, 2 * np.pi * fd, "--", label="no warping: W = w fs' = 2 pi f")
ax.set(xlabel="digital f / Nyquist", ylabel="analog W (rad/s)", ylim=(0, 20), title="Bilinear frequency warping")
ax.legend(fontsize=7)
save(fig, "02_warping.png")
