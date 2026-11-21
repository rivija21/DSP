"""Cross-checks of dspkit.core, dspkit.fir, dspkit.iir against scipy.signal."""
import numpy as np
import pytest
import scipy.signal as ss

from dspkit import core, fir, iir

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- core
def test_convolve():
    x, h = rng.standard_normal(300), rng.standard_normal(17)
    assert np.allclose(core.convolve(x, h), np.convolve(x, h))


def test_lfilter_and_streaming():
    b, a = ss.butter(6, 0.25)
    x = rng.standard_normal(800)
    y_ref = ss.lfilter(b, a, x)
    y, _ = core.lfilter(b, a, x)
    assert np.allclose(y, y_ref, atol=1e-10)
    y1, z = core.lfilter(b, a, x[:333])
    y2, _ = core.lfilter(b, a, x[333:], z)
    assert np.allclose(np.r_[y1, y2], y_ref, atol=1e-10)


def test_fir_filter_streaming():
    h = ss.firwin(41, 0.2)
    x = rng.standard_normal(500)
    y1, z = core.fir_filter(h, x[:123])
    y2, _ = core.fir_filter(h, x[123:], z)
    assert np.allclose(np.r_[y1, y2], ss.lfilter(h, 1, x))


def test_freqz_and_group_delay():
    b, a = ss.cheby1(5, 1, 0.3)
    _, H = core.freqz(b, a, 1024)
    _, Hs = ss.freqz(b, a, 1024, fs=2)
    assert np.allclose(H, Hs)
    f, g = core.group_delay(b, a, 1024)
    _, gs = ss.group_delay((b, a), 1024, fs=2)
    m = f < 0.9                       # avoid the zeros at z = -1 (0/0 at Nyquist)
    assert np.allclose(g[m], gs[m], atol=1e-6)


def test_tf_zpk_roundtrip():
    b, a = ss.butter(4, [0.2, 0.4], "bandpass")
    z, p, k = core.tf2zpk(b, a)
    b2, a2 = core.zpk2tf(z, p, k)
    assert np.allclose(b, b2, atol=1e-12) and np.allclose(a, a2, atol=1e-12)


# ---------------------------------------------------------------- fir
@pytest.mark.parametrize("name,sname", [("hann", "hann"), ("hamming", "hamming"),
                                        ("blackman", "blackman"), ("rect", "boxcar")])
def test_windows(name, sname):
    assert np.allclose(fir.get_window(name, 33), ss.get_window(sname, 33, fftbins=False))


def test_kaiser_window_and_order():
    assert np.allclose(fir.get_window("kaiser", 45, beta=6.5), ss.windows.kaiser(45, 6.5))
    assert fir.kaiserord(65, 0.04) == pytest.approx(ss.kaiserord(65, 0.04))


@pytest.mark.parametrize("cut,pz", [(0.3, True), (0.3, False), ([0.2, 0.5], False),
                                    ([0.2, 0.5], True), ([0.1, 0.3, 0.6], True)])
def test_firwin(cut, pz):
    assert np.allclose(fir.firwin(61, cut, pass_zero=pz), ss.firwin(61, cut, pass_zero=pz))


def test_freq_sampling_interpolates_samples():
    A = np.r_[np.ones(6), 0.4, np.zeros(9)]          # L = 15 -> N = 31
    h = fir.freq_sampling(A)
    N = len(h)
    Hk = np.fft.fft(h)
    amp = np.real(Hk * np.exp(1j * 2 * np.pi * np.arange(N) * (N - 1) / 2 / N))
    assert np.allclose(amp[:len(A)], A, atol=1e-12)
    assert np.allclose(h, h[::-1])                   # linear phase


def test_firls():
    args = (61, [0, 0.3, 0.4, 1], [1, 1, 0, 0])
    assert np.allclose(fir.firls(*args, weight=[1, 10]), ss.firls(*args, weight=[1, 10]), atol=2e-4)


@pytest.mark.parametrize("N,bands,des,wt", [
    (51, [0, 0.3, 0.4, 1], [1, 0], [1, 1]),
    (41, [0, 0.2, 0.3, 1], [1, 0], [1, 10]),
    (61, [0, 0.2, 0.3, 0.5, 0.6, 1], [0, 1, 0], [1, 1, 1]),
    (31, [0, 0.4, 0.5, 1], [0, 1], [1, 1]),
])
def test_remez_matches_scipy(N, bands, des, wt):
    h = fir.remez(N, bands, des, wt)
    assert np.allclose(h, ss.remez(N, bands, des, weight=wt, fs=2), atol=1e-4)


def test_remez_is_equiripple():
    h, info = fir.remez(51, [0, 0.3, 0.4, 1], [1, 0], return_info=True)
    f, H = core.freqz(h, 1, 8192)
    stop = np.abs(H[f >= 0.4]).max()
    assert stop == pytest.approx(info["delta"], rel=1e-2)


# ---------------------------------------------------------------- iir
@pytest.mark.parametrize("ftype", ["butter", "cheby1", "cheby2"])
@pytest.mark.parametrize("N,Wn,bt", [(4, 0.3, "lowpass"), (5, 0.3, "highpass"),
                                     (3, [0.2, 0.4], "bandpass"), (4, [0.2, 0.4], "bandstop"),
                                     (8, [0.1, 0.12], "bandpass")])
def test_iir_designs(ftype, N, Wn, bt):
    kw = dict(rp=1, rs=40, btype=bt, ftype=ftype)
    z, p, k = iir.iirfilter(N, Wn, output="zpk", **kw)
    zs, ps, ks = ss.iirfilter(N, Wn, output="zpk", **kw)
    assert k == pytest.approx(ks, rel=1e-9)
    sos = iir.iirfilter(N, Wn, **kw)
    soss = ss.iirfilter(N, Wn, output="sos", **kw)
    _, H = core.sosfreqz(sos, 2048)
    _, Hs = ss.sosfreqz(soss, 2048)
    assert np.allclose(H, Hs, atol=1e-9)
    x = rng.standard_normal(400)
    assert np.allclose(iir.sosfilt(sos, x)[0], ss.sosfilt(soss, x), atol=1e-9)


def _same_roots(r1, r2):
    """Order-independent root comparison (rounding-robust sort key)."""
    key = lambda r: sorted(np.round(np.asarray(r, complex), 9), key=lambda c: (c.real, c.imag))
    return np.allclose(key(r1), key(r2))


def test_prototypes():
    for N in (3, 4, 7):
        assert _same_roots(iir.buttap(N)[1], ss.buttap(N)[1])
        assert _same_roots(iir.cheb1ap(N, 0.5)[1], ss.cheb1ap(N, 0.5)[1])
        assert _same_roots(iir.cheb2ap(N, 50)[0], ss.cheb2ap(N, 50)[0])
        assert _same_roots(iir.cheb2ap(N, 50)[1], ss.cheb2ap(N, 50)[1])


def test_buttord():
    N, Wn = iir.buttord(0.2, 0.3, 1, 40)
    Ns, Wns = ss.buttord(0.2, 0.3, 1, 40)
    assert N == Ns and Wn == pytest.approx(Wns)


def test_sos_stream():
    sos = iir.butter(6, 0.2)
    x = rng.standard_normal(300)
    y1, z = iir.sosfilt(sos, x[:100])
    y2, _ = iir.sosfilt(sos, x[100:], z)
    assert np.allclose(np.r_[y1, y2], ss.sosfilt(sos, x))
