"""Cross-checks of dspkit.fft, fixed, multirate, adaptive."""
import numpy as np
import pytest
import scipy.signal as ss

from dspkit import fft as F
from dspkit import fixed as fx
from dspkit import multirate as mr
from dspkit import adaptive as ad

rng = np.random.default_rng(1)


# ---------------------------------------------------------------- fft
@pytest.mark.parametrize("N", [1, 2, 8, 64, 1024])
def test_fft_variants(N):
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    X = np.fft.fft(x)
    assert np.allclose(F.fft(x), X)
    assert np.allclose(F.fft_recursive(x), X)
    if N <= 256:
        assert np.allclose(F.dft(x), X)
    assert np.allclose(F.ifft(X), x)


def test_bit_reverse():
    assert list(F.bit_reverse_indices(8)) == [0, 4, 2, 6, 1, 5, 3, 7]


def test_rfft_packed():
    x = rng.standard_normal(512)
    assert np.allclose(F.rfft_packed(x), np.fft.rfft(x))


def test_goertzel():
    x = rng.standard_normal(205)
    for k in (0, 18, 37, 102):
        assert F.goertzel(x, k) == pytest.approx(np.fft.fft(x)[k])


@pytest.mark.parametrize("fn", [F.fft_convolve, F.overlap_add, F.overlap_save])
def test_fast_convolution(fn):
    x, h = rng.standard_normal(1500), rng.standard_normal(101)
    assert np.allclose(fn(x, h), np.convolve(x, h))


# ---------------------------------------------------------------- fixed
def test_quantize_and_ranges():
    q = fx.QFormat(1, 15)
    assert (q.min_code, q.max_code) == (-32768, 32767)
    c = fx.quantize([0.5, -1.0, 1.0, 1e-9], q)
    assert list(c) == [16384, -32768, 32767, 0]           # 1.0 saturates
    assert fx.quantize([1.0], q, overflow="wrap")[0] == -32768


@pytest.mark.parametrize("mode,expect", [
    ("floor", [1, -2, 1, -2]), ("trunc", [1, -1, 1, -1]),
    ("round", [2, -1, 1, -2]), ("convergent", [2, -2, 1, -2])])
def test_round_shift(mode, expect):
    # values 1.5, -1.5, 1.25, -1.75 with 2 fractional bits
    v = np.array([6, -6, 5, -7])
    assert list(fx.round_shift(v, 2, mode)) == expect


def test_fir_fixed_matches_float_within_lsb():
    h = ss.firwin(31, 0.25)
    qh, qx = fx.QFormat(1, 15), fx.QFormat(1, 15)
    hc = fx.quantize(h, qh)
    x = 0.5 * rng.uniform(-1, 1, 2000)
    xc = fx.quantize(x, qx)
    y = fx.fir_fixed(xc, hc, 15, 15, fx.QFormat(1, 15))
    y_ref = ss.lfilter(fx.to_real(hc, 15), 1, fx.to_real(xc, 15))
    assert np.max(np.abs(fx.to_real(y, 15) - y_ref)) <= 2 ** -15 / 2 + 1e-12


def test_fir_fixed_wrap_accumulator_still_exact():
    h = ss.firwin(31, 0.25)
    hc = fx.quantize(h, fx.QFormat(1, 15))
    xc = fx.quantize(rng.uniform(-1, 1, 500), fx.QFormat(1, 15))
    need = fx.fir_acc_bits(hc, 16)
    _, acc_full = fx.fir_fixed(xc, hc, 15, 15, fx.QFormat(1, 15), return_acc=True)
    _, acc_wrap = fx.fir_fixed(xc, hc, 15, 15, fx.QFormat(1, 15), acc_bits=need, return_acc=True)
    assert np.array_equal(acc_full, acc_wrap)


def test_biquad_fixed_tracks_float():
    from dspkit import iir
    sos = iir.butter(2, 0.2)
    x = 0.4 * np.sin(2 * np.pi * 0.02 * np.arange(1000))
    y, _ = fx.sos_fixed(x, sos, fx.QFormat(2, 14), fx.QFormat(1, 15))
    y_ref = ss.sosfilt(sos, x)
    assert fx.sqnr_db(y_ref, y) > 60


def test_hex_export(tmp_path):
    assert fx.to_hex([-1, 1, -32768], 16) == ["ffff", "0001", "8000"]
    p = tmp_path / "t.mem"
    fx.write_mem(p, [1, -2], 16, comment="test")
    assert p.read_text().splitlines()[-2:] == ["0001", "fffe"]


# ---------------------------------------------------------------- multirate
@pytest.mark.parametrize("M", [2, 3, 8])
def test_polyphase_decimate(M):
    x = rng.standard_normal(1001)
    h = mr.antialias_filter(M)
    assert np.allclose(mr.polyphase_decimate(x, h, M), mr.decimate_direct(x, h, M))


@pytest.mark.parametrize("L", [2, 5])
def test_polyphase_interpolate(L):
    x = rng.standard_normal(300)
    h = L * mr.antialias_filter(L)
    assert np.allclose(mr.polyphase_interpolate(x, h, L), mr.interpolate_direct(x, h, L))


def test_resample_poly_tone():
    fs, f0 = 48000, 1000
    x = np.sin(2 * np.pi * f0 * np.arange(4800) / fs)
    y = mr.resample_poly(x, 2, 3)                            # 48k -> 32k
    assert len(y) == pytest.approx(len(x) * 2 / 3, abs=50)
    # unity passband gain in steady state
    assert np.max(np.abs(y[200:-200])) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("R,N,M", [(8, 3, 1), (16, 4, 1), (5, 2, 2)])
def test_cic_decimate_bit_exact(R, N, M):
    x = rng.integers(-2 ** 15, 2 ** 15, 4000)
    y = mr.cic_decimate(x, R, N, M, in_bits=16)
    ref = np.convolve(x, mr.cic_impulse(R, N, M).astype(np.int64))[: len(x)][::R]
    assert np.array_equal(y, ref)


def test_cic_decimate_dc_gain():
    R, N = 16, 4
    y = mr.cic_decimate(np.full(4000, 2 ** 15 - 1), R, N)
    assert y[-1] == (2 ** 15 - 1) * R ** N                 # wrapped integrators, exact output


def test_cic_interpolate_bit_exact():
    R, N = 8, 3
    x = rng.integers(-2 ** 15, 2 ** 15, 500)
    y = mr.cic_interpolate(x, R, N, in_bits=16)
    ref = np.convolve(mr.upsample(x, R), mr.cic_impulse(R, N).astype(np.int64))[: len(x) * R]
    assert np.array_equal(y, ref)


def test_cic_compensator_flattens_passband():
    R, N = 16, 4
    h = mr.cic_compensator(31, R, N, f_pass=0.4, f_stop=0.7)
    f = np.linspace(0, 0.4, 100)
    Hc = np.abs(np.exp(-1j * np.pi * np.outer(f, np.arange(len(h)))) @ h)
    total = Hc * mr.cic_response(f / (2 * R), R, N)
    assert np.ptp(20 * np.log10(total)) < 0.1             # < 0.1 dB passband ripple


# ---------------------------------------------------------------- adaptive
def _plant_setup(n=4000):
    h = np.array([0.6, -0.3, 0.2, 0.1, -0.05])
    x = rng.standard_normal(n)
    d = np.convolve(x, h)[:n] + 1e-3 * rng.standard_normal(n)
    return h, x, d


@pytest.mark.parametrize("algo,kw", [(ad.lms, dict(mu=0.01)), (ad.nlms, dict(mu=0.5)),
                                     (ad.rls, dict(lam=0.999))])
def test_system_identification(algo, kw):
    h, x, d = _plant_setup()
    _, e, w = algo(x, d, 5, **kw)
    assert np.allclose(w, h, atol=5e-3)


def test_regressor_matrix():
    U = ad.regressor_matrix([1, 2, 3], 2)
    assert U.tolist() == [[1, 0], [2, 1], [3, 2]]
