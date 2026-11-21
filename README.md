# dspkit: Stage 1 DSP reference library

A pure-NumPy library for designing and implementing filters from first principles. Every block is written so you can read it and port it to C or HDL, and each one is checked against `scipy.signal` / `numpy.fft` in `tests/`.

```
dsp-stage1/
├── dspkit/
│   ├── core.py        lfilter (DF2T), fir_filter, convolve, freqz, group_delay, tf<->zpk
│   ├── fir.py         windows, firwin, kaiserord, freq_sampling, firls, remez
│   ├── iir.py         buttap/cheb1ap/cheb2ap, lp2xx, bilinear, zpk2sos, sosfilt, buttord
│   ├── fft.py         dft, fft_recursive, fft (iterative), rfft_packed, goertzel, OLA/OLS, stft
│   ├── fixed.py       QFormat, quantize, round_shift, fir_fixed, biquad_fixed, write_mem
│   ├── multirate.py   polyphase decimate/interpolate, resample_poly, CIC, CIC compensator
│   └── adaptive.py    lms, nlms, rls
├── tests/             76 cross-checks against SciPy / NumPy
├── examples/          ex01..ex06 scripts -> examples/figures/*.png, *.mem golden vectors
├── EXERCISES.md       re-implementation tasks with acceptance tests
└── run_examples.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # 76 tests, ~2 s
python run_examples.py      # regenerates all figures and .mem files
```

## Conventions (used by every module)

| Item | Convention |
|---|---|
| Transfer function | `b`, `a` in ascending powers of z⁻¹; `a[0]` normalised to 1 |
| Frequency units | explicit `fs`; default `fs=2.0`, so frequencies are fractions of Nyquist |
| SOS | array `(n_sections, 6)`, rows `[b0 b1 b2 1 a1 a2]`, highest-Q section last |
| zpk | `(z, p, k)`: complex root arrays and a real gain |
| Filter state | `zi`/`zf` = DF2T delay-line contents; pass `zf` back as `zi` to stream in blocks |
| Fixed point | `QFormat(m, n)`: m integer bits **including sign**, n fractional bits; data held as int64 codes |
| Linear-phase designers | `firls` / `remez` are Type I (odd number of taps) |

## Reading order

1. `core.lfilter`: the DF2T loop is the template for every recursive structure.
2. `fir.firwin`, then `fir.kaiserord`, then `fir.firls`, then `fir.remez`.
3. `iir.buttap`, then `iir.bilinear_zpk`, then `iir.iirfilter`, then `iir.zpk2sos`, then `iir.sosfilt`.
4. `fft.fft_recursive`, then `fft.fft`, then `fft.goertzel`, then `fft.overlap_save`.
5. `fixed`: all of it. This is the model you will use as the reference for the FPGA work in Stage 3.
6. `multirate.polyphase_decimate`, then `multirate.cic_decimate`.
7. `adaptive.lms`, then `nlms`, then `rls`.

Each module docstring states the maths it implements, and every function docstring gives the exact recurrence.
