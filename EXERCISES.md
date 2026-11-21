# Exercises: re-implement each block yourself

Rule: write your version in `mine/<module>.py` **without looking at dspkit**. Then check it with the acceptance test, which compares your output against `scipy.signal` or `dspkit`. Each exercise has a level from ★ (an evening) to ★★★ (a weekend).

## FIR

1. ★ **Windowed-sinc bandpass.** `bp(numtaps, f1, f2, window)`. Pass when `np.allclose(bp(...), scipy.signal.firwin(numtaps, [f1, f2], pass_zero=False, window=window))` holds.
2. ★ **Kaiser auto-design.** Given (passband ripple δp, stopband attenuation A, edges), return taps that **meet** the spec. Start from the Kaiser formula and increase N until the spec is met. Check by measuring with `freqz`.
3. ★★ **Frequency sampling with an optimised transition sample.** Scan T₁ ∈ [0, 1] and plot the minimum stopband attenuation against T₁. You should find the optimum near 0.39 for a ~41-tap lowpass.
4. ★★ **Type II least-squares FIR.** Extend `firls_grid` to even N. Hint: A(ω) = Σ bₖ cos((k−½)ω). Pass when the result matches `scipy.signal.firls` for even numtaps.
5. ★★★ **Remez exchange.** Implement it from the docstring algorithm only. Pass when the taps match `scipy.signal.remez` to 1e-4 and the stopband peak equals δ to within 1 %.
6. ★★ **Half-band filter.** Design one with Remez and confirm that every second tap is 0, except the centre tap, which is 0.5. Explain why this halves the number of multipliers.

## IIR

7. ★ **Butterworth prototype poles** from the formula, then `lp2lp` and `bilinear`. Pass when the result matches `scipy.signal.butter(N, Wn, output='zpk')`.
8. ★★ **Elliptic prototype** (`ellipap`), using Jacobi elliptic functions (`scipy.special.ellipj` is allowed). Compare against `scipy.signal.ellipap`.
9. ★★ **Biquad cookbook (RBJ).** Implement peaking-EQ, low-shelf and notch biquads from analog prototypes plus a prewarped bilinear transform. You will reuse these in Stage 2 for the live equaliser.
10. ★★ **Pairing experiment.** Order the sections of a 10th-order Chebyshev I filter in two ways: highest-Q first, then highest-Q last. Run both in `fixed.sos_fixed` with Q1.15 I/O and compare peak internal levels and SQNR.
11. ★ **Minimum-phase check.** Write `is_minimum_phase(b)` using the zeros. Then convert a linear-phase FIR to minimum phase by reflecting its zeros inside the unit circle, and compare group delays.

## FFT

12. ★ **Radix-2 DIF** (decimation in frequency): natural-order input, bit-reversed output. Pass when the result matches `np.fft.fft`.
13. ★★ **Radix-4 FFT** for N = 4ᵏ. Count the real multiplications and compare with radix-2.
14. ★★ **Bluestein (chirp-z)** for arbitrary N, built on your radix-2 FFT. Pass when the result matches `np.fft.fft` for N = 1000.
15. ★ **Fixed-point FFT.** Run your iterative FFT on Q1.15 data, shifting right by 1 after each stage to scale. Measure SQNR against N and explain the ~3 dB loss per stage.
16. ★ **Goertzel DTMF decoder** for a whole tone sequence (frames + twist check + validity rules).

## Fixed point

17. ★ **Q-format calculator.** Given an FIR's taps and its input format, print the exact accumulator width, the guard bits, and the output shift needed to land in Q1.15.
18. ★★ **Error-feedback biquad.** Add first-order noise shaping (feed back the rounding error) to `biquad_fixed`. Show that the limit cycle from `ex04` disappears.
19. ★★ **Golden-vector generator.** Write a script that produces `.mem` files of taps, inputs and expected outputs for any FIR, together with a cocotb or SystemVerilog testbench skeleton that reads them. This is the entry point to Stage 3.

## Multirate

20. ★ **Efficient rational resampler.** Compute only the outputs that are kept (the polyphase index advances by M mod L). Pass when the result matches `dspkit.multirate.resample_poly`.
21. ★★ **CIC interpolator + compensator** (pre-compensation before the CIC). Verify the passband is flat at the output.
22. ★★ **Hogenauer register pruning.** Compute how many LSBs each CIC stage can discard for a given output word length. Then check bit-true behaviour against the unpruned model: the error must stay within the predicted bound.
23. ★★ **Two-stage decimate-by-64.** Compare one FIR, CIC(16) + FIR(4), and half-band cascades in multiplies per input sample.

## Adaptive

24. ★ **Choosing μ for LMS.** Measure the input power and plot convergence time against μ. Find the divergence point empirically and compare it with 2/(M·Pₓ).
25. ★★ **Adaptive line enhancer:** recover a sinusoid buried in white noise with LMS on a delayed copy of the input.
26. ★★ **Sign-error LMS (sign-data LMS)**, the multiplier-free variant used in hardware. Compare it with LMS on the same task.
