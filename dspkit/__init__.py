"""
dspkit -- Stage 1 DSP reference library (pure NumPy, designs from first principles).

Modules
    core       lfilter (DF2T), fir_filter, convolve, freqz, group_delay, tf<->zpk
    fir        windows, firwin, kaiserord, freq_sampling, firls, remez (Parks-McClellan)
    iir        analog prototypes, frequency transforms, bilinear, zpk2sos, sosfilt
    fft        dft, radix-2 fft (recursive/iterative), rfft trick, goertzel, OLA/OLS, stft
    fixed      Q formats, rounding/overflow, bit-true FIR and biquad, .mem export
    multirate  polyphase decimation/interpolation, resampling, CIC + compensator
    adaptive   LMS, NLMS, RLS

Every function is cross-checked against scipy.signal / numpy.fft in tests/.
"""
from . import core, fir, iir, fft, fixed, multirate, adaptive  # noqa: F401

__version__ = "1.0.0"
