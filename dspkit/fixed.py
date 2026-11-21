"""
dspkit.fixed -- bit-true fixed-point modelling (the bridge to FPGA / MCU code).

Notation used here:  Q(m, n)
    m = integer bits INCLUDING the sign bit, n = fractional bits, word W = m + n.
    Q(1,15): 16-bit, range [-1, 1 - 2^-15], LSB = 2^-15      ("Q15" / "Q1.15")
    Q(2,14): 16-bit, range [-2, 2 - 2^-14]                   (biquad a1 needs |a1| < 2)
A real value r is stored as the integer code c = r * 2^n. All arithmetic below is
done on integer codes held in numpy int64, which models a hardware datapath
exactly as long as no intermediate exceeds 63 bits.

Arithmetic rules (what the hardware does)
    product   Q(m1,n1) x Q(m2,n2) -> Q(m1+m2, n1+n2)     (full precision, W1+W2 bits)
    add       align binary points, +1 bit of growth per addition level
    N-term sum of products grows by ceil(log2 N) bits (worst case)
    requantise: right-shift by (n_in - n_out) with a ROUNDING mode, then handle
                OVERFLOW (saturate or wrap) into the output word.

Rounding modes for a right shift by s bits
    'floor'      drop LSBs (free in hardware), bias -LSB/2
    'trunc'      toward zero (sign-magnitude style), bias toward 0
    'round'      add 2^(s-1) then floor: round-half-up, tiny +bias
    'convergent' round-half-to-even: unbiased (what DSP48 'symmetric/convergent' modes give)
Overflow modes
    'saturate'   clamp to [min, max]          (needs comparators)
    'wrap'       two's-complement wrap-around (free; modular arithmetic)
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "QFormat", "quantize", "to_real", "round_shift", "handle_overflow", "requantize",
    "fir_fixed", "fir_acc_bits", "biquad_fixed", "sos_fixed",
    "sqnr_db", "to_hex", "write_mem",
]


class QFormat:
    """Fixed-point format descriptor Q(m, n) (m includes the sign bit if signed)."""

    def __init__(self, m, n, signed=True):
        self.m, self.n, self.signed = int(m), int(n), bool(signed)
        self.W = self.m + self.n
        if self.W > 62:
            raise ValueError("word too wide for int64 modelling")
        if signed:
            self.min_code = -(1 << (self.W - 1))
            self.max_code = (1 << (self.W - 1)) - 1
        else:
            self.min_code = 0
            self.max_code = (1 << self.W) - 1

    @property
    def lsb(self):
        return 2.0 ** -self.n

    @property
    def min(self):
        return self.min_code * self.lsb

    @property
    def max(self):
        return self.max_code * self.lsb

    def __repr__(self):
        s = "s" if self.signed else "u"
        return f"Q({self.m},{self.n}) [{self.W}-bit {s}, range {self.min:g}..{self.max:g}]"


# --------------------------------------------------------------------------- #
def _round_real(v, mode):
    if mode == "floor":
        return np.floor(v)
    if mode == "trunc":
        return np.trunc(v)
    if mode == "round":
        return np.floor(v + 0.5)
    if mode == "convergent":
        return np.round(v)                 # numpy rounds half to even
    raise ValueError(mode)


def handle_overflow(codes, q, mode="saturate"):
    """Force integer codes into q's range by saturation or two's-complement wrap."""
    codes = np.asarray(codes, dtype=np.int64)
    if mode == "saturate":
        return np.clip(codes, q.min_code, q.max_code)
    if mode == "wrap":
        span = np.int64(1) << q.W
        return ((codes - q.min_code) % span) + q.min_code
    raise ValueError(mode)


def quantize(x, q, rounding="round", overflow="saturate"):
    """Real values -> integer codes in format q."""
    v = _round_real(np.asarray(x, float) * (2.0 ** q.n), rounding)
    return handle_overflow(v.astype(np.int64), q, overflow)


def to_real(codes, q_or_n):
    """Integer codes -> real values. Accepts a QFormat or a fractional-bit count."""
    n = q_or_n.n if isinstance(q_or_n, QFormat) else int(q_or_n)
    return np.asarray(codes, dtype=float) * 2.0 ** -n


def round_shift(v, s, mode="round"):
    """Arithmetic right shift of integer codes by s >= 0 bits with rounding.
    (numpy '>>' on signed ints is an arithmetic shift = floor division by 2^s.)"""
    v = np.asarray(v, dtype=np.int64)
    if s == 0:
        return v.copy()
    if s < 0:
        return v << (-s)
    if mode == "floor":
        return v >> s
    if mode == "trunc":
        return np.where(v < 0, -((-v) >> s), v >> s)
    if mode == "round":
        return (v + (np.int64(1) << (s - 1))) >> s
    if mode == "convergent":
        fl = v >> s
        rem = v - (fl << s)                       # 0 .. 2^s - 1
        half = np.int64(1) << (s - 1)
        up = (rem > half) | ((rem == half) & (fl & 1).astype(bool))
        return fl + up
    raise ValueError(mode)


def requantize(codes, n_in, q_out, rounding="round", overflow="saturate"):
    """Change binary point from n_in fractional bits to q_out.n, then fit q_out's word."""
    return handle_overflow(round_shift(codes, n_in - q_out.n, rounding), q_out, overflow)


# --------------------------------------------------------------------------- #
# Bit-true FIR
# --------------------------------------------------------------------------- #
def fir_acc_bits(h_codes, x_bits):
    """Accumulator width that can never overflow for a W_x-bit signed input.

    max |acc| = 2^(Wx-1) * sum |h_k|   ->   bits = ceil(log2(max|acc|)) + 1 (sign)
    Compare with the cruder bound Wx + Wh + ceil(log2 N).
    """
    s = int(np.sum(np.abs(np.asarray(h_codes, dtype=np.int64))))
    return int(np.ceil(np.log2(s * 2.0 ** (x_bits - 1)))) + 1


def fir_fixed(x_codes, h_codes, n_x, n_h, q_out, acc_bits=None,
              rounding="round", overflow="saturate", return_acc=False):
    """Bit-true causal FIR: y[n] = Q_out( sum_k h[k] x[n-k] ).

    x_codes : input integer codes with n_x fractional bits.
    h_codes : coefficient codes with n_h fractional bits.
    The accumulator holds full products (n_x + n_h fractional bits). If acc_bits is
    given, the accumulator is WRAPPED to that width after every addition -- exactly
    like a finite hardware register. (Thanks to modular arithmetic the final value
    is still correct whenever the TRUE sum fits, even if partial sums overflowed.)
    """
    x = np.asarray(x_codes, dtype=np.int64)
    h = np.asarray(h_codes, dtype=np.int64)
    acc = np.zeros(len(x), dtype=np.int64)
    qa = QFormat(acc_bits, 0) if acc_bits else None
    for k, hk in enumerate(h):                       # accumulate tap by tap
        acc[k:] += hk * x[:len(x) - k]
        if qa is not None:
            acc = handle_overflow(acc, qa, "wrap")
    y = requantize(acc, n_x + n_h, q_out, rounding, overflow)
    return (y, acc) if return_acc else y


# --------------------------------------------------------------------------- #
# Bit-true biquad (Direct Form I, single wide accumulator = one DSP MAC chain)
# --------------------------------------------------------------------------- #
def biquad_fixed(x_codes, coef_codes, n_coef, q_io, rounding="round",
                 overflow="saturate", state=None):
    """DF-I biquad:
        acc  = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]   (full precision)
        y[n] = Q_io( acc >> n_coef )
    DF-I is preferred in fixed point: the only rounding point is the output, and
    the recursion operates on the (saturated) output word, so internal nodes
    cannot overflow silently (DF-II's internal w[n] can have huge gain).

    coef_codes : [b0, b1, b2, a0, a1, a2] integer codes with n_coef fractional bits
                 (a0 ignored, assumed 1).
    state      : (x1, x2, y1, y2) codes, or None.
    Returns y codes and final state.
    """
    b0, b1, b2, _, a1, a2 = (int(c) for c in coef_codes)
    x1, x2, y1, y2 = (0, 0, 0, 0) if state is None else (int(s) for s in state)
    x = np.asarray(x_codes, dtype=np.int64)
    y = np.empty_like(x)
    half = 1 << (n_coef - 1) if n_coef > 0 else 0
    for i, xn in enumerate(x):
        xn = int(xn)
        acc = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2       # Python int: exact
        if rounding == "round":
            yn = (acc + half) >> n_coef
        elif rounding == "floor":
            yn = acc >> n_coef
        else:
            yn = int(round_shift(np.array([acc]), n_coef, rounding)[0])
        yn = int(handle_overflow(np.array([yn]), q_io, overflow)[0])
        x2, x1 = x1, xn
        y2, y1 = y1, yn
        y[i] = yn
    return y, (x1, x2, y1, y2)


def sos_fixed(x, sos, q_coef, q_io, rounding="round", overflow="saturate"):
    """Quantise an SOS cascade and run it bit-true on real input x.
    Returns (y_real, coef_codes) so the exact coefficients can be exported."""
    sos = np.atleast_2d(sos)
    coef = quantize(sos, q_coef, "round", "saturate")
    y = quantize(x, q_io, rounding, "saturate")
    for c in coef:
        y, _ = biquad_fixed(y, c, q_coef.n, q_io, rounding, overflow)
    return to_real(y, q_io), coef


# --------------------------------------------------------------------------- #
# Metrics and export
# --------------------------------------------------------------------------- #
def sqnr_db(ref, test):
    """Signal-to-quantisation-noise ratio 10 log10( sum ref^2 / sum (ref-test)^2 ).
    Rule of thumb for a full-scale sine: 6.02 W + 1.76 dB."""
    ref = np.asarray(ref, float)
    err = ref - np.asarray(test, float)
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum(err ** 2), 1e-300))


def to_hex(codes, width):
    """Two's-complement hex strings (width bits), e.g. -1 @16b -> 'ffff'."""
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    return [format(int(c) & mask, f"0{digits}x") for c in np.ravel(codes)]


def write_mem(path, codes, width, comment=None):
    """Write a $readmemh-compatible file (one word per line).
    Verilog:  logic signed [W-1:0] rom [N]; initial $readmemh("taps.mem", rom);"""
    with open(path, "w") as f:
        if comment:
            for line in str(comment).splitlines():
                f.write(f"// {line}\n")
        f.write("\n".join(to_hex(codes, width)) + "\n")
