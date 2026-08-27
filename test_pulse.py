"""Dedicated analysis of the AxoGraph '0001 Test Pulse 1-Ch' protocol.

The acquired test-pulse files used in this project contain a single measured
Current-1 channel while the protocol applies a small voltage-command pulse.
Therefore Rs/Rm/Cm must be derived from the measured current transient, not
from a current-injection step or a membrane-voltage transient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from models import Recording, Sweep


@dataclass
class TestPulseProperties:
    series_resistance: Optional[float] = None
    membrane_resistance: Optional[float] = None
    membrane_capacitance: Optional[float] = None
    n_valid_sweeps: int = 0
    sweep_values: list = field(default_factory=list)
    pulse_amplitude_mv: float = 10.0
    pulse_width_ms: float = 20.0


def _single_exp(t_ms, i_ss, amplitude, tau_ms):
    return i_ss + amplitude * np.exp(-t_ms / tau_ms)


def _find_test_pulse_window(signal: np.ndarray, fs: float):
    """Find the first rectangular test pulse from the measured current trace.

    The standard protocol begins near 20 ms and lasts ~20 ms, but derivative
    detection is used so minor protocol timing changes remain supported.
    """
    if len(signal) < 20 or fs <= 0:
        return None, None
    x = np.asarray(signal, dtype=float)
    # lightly smooth before differentiating; preserves the fast charging peak
    win = max(int(round(0.0001 * fs)), 1)
    if win > 1:
        kernel = np.ones(win) / win
        smooth = np.convolve(x, kernel, mode="same")
    else:
        smooth = x
    d = np.abs(np.diff(smooth))
    # Ignore the outer 2 ms where edge artifacts can dominate.
    margin = max(int(0.002 * fs), 1)
    search = d[margin: max(len(d) - margin, margin + 1)]
    if not len(search):
        return None, None
    onset = margin + int(np.argmax(search))
    min_sep = max(int(0.005 * fs), 2)
    max_sep = max(int(0.060 * fs), min_sep + 2)
    lo = min(onset + min_sep, len(d) - 1)
    hi = min(onset + max_sep, len(d))
    if hi <= lo:
        return None, None
    offset = lo + int(np.argmax(d[lo:hi]))
    return onset + 1, offset + 1


def _analyze_current_test_pulse(sweep: Sweep, pulse_amplitude_mv: float = 10.0):
    """Return (Rs MOhm, Rm MOhm, Cm pF) for one Current-1 test-pulse trace.

    AxoGraph's test-pulse convention is approximated as:
      Rs = dV / I_peak
      Rtotal = dV / I_steady
      Rm = Rtotal - Rs
      Cm = tau / Rm

    where dV is the known command step (default 10 mV), I_peak is the initial
    current above baseline and I_steady is the late pulse current above
    baseline. The decay tau is fit from the post-peak current transient.
    """
    # In 1-channel test pulse files Neo can classify Current-1 as the Recording
    # voltage field because it is the only analog signal. Accept either field.
    raw = sweep.current if sweep.current is not None else sweep.voltage
    if raw is None or len(raw) < 20:
        return None
    x = np.asarray(raw, dtype=float)
    fs = float(sweep.sampling_rate)
    onset, offset = _find_test_pulse_window(x, fs)
    if onset is None or offset is None or offset <= onset + 5:
        return None

    pre_n = max(int(0.005 * fs), 5)
    baseline = float(np.median(x[max(0, onset - pre_n):onset]))
    pulse = x[onset:offset] - baseline
    if len(pulse) < 10:
        return None

    # Choose polarity from the largest early deflection and rectify accordingly.
    early_n = min(max(int(0.002 * fs), 5), len(pulse))
    sign = 1.0 if abs(np.max(pulse[:early_n])) >= abs(np.min(pulse[:early_n])) else -1.0
    y = pulse * sign
    peak_search_n = min(max(int(0.003 * fs), 8), len(y))
    peak_rel = int(np.argmax(y[:peak_search_n]))
    i_peak = float(y[peak_rel])
    tail_n = min(max(int(0.003 * fs), 5), max(len(y) // 4, 5))
    i_ss = float(np.median(y[-tail_n:]))
    if i_peak <= 0 or i_ss <= 0 or i_peak <= i_ss:
        return None

    # Determine whether raw units are pA or A. Normal loader normally converts
    # current to pA, but the one-channel fallback may leave the current trace in
    # voltage's slot. Values near 1e-9 strongly indicate amperes.
    scale_to_pa = 1e12 if max(abs(i_peak), abs(i_ss)) < 1e-3 else 1.0
    i_peak_pa = i_peak * scale_to_pa
    i_ss_pa = i_ss * scale_to_pa

    rs = pulse_amplitude_mv / i_peak_pa * 1000.0
    r_total = pulse_amplitude_mv / i_ss_pa * 1000.0
    rm = r_total - rs
    if not (0 < rs < 1000 and 0 < rm < 10000):
        return None

    # Fit exponential current decay from the peak until shortly before offset.
    fit_y = y[peak_rel:]
    # Exclude the final ~0.5 ms to avoid the offset transient.
    trim = max(int(0.0005 * fs), 1)
    if len(fit_y) > trim + 5:
        fit_y = fit_y[:-trim]
    t_ms = np.arange(len(fit_y)) / fs * 1000.0
    tau_ms = None
    try:
        amp0 = float(fit_y[0] - i_ss)
        popt, _ = curve_fit(
            _single_exp, t_ms, fit_y,
            p0=[i_ss, amp0, 1.0],
            bounds=([0.0, 0.0, 0.01], [np.inf, np.inf, 100.0]),
            maxfev=10000,
        )
        tau_ms = float(popt[2])
    except Exception:
        pass
    if tau_ms is None or not (0.01 < tau_ms < 100.0):
        return None
    cm = tau_ms * 1000.0 / rm
    if not (0 < cm < 10000):
        return None
    return rs, rm, cm, onset, offset, i_peak_pa, i_ss_pa, tau_ms


def compute_test_pulse_properties(recording: Recording,
                                  pulse_amplitude_mv: float = 10.0,
                                  ) -> TestPulseProperties:
    values = []
    for sweep in recording.sweeps:
        result = _analyze_current_test_pulse(sweep, pulse_amplitude_mv)
        if result is not None:
            rs, rm, cm, onset, offset, i_peak, i_ss, tau = result
            values.append({
                "Sweep": sweep.index,
                "Rs (MOhm)": rs,
                "Rm (MOhm)": rm,
                "Cm (pF)": cm,
                "I peak (pA)": i_peak,
                "I steady (pA)": i_ss,
                "Tau (ms)": tau,
                "Onset (ms)": onset / sweep.sampling_rate * 1000.0,
                "Offset (ms)": offset / sweep.sampling_rate * 1000.0,
            })
    if not values:
        return TestPulseProperties(pulse_amplitude_mv=pulse_amplitude_mv)
    return TestPulseProperties(
        series_resistance=float(np.mean([v["Rs (MOhm)"] for v in values])),
        membrane_resistance=float(np.mean([v["Rm (MOhm)"] for v in values])),
        membrane_capacitance=float(np.mean([v["Cm (pF)"] for v in values])),
        n_valid_sweeps=len(values),
        sweep_values=values,
        pulse_amplitude_mv=pulse_amplitude_mv,
    )
