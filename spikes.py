"""
Action potential (spike) detection and feature extraction from a single
current-clamp sweep.

Threshold is defined by the standard max-dV/dt criterion: walking backward
from each peak, threshold is the first point where dV/dt drops below a
fraction of its own maximum rate of rise.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.signal import find_peaks

from .models import Sweep, SpikeFeatures


def detect_spikes(sweep: Sweep, min_peak_mv: float = -10.0,
                   min_prominence_mv: float = 20.0,
                   refractory_ms: float = 1.0) -> List[SpikeFeatures]:
    """Detect spikes within the step window (falls back to the whole sweep
    if no step was detected) and compute per-spike features."""
    v = sweep.voltage
    fs = sweep.sampling_rate
    if len(v) == 0 or fs == 0:
        return []

    start = sweep.step_onset_idx if sweep.step_onset_idx is not None else 0
    end = sweep.step_offset_idx if sweep.step_offset_idx is not None else len(v) - 1
    end = min(end + int(0.05 * fs), len(v) - 1)  # include a little tail for AHP

    window_v = v[start:end + 1]
    if len(window_v) < 3:
        return []

    distance = max(int(refractory_ms / 1000 * fs), 1)
    peaks, props = find_peaks(
        window_v, height=min_peak_mv, prominence=min_prominence_mv,
        distance=distance,
    )
    if len(peaks) == 0:
        return []

    dv = np.gradient(v) * fs / 1000.0  # mV/ms

    spikes: List[SpikeFeatures] = []
    prev_peak_time = None
    for k, p in enumerate(peaks):
        global_idx = start + p
        peak_time = global_idx / fs
        peak_v = float(v[global_idx])

        # --- threshold via max dV/dt, searching backward up to 3 ms ---
        search_back = int(0.003 * fs)
        lo = max(global_idx - search_back, 0)
        rise_segment = dv[lo:global_idx + 1]
        if len(rise_segment) == 0:
            continue
        # Find the point of maximum rate of rise first (this sits on the
        # upstroke, before the peak, where slope flattens back toward 0).
        max_rise_rel = int(np.argmax(rise_segment))
        max_rise = float(rise_segment[max_rise_rel])
        rise_thresh = max(0.05 * max_rise, 5.0)  # mV/ms, floor at 5 mV/ms
        # Then walk backward in time from that point to find where the
        # upstroke first exceeds threshold -- the classic AP threshold.
        thr_idx = lo
        for j in range(max_rise_rel, -1, -1):
            if rise_segment[j] < rise_thresh:
                thr_idx = lo + j
                break
        threshold_v = float(v[thr_idx])
        threshold_t = thr_idx / fs

        amplitude = peak_v - threshold_v

        # --- half width: width at half-max amplitude around the peak ---
        half_v = threshold_v + amplitude / 2.0
        # walk left
        li = global_idx
        while li > lo and v[li] > half_v:
            li -= 1
        # walk right
        search_fwd = int(0.005 * fs)
        hi_bound = min(global_idx + search_fwd, len(v) - 1)
        ri = global_idx
        while ri < hi_bound and v[ri] > half_v:
            ri += 1
        half_width_ms = (ri - li) / fs * 1000.0 if ri > li else np.nan

        max_fall = float(np.min(dv[global_idx:hi_bound + 1])) if hi_bound > global_idx else np.nan

        # --- AHP: minimum voltage between this spike and the next (or tail) ---
        next_global = start + peaks[k + 1] if k + 1 < len(peaks) else min(global_idx + int(0.05 * fs), len(v) - 1)
        ahp_search = v[global_idx:next_global + 1]
        if len(ahp_search) > 0:
            ahp_rel_idx = int(np.argmin(ahp_search))
            ahp_v = float(ahp_search[ahp_rel_idx])
            ahp_t = (global_idx + ahp_rel_idx) / fs
        else:
            ahp_v, ahp_t = None, None

        isi_prev_ms = None
        if prev_peak_time is not None:
            isi_prev_ms = (peak_time - prev_peak_time) * 1000.0
        prev_peak_time = peak_time

        spikes.append(SpikeFeatures(
            peak_time=peak_time,
            peak_voltage=peak_v,
            threshold_voltage=threshold_v,
            threshold_time=threshold_t,
            amplitude=amplitude,
            half_width=half_width_ms,
            max_rise_slope=max_rise,
            max_fall_slope=max_fall,
            ahp_voltage=ahp_v,
            ahp_time=ahp_t,
            isi_prev_ms=isi_prev_ms,
        ))

    return spikes


def adaptation_index(spikes: List[SpikeFeatures]) -> Optional[float]:
    """Mean of consecutive-ISI ratios: (ISI[i+1]-ISI[i]) / (ISI[i+1]+ISI[i]).
    Positive values indicate slowing (adapting) firing. Requires >=3 spikes."""
    isis = [s.isi_prev_ms for s in spikes if s.isi_prev_ms is not None]
    if len(isis) < 2:
        return None
    ratios = []
    for i in range(len(isis) - 1):
        a, b = isis[i], isis[i + 1]
        if (a + b) == 0:
            continue
        ratios.append((b - a) / (b + a))
    return float(np.mean(ratios)) if ratios else None
