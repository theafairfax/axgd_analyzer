"""
Action potential (spike) detection and feature extraction from a single
current-clamp sweep.

Threshold is defined by the standard max-dV/dt criterion: walking backward
from each peak, threshold is the first point where dV/dt drops below a
fraction of its own maximum rate of rise.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from models import Sweep, SpikeFeatures, SpikeEventFeatures
from template_matching import (
    DEFAULT_TEMPLATE_MS, DEFAULT_TEMPLATE_MV, TemplateConfig,
    detect_events_by_template,
)


def _spike_features_at_peak(v: np.ndarray, dv: np.ndarray, fs: float,
                             global_idx: int, next_global_idx: int,
                             prev_peak_time: Optional[float],
                             rise_thresh_frac: float = 0.05,
                             ) -> SpikeFeatures:
    """Compute one spike's features given its peak sample index. Shared by
    both the prominence-based detector and the template-matching detector
    so threshold/half-width/AHP logic stays in one place."""
    peak_time = global_idx / fs
    peak_v = float(v[global_idx])

    # --- threshold via max dV/dt, searching backward up to 3 ms ---
    search_back = int(0.003 * fs)
    lo = max(global_idx - search_back, 0)
    rise_segment = dv[lo:global_idx + 1]
    max_rise_rel = int(np.argmax(rise_segment)) if len(rise_segment) else 0
    max_rise = float(rise_segment[max_rise_rel]) if len(rise_segment) else np.nan
    rise_thresh = max(rise_thresh_frac * max_rise, 5.0)  # mV/ms, floor at 5 mV/ms
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
    li = global_idx
    while li > lo and v[li] > half_v:
        li -= 1
    search_fwd = int(0.005 * fs)
    hi_bound = min(global_idx + search_fwd, len(v) - 1)
    ri = global_idx
    while ri < hi_bound and v[ri] > half_v:
        ri += 1
    half_width_ms = (ri - li) / fs * 1000.0 if ri > li else np.nan

    max_fall = float(np.min(dv[global_idx:hi_bound + 1])) if hi_bound > global_idx else np.nan

    # --- AHP: minimum voltage between this spike and the next (or tail) ---
    ahp_search = v[global_idx:next_global_idx + 1]
    if len(ahp_search) > 0:
        ahp_rel_idx = int(np.argmin(ahp_search))
        ahp_v = float(ahp_search[ahp_rel_idx])
        ahp_t = (global_idx + ahp_rel_idx) / fs
    else:
        ahp_v, ahp_t = None, None

    isi_prev_ms = (peak_time - prev_peak_time) * 1000.0 if prev_peak_time is not None else None

    return SpikeFeatures(
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
    )


def detect_spikes(sweep: Sweep, min_peak_mv: float = -10.0,
                   min_prominence_mv: float = 20.0,
                   refractory_ms: float = 1.0) -> List[SpikeFeatures]:
    """Detect spikes within the step window (falls back to the whole sweep
    if no step was detected) and compute per-spike features. Uses simple
    peak-prominence detection -- see ``detect_spikes_template`` for the
    template-matching detector that mirrors AxoGraph's Events module."""
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
        next_global = start + peaks[k + 1] if k + 1 < len(peaks) else min(global_idx + int(0.05 * fs), len(v) - 1)
        spike = _spike_features_at_peak(v, dv, fs, global_idx, next_global, prev_peak_time)
        prev_peak_time = spike.peak_time
        spikes.append(spike)

    return spikes


def detect_spikes_template(sweep: Sweep, cfg: TemplateConfig = TemplateConfig(),
                            template_ms: np.ndarray = DEFAULT_TEMPLATE_MS,
                            template_mv: np.ndarray = DEFAULT_TEMPLATE_MV,
                            ) -> List[SpikeFeatures]:
    """Detect spikes via sliding template-matching (Clements & Bekkers),
    using the built-in default AP template (or a custom one), restricted to
    the latency window from ``cfg`` relative to step onset -- mirrors the
    AxoGraph 'Events' settings on the summary sheet (A1-A9)."""
    v = sweep.voltage
    fs = sweep.sampling_rate
    if len(v) == 0 or fs == 0 or sweep.step_onset_idx is None:
        return []

    onset = sweep.step_onset_idx
    search_start = onset + int(cfg.latency_start_ms / 1000.0 * fs)
    if cfg.latency_end_ms is not None:
        search_end = min(onset + int(cfg.latency_end_ms / 1000.0 * fs), len(v))
    else:
        tail = sweep.step_offset_idx if sweep.step_offset_idx is not None else len(v) - 1
        search_end = min(tail + int(0.05 * fs), len(v))
    search_start = max(search_start, 0)
    if search_end <= search_start:
        return []

    peak_idxs = detect_events_by_template(
        v, fs, template_ms, template_mv, cfg,
        search_start_idx=search_start, search_end_idx=search_end,
    )
    if len(peak_idxs) == 0:
        return []

    dv = np.gradient(v) * fs / 1000.0  # mV/ms

    spikes: List[SpikeFeatures] = []
    prev_peak_time = None
    for k, global_idx in enumerate(peak_idxs):
        next_global = int(peak_idxs[k + 1]) if k + 1 < len(peak_idxs) else min(global_idx + int(0.05 * fs), len(v) - 1)
        spike = _spike_features_at_peak(
            v, dv, fs, int(global_idx), next_global, prev_peak_time,
            rise_thresh_frac=cfg.voltage_threshold_pct,
        )
        if spike.amplitude < cfg.amplitude_reject_mv:
            continue
        prev_peak_time = spike.peak_time
        spikes.append(spike)

    return spikes


def _fit_exp_tau(t_ms: np.ndarray, v: np.ndarray, tau_guess_ms: float = 2.0
                  ) -> Optional[float]:
    """Fit v(t) = v_inf + delta*exp(-t/tau) and return tau in ms, or None."""
    if len(t_ms) < 5:
        return None
    v_inf_guess = float(v[-1])
    delta_guess = float(v[0] - v_inf_guess)
    try:
        popt, _ = curve_fit(
            lambda t, v_inf, delta, tau: v_inf + delta * np.exp(-t / tau),
            t_ms, v, p0=[v_inf_guess, delta_guess, tau_guess_ms], maxfev=5000,
        )
        tau = float(popt[2])
        if 0.01 < tau < 200.0:
            return tau
    except Exception:
        return None
    return None


def build_spike_event_features(v: np.ndarray, fs: float, spike: SpikeFeatures,
                                step_onset_idx: int) -> SpikeEventFeatures:
    """Build the SPIKE-block waveform summary (Location/Onset/Rise/Width/
    Decay) for one spike, relative to step onset -- matches the SPIKE
    column group on the summary sheet."""
    onset_t_ms = step_onset_idx / fs * 1000.0
    peak_idx = int(round(spike.peak_time * fs))
    thr_idx = int(round(spike.threshold_time * fs))

    # Decay: fit from the peak down toward the AHP trough (or a fixed 5 ms
    # window if no trough was found).
    if spike.ahp_time is not None:
        decay_end_idx = int(round(spike.ahp_time * fs))
    else:
        decay_end_idx = min(peak_idx + int(0.005 * fs), len(v) - 1)
    decay_end_idx = max(decay_end_idx, peak_idx + 2)
    decay_end_idx = min(decay_end_idx, len(v) - 1)
    seg_v = v[peak_idx:decay_end_idx + 1]
    seg_t = (np.arange(peak_idx, decay_end_idx + 1) - peak_idx) / fs * 1000.0
    decay_ms = _fit_exp_tau(seg_t, seg_v, tau_guess_ms=1.0)
    if decay_ms is None:
        decay_ms = float(seg_t[-1]) if len(seg_t) else np.nan

    return SpikeEventFeatures(
        peak_voltage=spike.peak_voltage,
        location_ms=spike.peak_time * 1000.0 - onset_t_ms,
        onset_ms=spike.threshold_time * 1000.0 - onset_t_ms,
        rise_ms=(peak_idx - thr_idx) / fs * 1000.0,
        width_ms=spike.half_width,
        decay_ms=decay_ms,
    )


def build_ahp_event_features(v: np.ndarray, fs: float, spike: SpikeFeatures,
                              step_onset_idx: int,
                              baseline_v: Optional[float] = None,
                              ) -> Optional[SpikeEventFeatures]:
    """Build the AHP-block waveform summary for one spike's after-
    hyperpolarization: onset is where the trace falls back through the
    pre-spike baseline after the peak, location/rise describe the trough,
    width is the half-amplitude width of the dip, and decay is the
    recovery time constant back toward baseline."""
    if spike.ahp_voltage is None or spike.ahp_time is None:
        return None
    onset_t_ms = step_onset_idx / fs * 1000.0
    peak_idx = int(round(spike.peak_time * fs))
    trough_idx = int(round(spike.ahp_time * fs))
    if trough_idx <= peak_idx:
        return None

    base_v = baseline_v if baseline_v is not None else spike.threshold_voltage
    ahp_amp = base_v - spike.ahp_voltage  # positive = hyperpolarized below baseline

    # Onset: first point after the peak where V drops back through baseline.
    onset_idx = peak_idx
    for idx in range(peak_idx, trough_idx + 1):
        if v[idx] <= base_v:
            onset_idx = idx
            break
    else:
        onset_idx = peak_idx

    # Half-amplitude width of the dip around the trough.
    half_v = spike.ahp_voltage + ahp_amp / 2.0 if ahp_amp > 0 else spike.ahp_voltage
    li = trough_idx
    lo_bound = onset_idx
    while li > lo_bound and v[li] < half_v:
        li -= 1
    search_fwd = int(0.02 * fs)
    hi_bound = min(trough_idx + search_fwd, len(v) - 1)
    ri = trough_idx
    while ri < hi_bound and v[ri] < half_v:
        ri += 1
    width_ms = (ri - li) / fs * 1000.0 if ri > li else np.nan

    # Decay: recovery from trough back toward baseline.
    decay_end_idx = min(trough_idx + int(0.02 * fs), len(v) - 1)
    seg_v = v[trough_idx:decay_end_idx + 1]
    seg_t = (np.arange(trough_idx, decay_end_idx + 1) - trough_idx) / fs * 1000.0
    decay_ms = _fit_exp_tau(seg_t, seg_v, tau_guess_ms=5.0)
    if decay_ms is None:
        decay_ms = float(seg_t[-1]) if len(seg_t) else np.nan

    return SpikeEventFeatures(
        peak_voltage=spike.ahp_voltage,
        location_ms=spike.ahp_time * 1000.0 - onset_t_ms,
        onset_ms=onset_idx / fs * 1000.0 - onset_t_ms,
        rise_ms=(trough_idx - onset_idx) / fs * 1000.0,
        width_ms=width_ms,
        decay_ms=decay_ms,
    )


def first_singlet_or_doublet(spikes: List[SpikeFeatures], burst_window_ms: float = 20.0
                              ) -> List[SpikeFeatures]:
    """Return just the first spike, or the first two if they form a burst
    (2nd spike's ISI to the 1st is within ``burst_window_ms``) -- matches
    note A9: 'For Shape the very first singlet or doublet is used'."""
    if not spikes:
        return []
    if len(spikes) >= 2 and spikes[1].isi_prev_ms is not None and spikes[1].isi_prev_ms <= burst_window_ms:
        return spikes[:2]
    return spikes[:1]


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
