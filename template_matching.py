"""
Sliding template-matching event detection (Clements & Bekkers, 1997), the
same underlying method AxoGraph's "Events" module uses for spike/AHP
detection with a user-supplied template waveform.

Rather than requiring the user to re-upload an .axgd/.axgx template file
every session, the default action-potential template Isaac built in
AxoGraph ("0001 Bursty Template Function") is embedded here as
``DEFAULT_TEMPLATE_MS`` / ``DEFAULT_TEMPLATE_MV`` (extracted once from
0001_Template_Function.axgx: 400 samples @ 80 kHz, -0.9875 ms to +4.0 ms,
baseline ~1 ms then a single AP-shaped deflection). Users can still tune or
replace it from the sidebar.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, resample


_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TEMPLATE_CSV = os.path.join(_HERE, "default_template.csv")


def _load_default_template() -> Tuple[np.ndarray, np.ndarray]:
    """Load the embedded default template (time_ms, value) from disk if the
    CSV asset is present; otherwise fall back to a small built-in
    alpha-function-shaped approximation so the app still runs."""
    if os.path.exists(_DEFAULT_TEMPLATE_CSV):
        arr = np.loadtxt(_DEFAULT_TEMPLATE_CSV, delimiter=",", skiprows=1)
        return arr[:, 0], arr[:, 1]
    # Fallback: synthesize a rough AP-like template (1 ms baseline, sharp
    # rise, decay with slight undershoot) at 80 kHz over 5 ms.
    fs = 80000.0
    t_ms = (np.arange(400) / fs) * 1000.0 - 0.9875
    v = np.zeros_like(t_ms)
    rise_mask = (t_ms >= 0) & (t_ms < 0.6)
    fall_mask = (t_ms >= 0.6)
    v[rise_mask] = 0.078 * (t_ms[rise_mask] / 0.6)
    tau = 0.5
    v[fall_mask] = 0.078 * np.exp(-(t_ms[fall_mask] - 0.6) / tau) - 0.003
    return t_ms, v


DEFAULT_TEMPLATE_MS, DEFAULT_TEMPLATE_MV = _load_default_template()
DEFAULT_TEMPLATE_FS = 1000.0 / np.median(np.diff(DEFAULT_TEMPLATE_MS))
# Peak location within the template, relative to the template's own start
# (used to align a detected window back onto the spike's true peak time).
DEFAULT_TEMPLATE_PEAK_OFFSET_MS = float(
    DEFAULT_TEMPLATE_MS[int(np.argmax(DEFAULT_TEMPLATE_MV))] - DEFAULT_TEMPLATE_MS[0]
)


@dataclass(frozen=True)
class TemplateConfig:
    """Mirrors the AxoGraph Events-detection settings Isaac used
    (see notes on the 'Current Clamp - Isaac' summary sheet)."""
    baseline_ms: float = 1.0          # flat portion before the event
    length_ms: float = 4.0            # event portion after baseline
    threshold: float = 1.0            # detection-criterion (scale/SE) cutoff
    min_separation_ms: float = 2.0    # min time between accepted events
    amplitude_reject_mv: float = 20.0  # events smaller than this are dropped
    latency_start_ms: float = 20.0    # ignore events before this (from step onset)
    latency_end_ms: Optional[float] = 530.0  # ignore events after this (from step onset)
    n_episodes: int = 10              # number of sweeps to pool for "Captured Aps"
    burst_window_ms: float = 20.0     # ISI window defining a burst at rheobase
    voltage_threshold_pct: float = 0.10  # fraction of max dV/dt defining AP onset (note D8)


def _resample_template(template_ms: np.ndarray, template_mv: np.ndarray,
                        target_fs: float) -> np.ndarray:
    """Resample a template (given at its own native rate) onto a trace
    sampled at ``target_fs`` Hz, preserving its duration."""
    duration_s = (template_ms[-1] - template_ms[0]) / 1000.0
    n_target = max(int(round(duration_s * target_fs)), 2)
    resampled = resample(template_mv, n_target)
    return resampled


def clements_bekkers_scan(data: np.ndarray, fs: float, template: np.ndarray,
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Slide ``template`` over ``data`` and compute the Clements & Bekkers
    (1997) scale and detection-criterion (scale / standard-error) at every
    position. Returns (detection_criterion, scale), each of length
    ``len(data) - len(template) + 1``, indexed by the window's *start*
    sample in ``data``.
    """
    template = np.asarray(template, dtype=float)
    n = len(template)
    if len(data) < n:
        return np.array([]), np.array([])

    sum_t = float(np.sum(template))
    sum_t2 = float(np.sum(template ** 2))
    denom = sum_t2 - (sum_t ** 2) / n
    if denom <= 0:
        return np.array([]), np.array([])

    csum = np.cumsum(np.insert(data, 0, 0.0))
    sum_d = csum[n:] - csum[:-n]
    csum2 = np.cumsum(np.insert(data ** 2, 0, 0.0))
    sum_d2 = csum2[n:] - csum2[:-n]

    # Cross-correlation sum(template[k] * data[i+k]) for every window start i
    sum_td = np.convolve(data, template[::-1], mode="valid")

    scale = (sum_td - sum_t * sum_d / n) / denom
    offset = (sum_d - scale * sum_t) / n

    sse = (sum_d2 - 2 * scale * sum_td - 2 * offset * sum_d
           + (scale ** 2) * sum_t2 + 2 * scale * offset * sum_t
           + n * (offset ** 2))
    sse = np.maximum(sse, 1e-12)
    se = np.sqrt(sse / max(n - 1, 1))
    dc = scale / se
    return dc, scale


def detect_events_by_template(
    data: np.ndarray, fs: float,
    template_ms: np.ndarray = DEFAULT_TEMPLATE_MS,
    template_mv: np.ndarray = DEFAULT_TEMPLATE_MV,
    cfg: TemplateConfig = TemplateConfig(),
    search_start_idx: int = 0,
    search_end_idx: Optional[int] = None,
) -> np.ndarray:
    """Detect event peak indices (into ``data``) using sliding
    template-matching, restricted to ``data[search_start_idx:search_end_idx]``.

    Only positive-going matches (scale > 0) are kept, consistent with
    detecting depolarizing action potentials. Returns sorted, deduplicated
    peak sample indices (global, i.e. relative to ``data``).
    """
    if search_end_idx is None:
        search_end_idx = len(data)
    window = data[search_start_idx:search_end_idx]
    if len(window) < 10:
        return np.array([], dtype=int)

    template = _resample_template(template_ms, template_mv, fs)
    if len(template) < 3 or len(template) >= len(window):
        return np.array([], dtype=int)

    dc, scale = clements_bekkers_scan(window, fs, template)
    if len(dc) == 0:
        return np.array([], dtype=int)

    dc_valid = np.where(scale > 0, dc, -np.inf)
    min_sep_samples = max(int(cfg.min_separation_ms / 1000.0 * fs), 1)
    peaks, _ = find_peaks(dc_valid, height=cfg.threshold, distance=min_sep_samples)
    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Align each detected window start to the template's true peak time,
    # then snap onto the nearest local maximum of the raw data (the actual
    # spike peak) within a small tolerance.
    peak_offset_samples = int(round(DEFAULT_TEMPLATE_PEAK_OFFSET_MS / 1000.0 * fs))
    tol = max(int(0.001 * fs), 1)  # +/- 1 ms snap tolerance
    event_idx = []
    for p in peaks:
        approx = p + peak_offset_samples
        lo = max(approx - tol, 0)
        hi = min(approx + tol + 1, len(window))
        if hi <= lo:
            continue
        local_peak = lo + int(np.argmax(window[lo:hi]))
        event_idx.append(local_peak + search_start_idx)

    event_idx = sorted(set(event_idx))
    return np.array(event_idx, dtype=int)
