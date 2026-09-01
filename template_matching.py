"""
Sliding template-matching event detection (Clements & Bekkers, 1997), the
same underlying method AxoGraph's "Events" module uses for spike/AHP
detection with a user-supplied template waveform.

The default action-potential template is embedded from the lab's AxoGraph
"0001 Bursty Template Function".  Two coordinates matter for AxoGraph-style
analysis:

* detection_index: the peak of the template-detection criterion.  This is the
  event-latency anchor AxoGraph passes into its capture code.
* peak_index: the nearest raw AP maximum.  This remains useful for conventional
  per-spike measurements, but must not be used as the capture origin.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, resample


_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TEMPLATE_CSV = os.path.join(_HERE, "default_template.csv")
_DEFAULT_TEMPLATE_AXGX = os.path.join(_HERE, "default_template.axgx")


def _load_default_template() -> Tuple[np.ndarray, np.ndarray]:
    if os.path.exists(_DEFAULT_TEMPLATE_CSV):
        arr = np.loadtxt(_DEFAULT_TEMPLATE_CSV, delimiter=",", skiprows=1)
        return arr[:, 0], arr[:, 1]
    if os.path.exists(_DEFAULT_TEMPLATE_AXGX):
        # The lab template is an AxoGraph exchange file containing explicit
        # time and voltage columns (both type 7), rather than a type-9 series
        # column. Preserve the original points used by AxoGraph Events.
        raw = open(_DEFAULT_TEMPLATE_AXGX, "rb").read()
        if raw[:4] != b"axgx":
            raise ValueError("default_template.axgx is not an AxoGraph exchange file")
        n_columns = struct.unpack(">l", raw[8:12])[0]
        pos, columns = 12, []
        for _ in range(n_columns):
            n, column_type, title_len = struct.unpack(">lll", raw[pos:pos + 12])
            pos += 12 + title_len
            if column_type == 9:
                start, increment = struct.unpack(">dd", raw[pos:pos + 16])
                pos += 16
                columns.append(start + np.arange(n) * increment)
            else:
                values = np.frombuffer(raw[pos:pos + 8 * n], dtype=">f8").astype(float)
                pos += 8 * n
                columns.append(values)
        if len(columns) < 2 or len(columns[0]) != len(columns[1]):
            raise ValueError("default_template.axgx must contain time and voltage columns")
        return columns[0] * 1000.0, columns[1]
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
DEFAULT_TEMPLATE_PEAK_OFFSET_MS = float(
    DEFAULT_TEMPLATE_MS[int(np.argmax(DEFAULT_TEMPLATE_MV))] - DEFAULT_TEMPLATE_MS[0]
)


@dataclass(frozen=True)
class TemplateConfig:
    """Lab-specific AxoGraph Events settings used by the analyzer."""
    baseline_ms: float = 1.0
    length_ms: float = 4.0
    threshold: float = 1.0
    min_separation_ms: float = 2.0
    amplitude_reject_mv: float = 20.0
    latency_start_ms: float = 20.0
    latency_end_ms: Optional[float] = 530.0
    n_episodes: int = 20
    burst_window_ms: float = 20.0
    voltage_threshold_pct: float = 0.10
    threshold_dvdt_mv_per_ms: Optional[float] = 20.0
    # AxoGraph captured-event window used by the lab workflow.  The source
    # code constructs x as (1-captureBaselinePoints)*dx and, for template
    # detection, captureOffset = captureBaselinePoints-templateBaselinePoints-1.
    capture_baseline_ms: float = 10.0
    capture_ms: float = 40.0
    # Direct raw/captured pairs place AxoGraph's capture anchor 9--11 samples
    # later than our criterion maximum at 80 kHz. Correct capture construction,
    # while preserving AxoGraph time zero at captured index 799.
    capture_anchor_correction_ms: float = 0.125


@dataclass(frozen=True)
class TemplateEvent:
    detection_index: int
    peak_index: int
    criterion: float
    scale: float


def _resample_template(template_ms: np.ndarray, template_mv: np.ndarray,
                        target_fs: float) -> np.ndarray:
    duration_s = (template_ms[-1] - template_ms[0]) / 1000.0
    n_target = max(int(round(duration_s * target_fs)), 2)
    return resample(template_mv, n_target)


def clements_bekkers_scan(data: np.ndarray, fs: float, template: np.ndarray,
                           ) -> Tuple[np.ndarray, np.ndarray]:
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
    sum_td = np.convolve(data, template[::-1], mode="valid")
    scale = (sum_td - sum_t * sum_d / n) / denom
    offset = (sum_d - scale * sum_t) / n
    sse = (sum_d2 - 2 * scale * sum_td - 2 * offset * sum_d
           + (scale ** 2) * sum_t2 + 2 * scale * offset * sum_t
           + n * (offset ** 2))
    sse = np.maximum(sse, 1e-12)
    se = np.sqrt(sse / max(n - 1, 1))
    return scale / se, scale


def detect_template_events(
    data: np.ndarray, fs: float,
    template_ms: np.ndarray = DEFAULT_TEMPLATE_MS,
    template_mv: np.ndarray = DEFAULT_TEMPLATE_MV,
    cfg: TemplateConfig = TemplateConfig(),
    search_start_idx: int = 0,
    search_end_idx: Optional[int] = None,
) -> list[TemplateEvent]:
    """Return AxoGraph-style template detection anchors plus raw AP peaks.

    The detection-criterion peak is preserved instead of being discarded after
    snapping to a voltage maximum.  AxoGraph's Detect Events source uses this
    latency anchor for both measurement and captured-event alignment.
    """
    if search_end_idx is None:
        search_end_idx = len(data)
    window = np.asarray(data[search_start_idx:search_end_idx], dtype=float)
    if len(window) < 10:
        return []
    template = _resample_template(template_ms, template_mv, fs)
    if len(template) < 3 or len(template) >= len(window):
        return []
    dc, scale = clements_bekkers_scan(window, fs, template)
    if len(dc) == 0:
        return []
    dc_valid = np.where(scale > 0, dc, -np.inf)
    min_sep = max(int(cfg.min_separation_ms / 1000.0 * fs), 1)
    detections, _ = find_peaks(dc_valid, height=cfg.threshold, distance=min_sep)
    if len(detections) == 0:
        return []

    # Infer event polarity from the supplied template. Positive AP detection is
    # unchanged; a template multiplied by -1 detects AHPs and snaps to minima.
    template_baseline = float(np.median(template_mv[:max(2, len(template_mv) // 10)]))
    positive_excursion = float(np.max(template_mv) - template_baseline)
    negative_excursion = float(template_baseline - np.min(template_mv))
    polarity = 1 if positive_excursion >= negative_excursion else -1
    extremum = int(np.argmax(template_mv) if polarity > 0 else np.argmin(template_mv))
    peak_offset_ms = float(template_ms[extremum] - template_ms[0])
    peak_offset = int(round(peak_offset_ms / 1000.0 * fs))
    tol = max(int(0.001 * fs), 1)
    out=[]
    for p0 in detections:
        p=int(p0)
        approx=p+peak_offset
        lo=max(approx-tol,0); hi=min(approx+tol+1,len(window))
        if hi<=lo:
            continue
        local_peak=lo+int(np.argmax(window[lo:hi]) if polarity > 0 else np.argmin(window[lo:hi]))
        out.append(TemplateEvent(
            detection_index=p+search_start_idx,
            peak_index=local_peak+search_start_idx,
            criterion=float(dc[p]),
            scale=float(scale[p]),
        ))
    # Keep detector order while dropping duplicate raw peaks.
    dedup=[]; seen=set()
    for e in out:
        if e.peak_index in seen: continue
        seen.add(e.peak_index); dedup.append(e)
    return dedup


def detect_events_by_template(
    data: np.ndarray, fs: float,
    template_ms: np.ndarray = DEFAULT_TEMPLATE_MS,
    template_mv: np.ndarray = DEFAULT_TEMPLATE_MV,
    cfg: TemplateConfig = TemplateConfig(),
    search_start_idx: int = 0,
    search_end_idx: Optional[int] = None,
) -> np.ndarray:
    """Backward-compatible API returning snapped raw AP peak indices."""
    return np.asarray([e.peak_index for e in detect_template_events(
        data,fs,template_ms,template_mv,cfg,search_start_idx,search_end_idx
    )],dtype=int)
