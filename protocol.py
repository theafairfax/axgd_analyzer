"""
Current-step detection for patch-clamp sweeps.
"""
from __future__ import annotations
import re
import numpy as np
from models import Recording, Sweep

DEFAULT_RHEOBASE_START_PA = 52.0
DEFAULT_RHEOBASE_INCREMENT_PA = 10.0
DEFAULT_RHEOBASE_ONSET_FRACTION = 0.10
DEFAULT_RHEOBASE_OFFSET_FRACTION = 0.70


def assumed_rheobase_current_pA(sweep_index: int) -> float:
    """Return the standard protocol's injected current for a sweep index."""
    return DEFAULT_RHEOBASE_START_PA + DEFAULT_RHEOBASE_INCREMENT_PA * float(sweep_index)


def _protocol_pulse_parameters(notes: str):
    """Extract onset, width, starting current and increment from AxoGraph notes."""
    if not notes:
        return None
    match = re.search(
        r"Pulse\s+#\d+\s*:\s*Current Command-\d+\s*[\r\n]+([^\r\n]+)",
        notes,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        values = [float(x) for x in match.group(1).split("\t") if x.strip()]
        if len(values) < 13:
            return None
        onset_ms, width_ms, start_pa, increment_pa = (
            values[3], values[5], values[11], values[12]
        )
        if onset_ms < 0 or width_ms <= 0:
            return None
        return onset_ms, width_ms, start_pa, increment_pa
    except (TypeError, ValueError):
        return None


def _find_step_edges(current: np.ndarray, fs: float):
    """Return the onset/offset pair for the dominant square current step."""
    di = np.diff(current)
    if not len(di):
        return None, None

    min_dur = max(int(0.02 * fs), 1)
    abs_di = np.abs(di)
    n_candidates = min(12, len(abs_di))
    candidates = np.argpartition(abs_di, -n_candidates)[-n_candidates:]
    candidates = np.sort(candidates)

    best = None
    for j, onset in enumerate(candidates[:-1]):
        for offset in candidates[j + 1:]:
            if offset - onset < min_dur:
                continue
            margin = max(int(0.1 * (offset - onset)), 1)
            lo = onset + margin
            hi = offset - margin
            if hi <= lo:
                continue
            baseline_end = max(onset - 10, 1)
            baseline = float(np.median(current[:baseline_end])) if onset > 10 else float(np.median(current[:max(onset, 1)]))
            plateau = float(np.median(current[lo:hi]))
            amplitude = plateau - baseline
            score = abs(amplitude)
            if best is None or score > best[0]:
                best = (score, int(onset), int(offset))

    if best is not None:
        return best[1], best[2]

    onset = int(np.argmax(abs_di))
    if onset + min_dur < len(di):
        offset = onset + min_dur + int(np.argmax(abs_di[onset + min_dur:]))
    else:
        offset = len(current) - 1
    return onset, offset


def detect_steps(recording: Recording) -> None:
    """Detect current-step onset, offset, baseline, and step amplitude."""
    if not recording.sweeps:
        return

    protocol_pulse = _protocol_pulse_parameters(recording.protocol_notes)

    for sweep in recording.sweeps:
        fs = sweep.sampling_rate
        v = sweep.voltage

        if sweep.current is not None and np.ptp(sweep.current) > 5.0:
            i = sweep.current
            onset, offset = _find_step_edges(i, fs)
            if onset is None or offset is None:
                continue

            baseline_i = float(np.median(i[:max(onset - 10, 1)])) if onset > 10 else float(np.median(i[:max(onset, 1)]))
            margin = max(int(0.1 * (offset - onset)), 1)
            lo = onset + margin
            hi = offset - margin
            step_i = float(np.median(i[lo:hi])) if hi > lo else float(np.median(i[onset:offset]))
            baseline_v = float(np.median(v[:max(onset - 10, 1)])) if onset > 10 else float(np.median(v[:int(0.1 * len(v))]))

            sweep.step_onset_idx = onset
            sweep.step_offset_idx = offset
            sweep.baseline_current = baseline_i
            sweep.step_amplitude = step_i - baseline_i
            sweep.baseline_voltage = baseline_v
        else:
            if protocol_pulse is not None:
                onset_ms, width_ms, start_pa, increment_pa = protocol_pulse
                default_onset = int(round(onset_ms / 1000.0 * fs))
                default_offset = int(round((onset_ms + width_ms) / 1000.0 * fs))
                step_amplitude = start_pa + increment_pa * float(sweep.index)
            else:
                default_onset = int(DEFAULT_RHEOBASE_ONSET_FRACTION * len(v))
                default_offset = int(DEFAULT_RHEOBASE_OFFSET_FRACTION * len(v))
                step_amplitude = assumed_rheobase_current_pA(sweep.index)
            default_onset = min(max(default_onset, 0), max(len(v) - 1, 0))
            default_offset = min(max(default_offset, default_onset + 1), len(v))
            sweep.step_onset_idx = default_onset
            sweep.step_offset_idx = default_offset
            sweep.baseline_current = 0.0
            sweep.baseline_voltage = float(np.median(v[:default_onset])) if default_onset > 0 else float(v[0])
            sweep.step_amplitude = step_amplitude
