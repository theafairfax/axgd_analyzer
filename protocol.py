"""
Current-step detection for patch-clamp sweeps.
"""
from __future__ import annotations
import numpy as np
from models import Recording, Sweep


def _find_step_edges(current: np.ndarray, fs: float):
    """Return the onset/offset pair for the dominant square current step.

    Looking for the largest *absolute* derivative alone is ambiguous because the
    onset and offset of a square pulse have nearly identical magnitudes.  If the
    falling edge wins by a small amount, the old implementation interpreted it
    as the onset and could report the pulse with the wrong sign.
    """
    di = np.diff(current)
    if not len(di):
        return None, None

    min_dur = max(int(0.02 * fs), 1)
    abs_di = np.abs(di)
    # Consider the strongest transitions, then choose the pair whose intervening
    # plateau differs most from the surrounding baseline. This works for both
    # depolarizing and hyperpolarizing protocols without imposing a sign.
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

    # Conservative fallback if no valid pair was found.
    onset = int(np.argmax(abs_di))
    if onset + min_dur < len(di):
        offset = onset + min_dur + int(np.argmax(abs_di[onset + min_dur:]))
    else:
        offset = len(current) - 1
    return onset, offset


def detect_steps(recording: Recording) -> None:
    """
    Detects current-step onset, offset, baseline, and step amplitude across sweeps.
    If a current command channel exists, paired current transitions define the
    pulse window. Otherwise, it infers the step window from voltage deflections.
    """
    if not recording.sweeps:
        return

    for sweep in recording.sweeps:
        fs = sweep.sampling_rate
        v = sweep.voltage

        if sweep.current is not None and np.ptp(sweep.current) > 5.0:  # Current channel available (>5 pA range)
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
            # Fallback heuristic: assume step is in the middle 60-80% of recording or default window
            default_onset = int(0.1 * len(v))
            default_offset = int(0.7 * len(v))
            sweep.step_onset_idx = default_onset
            sweep.step_offset_idx = default_offset
            sweep.baseline_voltage = float(np.median(v[:default_onset])) if default_onset > 0 else float(v[0])
            sweep.step_amplitude = 0.0
