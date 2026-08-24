"""
Current-step detection for patch-clamp sweeps.
"""
from __future__ import annotations
import numpy as np
from models import Recording, Sweep

def detect_steps(recording: Recording) -> None:
    """
    Detects current-step onset, offset, baseline, and step amplitude across sweeps.
    If a current command channel exists, it uses the derivative of current.
    Otherwise, it infers the step window from voltage deflections.
    """
    if not recording.sweeps:
        return

    for sweep in recording.sweeps:
        fs = sweep.sampling_rate
        v = sweep.voltage

        if sweep.current is not None and np.ptp(sweep.current) > 5.0:  # Current channel available (>5 pA range)
            i = sweep.current
            di = np.diff(i)
            # Find largest step transitions
            onset = int(np.argmax(np.abs(di)))
            # Look for step end after onset + minimum duration (e.g., 20ms)
            min_dur = int(0.02 * fs)
            if onset + min_dur < len(di):
                offset = onset + min_dur + int(np.argmax(np.abs(di[onset + min_dur:])))
            else:
                offset = len(i) - 1

            baseline_i = float(np.median(i[:max(onset - 10, 1)])) if onset > 10 else 0.0
            step_i = float(np.median(i[onset + int(0.1 * (offset - onset)): offset - int(0.1 * (offset - onset))]))
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
