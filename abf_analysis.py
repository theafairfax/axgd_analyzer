"""Protocol-independent AP/AHP measurements; times are sweep-relative ms."""
import numpy as np
from scipy.signal import find_peaks


def crossing(t, v, level, start, stop, rising):
    for j in range(max(0, start), min(stop, len(v) - 1)):
        if (v[j] <= level < v[j+1]) if rising else (v[j] >= level > v[j+1]):
            return float(t[j] + (t[j+1]-t[j]) * (level-v[j])/(v[j+1]-v[j]))
    return np.nan


def detect_peaks(t, v, height=-10., prominence=20., separation_ms=2.):
    dt = float(np.median(np.diff(t)))
    return find_peaks(v, height=height, prominence=prominence,
                      distance=max(1, int(round(separation_ms / (dt*1000)))))[0]


def analyze_events(t, v, start_s, end_s, height=-10., prominence=20.,
                   separation_ms=2., dvdt_threshold=20., ahp_window_ms=50.):
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(t) < 3 or len(t) != len(v) or not np.all(np.isfinite(v)):
        raise ValueError("Voltage must contain at least three finite samples.")
    dt = float(np.median(np.diff(t)))
    if dt <= 0 or start_s >= end_s:
        raise ValueError("Use increasing time and a nonempty analysis window.")
    tm = t * 1000.
    dv = np.gradient(v, tm)
    peaks = detect_peaks(t, v, height, prominence, separation_ms)
    # Determine thresholds for all events so AHP windows stop before the next AP.
    onsets = []
    previous = -1
    for peak in peaks:
        lo = max(previous + 1, peak - max(2, int(.005/dt)), 0)
        upstroke = lo + int(np.argmax(dv[lo:peak+1]))
        below = np.flatnonzero(dv[lo:upstroke+1] < dvdt_threshold)
        onset = lo + int(below[-1]) + 1 if len(below) and dv[upstroke] >= dvdt_threshold else None
        onsets.append(onset)
        previous = peak
    rows = []
    for n, peak in enumerate(peaks):
        if not start_s <= t[peak] < end_s:
            continue
        onset = onsets[n]
        row = {"AP": len(rows)+1, "Peak sample": int(peak),
               "SPIKE Peak Membrane Voltage-1 (mV)": float(v[peak]),
               "SPIKE Location (ms)": float(tm[peak]),
               "SPIKE Onset (ms)": np.nan, "SPIKE Rise (ms)": np.nan,
               "SPIKE Width (ms)": np.nan,
               "AHP Peak Membrane Voltage-1 (mV)": np.nan,
               "AHP Location (ms)": np.nan, "AHP Onset (ms)": np.nan,
               "AHP Rise (ms)": np.nan, "AHP Width (ms)": np.nan,
               "Threshold Vm (mV)": np.nan, "AP amplitude (mV)": np.nan,
               "AHP amplitude from threshold (mV)": np.nan,
               "AHP trough Vm (mV)": np.nan, "Status": ""}
        if onset is None:
            row["Status"] = "No dV/dt threshold crossing"
            rows.append(row)
            continue
        baseline = float(v[onset])
        next_start = (onsets[n+1] if onsets[n+1] is not None else int(peaks[n+1])) if n+1 < len(peaks) else len(v)-1
        # Bound AP width and AHP by next AP, search window, and analysis end.
        stop = min(next_start, int(np.searchsorted(t, end_s)), len(v)-1,
                   peak + max(2, int(ahp_window_ms/1000/dt)))
        half = baseline + (v[peak]-baseline)/2.
        left = crossing(tm, v, half, onset, peak, True)
        right = crossing(tm, v, half, peak, stop, False)
        row.update({"SPIKE Onset (ms)": float(tm[onset]),
                    "SPIKE Rise (ms)": float(tm[peak]-tm[onset]),
                    "SPIKE Width (ms)": right-left,
                    "Threshold Vm (mV)": baseline,
                    "AP amplitude (mV)": float(v[peak]-baseline)})
        # AHP starts when the repolarizing AP crosses back below its threshold Vm.
        ahp_on = crossing(tm, v, baseline, peak, stop, False)
        if not np.isfinite(ahp_on):
            row["Status"] = "No post-AP threshold return in AHP window"
        else:
            first = int(np.searchsorted(tm, ahp_on))
            if stop <= first:
                row["Status"] = "AHP window truncated"
            else:
                trough = first + int(np.argmin(v[first:stop+1]))
                if trough >= stop or v[trough] >= baseline:
                    row["Status"] = "No resolved AHP trough before window boundary"
                else:
                    ahalf = baseline + (v[trough]-baseline)/2.
                    down = crossing(tm, v, ahalf, max(peak, first-1), trough, False)
                    up = crossing(tm, v, ahalf, trough, stop, True)
                    row.update({
                        "AHP Peak Membrane Voltage-1 (mV)": float(v[trough]),
                        "AHP Location (ms)": float(tm[trough]),
                        "AHP Onset (ms)": ahp_on,
                        "AHP Rise (ms)": float(tm[trough]-ahp_on),
                        "AHP Width (ms)": up-down,
                        "AHP amplitude from threshold (mV)": float(v[trough]-baseline),
                        "AHP trough Vm (mV)": float(v[trough])})
                    row["Status"] = "OK" if np.isfinite(up) else "AHP half-recovery truncated; width unavailable"
        rows.append(row)
    return rows


def event_averages(rows):
    keys = [f"{block} {metric}" for block in ("SPIKE", "AHP")
            for metric in ("Peak Membrane Voltage-1 (mV)", "Location (ms)",
                           "Onset (ms)", "Rise (ms)", "Width (ms)")]
    result = []
    for key in keys:
        values = np.array([r[key] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result.append({"Metric": key, "Mean": float(np.mean(finite)) if len(finite) else np.nan,
                       "Valid events": len(finite), "Detected APs": len(rows)})
    return result


def rmp_summary(t, v, start, end, **detector):
    mask = (t >= start) & (t < end)
    if np.sum(mask) < 3:
        raise ValueError("The RMP window has fewer than three samples. Adjust settling time/end.")
    times, values = t[mask], v[mask]
    peaks = detect_peaks(times, values, **detector)
    duration = float(times[-1]-times[0]+np.median(np.diff(times)))
    return {"Average RMP (mV)": float(np.mean(values)),
            "SD (mV)": float(np.std(values)),
            "Window duration (s)": duration, "AP count": len(peaks),
            "Spiking frequency (Hz)": len(peaks)/duration}
