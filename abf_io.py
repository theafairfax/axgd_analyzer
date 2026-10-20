"""ABF1/ABF2 input independent of the AxoGraph protocol assumptions.

pyABF handles binary parsing and amplifier scaling. Values retain their ADC
units; no baseline subtraction, current-step assumptions, or ADC/DAC mixing.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import os
import tempfile

import numpy as np


@dataclass
class ABFTrace:
    sweep: int
    channel: int
    name: str
    unit: str
    time: np.ndarray
    values: np.ndarray


@contextmanager
def open_abf(file_bytes):
    """Give pyABF a closed temporary file (also works on Windows)."""
    if file_bytes[:4] not in (b"ABF ", b"ABF2"):
        raise ValueError("This file does not have a valid ABF1 or ABF2 header.")
    import pyabf

    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".abf", delete=False) as handle:
            path = handle.name
            handle.write(file_bytes)
        # Header inspection does not allocate the complete recording array.
        yield pyabf.ABF(path, loadData=False, cacheStimulusFiles=False)
    finally:
        if path is not None:
            os.remove(path)


def describe_abf(abf):
    return {
        "version": abf.abfVersionString,
        "sampling_rate_hz": float(abf.dataRate),
        "sweep_count": int(abf.sweepCount),
        "protocol": str(abf.protocol),
        "recorded": str(abf.abfDateTimeString),
        "commands": [
            {"channel": int(ch), "name": abf.dacNames[ch], "unit": abf.dacUnits[ch]}
            for ch in abf.channelList if ch < len(abf.dacNames)
        ],
        "channels": [
            {"channel": int(ch), "name": abf.adcNames[ch], "unit": abf.adcUnits[ch]}
            for ch in abf.channelList
        ],
    }


def inspect_abf_bytes(file_bytes):
    with open_abf(file_bytes) as abf:
        return describe_abf(abf)


def load_abf_traces(file_bytes, sweeps, channels):
    """Return only requested traces, copied so pyABF buffers can be freed.

    setSweep supports variable sweep lengths and gives sweep-relative seconds.
    pyABF currently loads its full scaled array on the first setSweep call.
    """
    with open_abf(file_bytes) as abf:
        if any(s not in abf.sweepList for s in sweeps):
            raise ValueError("Selected sweep is not present in this recording.")
        if any(c not in abf.channelList for c in channels):
            raise ValueError("Selected channel is not present in this recording.")
        traces = []
        for sweep in sweeps:
            for channel in channels:
                abf.setSweep(sweep, channel=channel, absoluteTime=False)
                traces.append(ABFTrace(
                    sweep=int(sweep), channel=int(channel),
                    name=abf.adcNames[channel], unit=abf.adcUnits[channel],
                    time=np.array(abf.sweepX, dtype=np.float64, copy=True),
                    values=np.array(abf.sweepY, copy=True),
                ))
        return traces


def summarize_traces(traces):
    """Per-sweep/channel statistics, without assigning physiological meaning."""
    rows = []
    for trace in traces:
        finite = trace.values[np.isfinite(trace.values)]
        rows.append({
            "Sweep (0-based)": trace.sweep, "Channel (0-based)": trace.channel,
            "Channel name": trace.name, "Unit": trace.unit,
            "Samples": len(trace.values),
            "Finite samples": len(finite),
            "Last sample time (s)": float(trace.time[-1]) if len(trace.time) else None,
            "Mean": float(np.mean(finite)) if len(finite) else None,
            "Minimum": float(np.min(finite)) if len(finite) else None,
            "Maximum": float(np.max(finite)) if len(finite) else None,
            "Peak-to-peak": float(np.ptp(finite)) if len(finite) else None,
        })
    return rows


VOLTAGE_SCALES = {"V": 1000., "mV": 1., "uV": .001, "µV": .001, "μV": .001}
CURRENT_SCALES = {"A": 1e12, "mA": 1e9, "uA": 1e6, "µA": 1e6, "μA": 1e6, "nA": 1e3, "pA": 1.}


def load_protocol_sweeps(file_bytes, voltage_channel, current_source="manual", current_channel=None):
    """Read measured Vm in mV and optional ADC current or DAC command in pA.

    A DAC command is explicitly selected and remains identified as a command.
    A missing/unsupported command raises rather than substituting a waveform.
    """
    with open_abf(file_bytes) as abf:
        unit = abf.adcUnits[voltage_channel].strip()
        if unit not in VOLTAGE_SCALES:
            raise ValueError("Select a voltage channel with V, mV, or uV units.")
        sweeps = []
        for index in abf.sweepList:
            abf.setSweep(index, channel=voltage_channel)
            t = np.array(abf.sweepX, copy=True)
            voltage = np.array(abf.sweepY, dtype=float, copy=True) * VOLTAGE_SCALES[unit]
            current = None
            if current_source == "ADC":
                abf.setSweep(index, channel=current_channel)
                iunit = abf.adcUnits[current_channel].strip()
                if iunit not in CURRENT_SCALES:
                    raise ValueError("Selected ADC current channel has unsupported units.")
                current = np.array(abf.sweepY, dtype=float, copy=True) * CURRENT_SCALES[iunit]
            elif current_source == "DAC":
                # pyABF associates sweepC with the currently selected channel's DAC.
                abf.setSweep(index, channel=current_channel)
                iunit = str(abf.sweepUnitsC).strip()
                if iunit not in CURRENT_SCALES:
                    raise ValueError("Selected command is not in current units; select another source.")
                current = np.array(abf.sweepC, dtype=float, copy=True) * CURRENT_SCALES[iunit]
            if current is not None and (len(current) != len(t) or not np.all(np.isfinite(current))):
                raise ValueError("Current waveform is missing, nonfinite, or differs from voltage length.")
            sweeps.append({"index": int(index), "time": t, "voltage": voltage, "current": current})
        return sweeps
