"""
Data models used throughout the AxoGraph (.axgd) patch-clamp analyzer.

These are plain dataclasses so they are easy to serialize, cache in
st.session_state, and convert to pandas DataFrames for export.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Metadata:
    """User-entered metadata describing a single recording / cell."""
    cell_id: str = ""
    condition: str = ""       # e.g. genotype, drug treatment, group label
    cell_type: str = ""       # e.g. "L2/3 pyramidal", "fast-spiking interneuron"
    animal_id: str = ""
    age: str = ""
    sex: str = ""
    recording_date: str = ""
    experimenter: str = ""
    notes: str = ""


@dataclass
class Sweep:
    """One episode/sweep from the .axgd file, with the current step detected."""
    index: int
    time: np.ndarray            # s
    voltage: np.ndarray         # mV
    current: Optional[np.ndarray]  # pA, may be None if no current channel
    sampling_rate: float        # Hz

    step_onset_idx: Optional[int] = None
    step_offset_idx: Optional[int] = None
    step_amplitude: Optional[float] = None   # pA, delta from baseline
    baseline_current: Optional[float] = None  # pA
    baseline_voltage: Optional[float] = None  # mV

    @property
    def step_onset_time(self) -> Optional[float]:
        if self.step_onset_idx is None:
            return None
        return self.step_onset_idx / self.sampling_rate

    @property
    def step_offset_time(self) -> Optional[float]:
        if self.step_offset_idx is None:
            return None
        return self.step_offset_idx / self.sampling_rate

    @property
    def step_duration(self) -> Optional[float]:
        if self.step_onset_idx is None or self.step_offset_idx is None:
            return None
        return (self.step_offset_idx - self.step_onset_idx) / self.sampling_rate


@dataclass
class SpikeFeatures:
    """Per-action-potential features."""
    peak_time: float            # s
    peak_voltage: float         # mV
    threshold_voltage: float    # mV
    threshold_time: float       # s
    amplitude: float            # mV, peak - threshold
    half_width: float           # ms
    max_rise_slope: float       # mV/ms
    max_fall_slope: float       # mV/ms
    ahp_voltage: Optional[float] = None   # mV, min voltage after spike
    ahp_time: Optional[float] = None      # s
    isi_prev_ms: Optional[float] = None   # interval from previous spike, ms


@dataclass
class SweepAnalysis:
    """Derived, per-sweep summary (spike train + subthreshold features)."""
    sweep_index: int
    step_amplitude: float           # pA
    n_spikes: int
    firing_rate_hz: float           # spikes / step duration
    spikes: List[SpikeFeatures] = field(default_factory=list)
    steady_state_voltage: Optional[float] = None   # mV, mean over last part of step
    voltage_deflection: Optional[float] = None      # mV, steady_state - baseline
    peak_hyperpolarization: Optional[float] = None  # mV, most negative point during step
    sag_ratio: Optional[float] = None
    adaptation_index: Optional[float] = None


@dataclass
class IntrinsicProperties:
    """Aggregate intrinsic electrophysiological properties for one recording."""
    resting_membrane_potential: Optional[float] = None  # mV
    input_resistance: Optional[float] = None             # MOhm
    membrane_tau: Optional[float] = None                  # ms
    membrane_capacitance: Optional[float] = None          # pF
    sag_ratio: Optional[float] = None
    rheobase: Optional[float] = None                      # pA
    first_spike_threshold: Optional[float] = None         # mV
    first_spike_amplitude: Optional[float] = None         # mV
    first_spike_half_width: Optional[float] = None        # ms
    max_firing_rate_hz: Optional[float] = None
    fi_slope: Optional[float] = None                       # Hz / pA
    adaptation_index: Optional[float] = None
    fi_curve: List[tuple] = field(default_factory=list)    # (current_pA, n_spikes)
    sweep_analyses: List[SweepAnalysis] = field(default_factory=list)

    def to_flat_dict(self) -> dict:
        """Flatten scalar properties for a comparison table / CSV export."""
        return {
            "Resting Vm (mV)": self.resting_membrane_potential,
            "Input resistance (MOhm)": self.input_resistance,
            "Membrane tau (ms)": self.membrane_tau,
            "Membrane capacitance (pF)": self.membrane_capacitance,
            "Sag ratio": self.sag_ratio,
            "Rheobase (pA)": self.rheobase,
            "1st spike threshold (mV)": self.first_spike_threshold,
            "1st spike amplitude (mV)": self.first_spike_amplitude,
            "1st spike half-width (ms)": self.first_spike_half_width,
            "Max firing rate (Hz)": self.max_firing_rate_hz,
            "F-I slope (Hz/pA)": self.fi_slope,
            "Adaptation index": self.adaptation_index,
        }


@dataclass
class Recording:
    """A fully loaded + analyzed .axgd file."""
    filename: str
    metadata: Metadata
    sweeps: List[Sweep]
    sampling_rate: float
    protocol_notes: str = ""
    has_current_channel: bool = True
    properties: Optional[IntrinsicProperties] = None

    @property
    def display_name(self) -> str:
        label = self.metadata.cell_id.strip() or self.filename
        if self.metadata.condition.strip():
            return f"{label} ({self.metadata.condition.strip()})"
        return label
