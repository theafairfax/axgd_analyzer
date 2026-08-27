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
class SpikeEventFeatures:
    """Single detected event's waveform, decomposed the way AxoGraph's
    template-matching Events module reports it (see the SPIKE/AHP column
    blocks on the 'Current Clamp - Isaac' summary sheet): a peak, its
    location, an onset, the rise time from onset to peak, the half-amplitude
    width, and a decay time constant. Used for both the SPIKE block (peak =
    the AP peak) and the AHP block (peak = the post-spike trough, i.e. the
    most negative point) -- the field names are shared but "peak" means
    "the extremum this block is built around" in both cases."""
    peak_voltage: float       # mV
    location_ms: float        # time of the peak/trough, relative to step onset
    onset_ms: float           # time the event starts, relative to step onset
    rise_ms: float             # onset -> peak/trough
    width_ms: float            # half-amplitude full width, ms
    decay_ms: float            # decay/recovery time constant, ms


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
    n_events_template: Optional[int] = None          # spikes found by template matching, this sweep


@dataclass
class IntrinsicProperties:
    """Aggregate intrinsic electrophysiological properties for one recording.

    Field groupings mirror the 'Current Clamp - Isaac' summary sheet:
    Rs/Rm/Cm, RMP Pre/Post, Captured Aps, and the SPIKE/AHP waveform blocks
    (``spike_event`` / ``ahp_event``), computed from the first singlet-or-
    doublet response at rheobase (see A9 on the sheet).
    """
    resting_membrane_potential: Optional[float] = None  # mV, legacy overall median RMP
    input_resistance: Optional[float] = None             # MOhm, single-exp Rin (legacy)
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

    # --- spreadsheet-matching fields ---
    series_resistance: Optional[float] = None     # MOhm, Rs (fast component of double-exp fit)
    membrane_resistance: Optional[float] = None    # MOhm, Rm (slow component / steady-state)
    rmp_pre: Optional[float] = None                 # mV, baseline before the step protocol
    rmp_post: Optional[float] = None                # mV, baseline after the step protocol
    captured_aps: Optional[int] = None              # total template-detected spikes, first N episodes
    spike_event: Optional[SpikeEventFeatures] = None
    ahp_event: Optional[SpikeEventFeatures] = None
    has_burst_at_rheobase: Optional[bool] = None
    n_burst_events_at_rheobase: Optional[int] = None

    def to_flat_dict(self) -> dict:
        """Flatten scalar properties for a comparison table / CSV export,
        using the same column names as the 'Current Clamp - Isaac' sheet
        where a metric matches one directly."""
        d = {
            "Rs (MOhm)": self.series_resistance,
            "Rm (MOhm)": self.membrane_resistance,
            "Cm (pF)": self.membrane_capacitance,
            "Captured Aps": self.captured_aps,
            "RMP Pre": self.rmp_pre,
            "RMP post": self.rmp_post,
            "Rheobase (pA)": self.rheobase,
            "Sag ratio": self.sag_ratio,
            "Membrane tau (ms)": self.membrane_tau,
            "Max firing rate (Hz)": self.max_firing_rate_hz,
            "F-I slope (Hz/pA)": self.fi_slope,
            "Adaptation index": self.adaptation_index,
            "Burst at rheobase": self.has_burst_at_rheobase,
        }
        if self.spike_event is not None:
            d.update({
                "Peak Membrane Voltage-1 (mV) [SPIKE]": self.spike_event.peak_voltage,
                "Location  (ms) [SPIKE]": self.spike_event.location_ms,
                "Onset  (ms) [SPIKE]": self.spike_event.onset_ms,
                "Rise  (ms) [SPIKE]": self.spike_event.rise_ms,
                "Width  (ms) [SPIKE]": self.spike_event.width_ms,
                "Decay  (ms) [SPIKE]": self.spike_event.decay_ms,
            })
        if self.ahp_event is not None:
            d.update({
                "Peak Membrane Voltage-1 (mV) [AHP]": self.ahp_event.peak_voltage,
                "Location  (ms) [AHP]": self.ahp_event.location_ms,
                "Onset  (ms) [AHP]": self.ahp_event.onset_ms,
                "Rise  (ms) [AHP]": self.ahp_event.rise_ms,
                "Width  (ms) [AHP]": self.ahp_event.width_ms,
                "Decay  (ms) [AHP]": self.ahp_event.decay_ms,
            })
        return d


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
