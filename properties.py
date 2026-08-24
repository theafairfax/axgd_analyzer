"""
Computation of intrinsic electrophysiological properties from a current-step
Recording: resting Vm, input resistance, membrane tau, sag ratio, rheobase,
spike waveform features, F-I curve/slope, and spike-frequency adaptation.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.optimize import curve_fit

from models import Recording, Sweep, SweepAnalysis, IntrinsicProperties
from protocol import detect_steps
from spikes import detect_spikes, adaptation_index as _adaptation_index


def _exp_decay(t, v_inf, delta, tau):
    return v_inf + delta * np.exp(-t / tau)


def _fit_tau(sweep: Sweep) -> Optional[float]:
    """Fit a single exponential to the voltage response just after step
    onset, returning tau in ms. Uses whichever subthreshold sweep is passed
    in (ideally the most hyperpolarizing one for a clean, spike-free fit)."""
    if sweep.step_onset_idx is None or sweep.step_offset_idx is None:
        return None
    fs = sweep.sampling_rate
    onset = sweep.step_onset_idx
    # fit over first ~60% of the step, skipping the first ~0.5 ms (capacitive spike)
    skip = max(int(0.0005 * fs), 1)
    fit_len = int(0.6 * (sweep.step_offset_idx - onset))
    fit_len = max(fit_len, skip + 10)
    lo = onset + skip
    hi = min(onset + fit_len, len(sweep.voltage))
    if hi - lo < 10:
        return None

    t = (np.arange(lo, hi) - onset) / fs
    v = sweep.voltage[lo:hi]
    v0 = float(sweep.voltage[onset])
    v_inf_guess = float(v[-1])
    delta_guess = v0 - v_inf_guess
    tau_guess = 0.02  # 20 ms

    try:
        popt, _ = curve_fit(
            _exp_decay, t, v,
            p0=[v_inf_guess, delta_guess, tau_guess],
            maxfev=5000,
        )
        tau = float(popt[2])
        if 0.0005 < tau < 1.0:  # sanity bounds: 0.5 ms - 1 s
            return tau * 1000.0  # ms
    except Exception:
        return None
    return None


def _analyze_sweep(sweep: Sweep) -> SweepAnalysis:
    spikes = detect_spikes(sweep)
    duration = sweep.step_duration or (len(sweep.voltage) / sweep.sampling_rate)
    firing_rate = len(spikes) / duration if duration else 0.0

    steady_state = None
    deflection = None
    peak_hyperpol = None
    sag_ratio = None

    if sweep.step_onset_idx is not None and sweep.step_offset_idx is not None and not spikes:
        onset, offset = sweep.step_onset_idx, sweep.step_offset_idx
        fs = sweep.sampling_rate
        tail_start = max(offset - int(0.1 * (offset - onset)), onset)
        steady_state = float(np.mean(sweep.voltage[tail_start:offset]))
        deflection = steady_state - (sweep.baseline_voltage or 0.0)

        if sweep.step_amplitude is not None and sweep.step_amplitude < 0:
            window = sweep.voltage[onset:offset]
            if len(window):
                peak_hyperpol = float(np.min(window))
                baseline_v = sweep.baseline_voltage or 0.0
                denom = (peak_hyperpol - baseline_v)
                if abs(denom) > 1e-6:
                    sag_ratio = (peak_hyperpol - steady_state) / denom

    return SweepAnalysis(
        sweep_index=sweep.index,
        step_amplitude=sweep.step_amplitude or 0.0,
        n_spikes=len(spikes),
        firing_rate_hz=firing_rate,
        spikes=spikes,
        steady_state_voltage=steady_state,
        voltage_deflection=deflection,
        peak_hyperpolarization=peak_hyperpol,
        sag_ratio=sag_ratio,
        adaptation_index=_adaptation_index(spikes) if len(spikes) >= 3 else None,
    )


def compute_properties(recording: Recording) -> IntrinsicProperties:
    detect_steps(recording)

    sweep_analyses = [_analyze_sweep(s) for s in recording.sweeps]
    recording_by_idx = {s.index: s for s in recording.sweeps}

    # --- Resting membrane potential: median baseline across all sweeps ---
    baselines = [s.baseline_voltage for s in recording.sweeps if s.baseline_voltage is not None]
    rmp = float(np.median(baselines)) if baselines else None

    # --- Input resistance: linear fit of voltage deflection vs current,
    #     using only subthreshold sweeps (no spikes) ---
    subthreshold = [
        (a.step_amplitude, a.voltage_deflection)
        for a in sweep_analyses
        if a.n_spikes == 0 and a.voltage_deflection is not None and a.step_amplitude
    ]
    input_resistance = None
    if len(subthreshold) >= 2:
        currents = np.array([c for c, _ in subthreshold])
        deflections = np.array([d for _, d in subthreshold])
        if np.ptp(currents) > 0:
            slope, _intercept = np.polyfit(currents, deflections, 1)
            # slope is in mV/pA. mV/pA = (1e-3 V)/(1e-12 A) = 1e9 Ohm = 1000 MOhm.
            input_resistance = float(slope) * 1000.0  # MOhm

    # --- Membrane tau: fit on the most hyperpolarizing subthreshold sweep ---
    membrane_tau = None
    hyperpol_candidates = [
        s for s, a in zip(recording.sweeps, sweep_analyses)
        if a.n_spikes == 0 and s.step_amplitude is not None and s.step_amplitude < 0
    ]
    if hyperpol_candidates:
        most_hyperpol = min(hyperpol_candidates, key=lambda s: s.step_amplitude)
        membrane_tau = _fit_tau(most_hyperpol)

    membrane_capacitance = None
    if membrane_tau is not None and input_resistance:
        # tau(ms) = Rin(MOhm) * C(pF) / 1000  =>  C = tau(ms)*1000/Rin(MOhm)... derive carefully:
        # tau (s) = R (Ohm) * C (F). R(MOhm)=R*1e6, C(pF)=C*1e-12
        # tau(ms) = R(MOhm)*1e6 * C(pF)*1e-12 * 1000 = R(MOhm)*C(pF)*1e-3
        # => C(pF) = tau(ms) / (R(MOhm) * 1e-3) = tau(ms)*1000 / R(MOhm)
        if input_resistance != 0:
            membrane_capacitance = membrane_tau * 1000.0 / input_resistance

    # --- Sag ratio: from the same most-hyperpolarizing sweep ---
    sag_ratio = None
    if hyperpol_candidates:
        most_hyperpol = min(hyperpol_candidates, key=lambda s: s.step_amplitude)
        match = next((a for a in sweep_analyses if a.sweep_index == most_hyperpol.index), None)
        if match:
            sag_ratio = match.sag_ratio

    # --- Rheobase: smallest positive step current that elicited >=1 spike ---
    firing_sweeps = sorted(
        [a for a in sweep_analyses if a.n_spikes > 0 and a.step_amplitude > 0],
        key=lambda a: a.step_amplitude,
    )
    rheobase = firing_sweeps[0].step_amplitude if firing_sweeps else None

    # --- First spike (at rheobase) properties ---
    first_spike_threshold = first_spike_amplitude = first_spike_half_width = None
    if firing_sweeps:
        first_spike = firing_sweeps[0].spikes[0]
        first_spike_threshold = first_spike.threshold_voltage
        first_spike_amplitude = first_spike.amplitude
        first_spike_half_width = first_spike.half_width

    # --- F-I curve & slope ---
    fi_curve = sorted(
        [(a.step_amplitude, a.n_spikes) for a in sweep_analyses if a.step_amplitude is not None],
        key=lambda x: x[0],
    )
    max_firing_rate = max((a.firing_rate_hz for a in sweep_analyses), default=None)
    fi_slope = None
    positive_points = [(c, n) for c, n in fi_curve if c > 0]
    if len(positive_points) >= 2:
        cs = np.array([c for c, _ in positive_points])
        ns = np.array([n for _, n in positive_points])
        if np.ptp(cs) > 0:
            slope, _ = np.polyfit(cs, ns, 1)
            step_duration = recording.sweeps[0].step_duration if recording.sweeps else None
            if step_duration:
                fi_slope = float(slope) / step_duration  # (spikes/pA) / s = Hz/pA

    # --- Adaptation index: from the sweep with the most spikes ---
    # --- Adaptation index: from the sweep with the most spikes ---
    adaptation = None
    richest = max(sweep_analyses, key=lambda a: a.n_spikes, default=None)
    if richest and richest.n_spikes >= 3:
        adaptation = richest.adaptation_index

def compute_phase_plane(time_ms, voltage_mv):
    """
    Computes dV/dt (V/s or mV/ms) against Voltage.
    """
    dt_s = np.gradient(time_ms) / 1000.0  # Convert ms to seconds
    dv_dt = np.gradient(voltage_mv) / dt_s  # dV/dt in V/s (or mV/ms)
    return voltage_mv, dv_dt


def analyze_fahp(time_ms, voltage_mv, spike_peak_idx, v_rest, next_spike_idx=None, max_fahp_window_ms=20.0):
    """
    Calculates fAHP amplitude (V_rest - V_min) ensuring no intervening spike within the window.
    """
    dt = time_ms[1] - time_ms[0]
    max_pts = int(max_fahp_window_ms / dt)
    
    # Define search window for trough
    start_idx = spike_peak_idx
    if next_spike_idx is not None:
        end_idx = min(start_idx + max_pts, next_spike_idx)
        # Check uninterrupted condition: exclude if subsequent spike occurs < 20 ms
        if (time_ms[next_spike_idx] - time_ms[spike_peak_idx]) < max_fahp_window_ms:
            return None
    else:
        end_idx = min(start_idx + max_pts, len(voltage_mv))
        
    post_spike_v = voltage_mv[start_idx:end_idx]
    if len(post_spike_v) == 0:
        return None
        
    trough_idx = start_idx + np.argmin(post_spike_v)
    v_min = voltage_mv[trough_idx]
    fahp_amplitude = v_rest - v_min  # Depolarization/Hyperpolarization relative to baseline
    fahp_duration = time_ms[trough_idx] - time_ms[spike_peak_idx]

    return {
        "v_min": v_min,
        "fahp_amplitude": fahp_amplitude,
        "fahp_duration_ms": fahp_duration,
        "trough_idx": trough_idx
    }

    return IntrinsicProperties(
        resting_membrane_potential=rmp,
        input_resistance=input_resistance,
        membrane_tau=membrane_tau,
        membrane_capacitance=membrane_capacitance,
        sag_ratio=sag_ratio,
        rheobase=rheobase,
        first_spike_threshold=first_spike_threshold,
        first_spike_amplitude=first_spike_amplitude,
        first_spike_half_width=first_spike_half_width,
        max_firing_rate_hz=max_firing_rate,
        fi_slope=fi_slope,
        adaptation_index=adaptation,
        fi_curve=fi_curve,
        sweep_analyses=sweep_analyses,
    )
