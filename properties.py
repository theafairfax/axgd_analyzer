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
from spikes import (
    detect_spikes, adaptation_index as _adaptation_index,
    detect_spikes_template, build_spike_event_features,
    build_ahp_event_features, first_singlet_or_doublet,
)
from template_matching import TemplateConfig


def _exp_decay(t, v_inf, delta, tau):
    return v_inf + delta * np.exp(-t / tau)


def _double_exp_decay(t, v_inf, a_fast, tau_fast, a_slow, tau_slow):
    return v_inf + a_fast * np.exp(-t / tau_fast) + a_slow * np.exp(-t / tau_slow)


def _fit_rs_rm_cm(sweep: Sweep) -> tuple:
    """Fit a two-exponential decay to the voltage response right after step
    onset on a hyperpolarizing, spike-free sweep, separating the fast
    (pipette access resistance, Rs) and slow (membrane Rm/Cm) components of
    the charging transient. Returns (Rs MOhm, Rm MOhm, Cm pF), any of which
    may be None if the fit fails or the step is unusable.

    NOTE: this estimates Rs from the fast component of the step response
    itself, since no dedicated bridge-balance/test-pulse channel is being
    read here. If your rig captures a short test pulse separately, that
    would give a cleaner Rs and this function should be pointed at it
    instead -- flagged for refinement once real recordings are available.
    """
    if (sweep.step_onset_idx is None or sweep.step_offset_idx is None
            or not sweep.step_amplitude):
        return None, None, None
    fs = sweep.sampling_rate
    onset = sweep.step_onset_idx
    fit_len = max(int(0.6 * (sweep.step_offset_idx - onset)), 20)
    lo, hi = onset, min(onset + fit_len, len(sweep.voltage))
    if hi - lo < 20:
        return None, None, None

    t_ms = (np.arange(lo, hi) - onset) / fs * 1000.0
    v = sweep.voltage[lo:hi]
    v0, v_inf_guess = float(v[0]), float(v[-1])
    delta = v0 - v_inf_guess
    p0 = [v_inf_guess, delta * 0.3, 0.3, delta * 0.7, 15.0]
    try:
        popt, _ = curve_fit(
            _double_exp_decay, t_ms, v, p0=p0, maxfev=10000,
            bounds=([-300, -300, 0.01, -300, 0.5], [300, 300, 5.0, 300, 300.0]),
        )
        v_inf, a_fast, tau_fast, a_slow, tau_slow = popt
        if tau_fast > tau_slow:  # keep "fast" faster than "slow"
            a_fast, tau_fast, a_slow, tau_slow = a_slow, tau_slow, a_fast, tau_fast
        i_step = sweep.step_amplitude  # pA
        if not i_step:
            return None, None, None
        rs = abs(a_fast / i_step) * 1000.0   # mV/pA -> MOhm
        rm = abs(a_slow / i_step) * 1000.0
        cm = (tau_slow * 1000.0 / rm) if rm else None  # ms / MOhm -> pF
        return rs, rm, cm
    except Exception:
        return None, None, None


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


def compute_properties(recording: Recording,
                        template_cfg: TemplateConfig = TemplateConfig(),
                        ) -> IntrinsicProperties:
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

    # --- Rs / Rm / Cm: double-exponential fit on the same hyperpolarizing
    #     sweep (fast=access resistance, slow=membrane) ---
    series_resistance = membrane_resistance = None
    if hyperpol_candidates:
        most_hyperpol = min(hyperpol_candidates, key=lambda s: s.step_amplitude)
        rs_fit, rm_fit, cm_fit = _fit_rs_rm_cm(most_hyperpol)
        series_resistance, membrane_resistance = rs_fit, rm_fit
        if cm_fit is not None:
            membrane_capacitance = cm_fit
    if membrane_resistance is None:
        membrane_resistance = input_resistance  # fall back to the steady-state I-V estimate

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

    # --- RMP Pre / Post: baseline before the first sweep's step, and the
    #     tail of the last sweep after its step ends ---
    rmp_pre = recording.sweeps[0].baseline_voltage if recording.sweeps else None
    rmp_post = None
    if recording.sweeps:
        last_sweep = recording.sweeps[-1]
        fs_last = last_sweep.sampling_rate
        if last_sweep.step_offset_idx is not None:
            tail_lo = min(last_sweep.step_offset_idx + int(0.05 * fs_last), len(last_sweep.voltage) - 1)
            tail = last_sweep.voltage[tail_lo:]
            if len(tail):
                rmp_post = float(np.median(tail))
        if rmp_post is None:
            rmp_post = last_sweep.baseline_voltage

    # --- Captured Aps: template-matched spikes pooled across the first
    #     N episodes (default 10), within the configured latency window ---
    template_spike_trains = {}
    captured_aps = 0
    for sweep in recording.sweeps[:template_cfg.n_episodes]:
        tspikes = detect_spikes_template(sweep, template_cfg)
        template_spike_trains[sweep.index] = tspikes
        captured_aps += len(tspikes)

    # --- SPIKE / AHP shape summary + burst-at-rheobase: from the first
    #     singlet-or-doublet response at rheobase, using template-matched
    #     spikes (matches how the summary sheet's SPIKE/AHP columns and
    #     "burst at rheobase" were defined -- notes A9/A10) ---
    spike_event = ahp_event = None
    has_burst = None
    n_burst_events = None
    if rheobase is not None:
        rheobase_sweep = next(
            (s for s in recording.sweeps if s.step_amplitude == rheobase), None
        )
        if rheobase_sweep is not None:
            rheobase_tspikes = template_spike_trains.get(rheobase_sweep.index)
            if rheobase_tspikes is None:
                rheobase_tspikes = detect_spikes_template(rheobase_sweep, template_cfg)
            shape_spikes = first_singlet_or_doublet(rheobase_tspikes, template_cfg.burst_window_ms)
            if shape_spikes:
                v = rheobase_sweep.voltage
                fs_rb = rheobase_sweep.sampling_rate
                onset_idx = rheobase_sweep.step_onset_idx or 0
                spike_event = build_spike_event_features(v, fs_rb, shape_spikes[0], onset_idx)
                ahp_event = build_ahp_event_features(
                    v, fs_rb, shape_spikes[0], onset_idx,
                    baseline_v=rheobase_sweep.baseline_voltage,
                )
            has_burst = (
                len(rheobase_tspikes) >= 2
                and rheobase_tspikes[1].isi_prev_ms is not None
                and rheobase_tspikes[1].isi_prev_ms <= template_cfg.burst_window_ms
            )
            n_burst_events = sum(
                1 for s in rheobase_tspikes
                if s.isi_prev_ms is not None and s.isi_prev_ms <= template_cfg.burst_window_ms
            ) + (1 if rheobase_tspikes else 0)

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
        series_resistance=series_resistance,
        membrane_resistance=membrane_resistance,
        rmp_pre=rmp_pre,
        rmp_post=rmp_post,
        captured_aps=captured_aps,
        spike_event=spike_event,
        ahp_event=ahp_event,
        has_burst_at_rheobase=has_burst,
        n_burst_events_at_rheobase=n_burst_events,
    )


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
