"""ABF current-clamp protocols with explicit channel and timing controls."""
import hashlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from abf_io import (inspect_abf_bytes, load_protocol_sweeps,
                    VOLTAGE_SCALES, CURRENT_SCALES)
from abf_analysis import analyze_events, event_averages, rmp_summary
from abf_waveform import render_event_inspector


@st.cache_data(show_spinner="Reading ABF header...", max_entries=3)
def _inspect(data):
    return inspect_abf_bytes(data)


@st.cache_data(show_spinner="Loading ABF protocol...", max_entries=3)
def _load(data, voltage, source, current):
    return load_protocol_sweeps(data, voltage, source, current)


def _num(label, value, key, minimum=None):
    return st.number_input(label, value=float(value), min_value=minimum, key=key)


def _csv(frame, label, name, key):
    st.download_button(label, frame.to_csv(index=False).encode(),
                       file_name=name, mime="text/csv", key=key)


def _plot(sweep, rows=None, window=None):
    t, v = sweep["time"], sweep["voltage"]
    fig, axes = plt.subplots(2 if sweep["current"] is not None else 1, 1,
                             figsize=(10, 5), squeeze=False, sharex=True)
    try:
        stride = max(1, int(np.ceil(len(t)/30000)))
        ax = axes[0, 0]
        ax.plot(t[::stride]*1000, v[::stride], lw=.7)
        if rows:
            idx = [r["Peak sample"] for r in rows]
            ax.scatter(t[idx]*1000, v[idx], c="red", s=18, label="Detected AP")
            ax.legend()
            for r in rows:
                if np.isfinite(r["AHP Location (ms)"]):
                    ax.scatter(r["AHP Location (ms)"], r["AHP trough Vm (mV)"],
                               c="blue", s=18)
        if window:
            ax.axvspan(window[0]*1000, window[1]*1000, alpha=.1, color="green")
        ax.set_ylabel("Vm (mV)")
        if sweep["current"] is not None:
            axes[1, 0].plot(t[::stride]*1000, sweep["current"][::stride], lw=.7)
            axes[1, 0].set_ylabel("Current / command (pA)")
        axes[-1, 0].set_xlabel("Time within sweep (ms)")
        st.pyplot(fig)
    finally:
        plt.close(fig)


def _dynamics(rows, prefix, filename):
    if not rows:
        st.info("No AP peaks detected in the analysis window. Check the trace, window and detection settings.")
        return
    averages = pd.DataFrame(event_averages(rows))
    st.dataframe(averages, hide_index=True, use_container_width=True)
    detail = pd.DataFrame(rows)
    detail = detail.drop(columns=[c for c in detail if c == "Peak sample" or c.startswith("_")])
    with st.expander("Individual events and measurement status", expanded=True):
        st.dataframe(detail, hide_index=True, use_container_width=True)
    averages.insert(0, "File", filename)
    detail.insert(0, "File", filename)
    _csv(averages, "Download average dynamics CSV", filename+"_averages.csv", prefix+"_avg")
    _csv(detail, "Download individual events CSV", filename+"_events.csv", prefix+"_events")


def _window(sweep, start_ms, end_ms, automatic):
    t, current = sweep["time"], sweep["current"]
    start, end = start_ms/1000., end_ms/1000.
    if automatic and current is not None:
        base = np.median(current[:max(1, int(.02*len(current)))])
        dev = np.abs(current-base)
        if np.max(dev) > 1.:
            active = np.flatnonzero(dev > max(1., .05*np.max(dev)))
            start = float(t[active[0]])
            end = float(t[min(active[-1]+1, len(t)-1)])
    if start < t[0] or end > t[-1] + np.median(np.diff(t)) or start >= end:
        raise ValueError("Analysis window falls outside this sweep. Adjust start/end times.")
    return start, end


def _protocol(kind):
    labels = {"rmp": "RMP monitor (0000)", "step": "Step rheobase (0002)",
              "ramp": "Ramp (0001)"}
    upload = st.file_uploader(labels[kind], type=["abf"], key=kind+"_upload")
    if upload is None:
        return
    data = upload.getvalue()
    prefix = kind+"_"+hashlib.sha256(data).hexdigest()[:12]
    info = _inspect(data)
    st.caption(f'{upload.name} | {info["sweep_count"]} sweeps | {info["sampling_rate_hz"]:g} Hz | {info["protocol"]}')
    channels = [c for c in info["channels"] if c["unit"].strip() in VOLTAGE_SCALES]
    if not channels:
        st.error("No supported voltage channel found. Expected V, mV, or uV units.")
        return
    vm = st.selectbox("Membrane voltage channel", channels,
                      format_func=lambda c: f'{c["channel"]}: {c["name"]} ({c["unit"]})',
                      key=prefix+"_vm")
    source, current = "manual", None
    if kind != "rmp":
        source = st.selectbox("Injected current source", ["manual", "DAC", "ADC"],
                              format_func=lambda s: {"manual": "Entered protocol",
                              "DAC": "ABF command waveform", "ADC": "Recorded current channel"}[s],
                              key=prefix+"_source")
        if source != "manual":
            options = info["commands"] if source == "DAC" else info["channels"]
            options = [c for c in options if c["unit"].strip() in CURRENT_SCALES]
            if not options:
                st.warning("No current source with supported units is available. Use entered protocol.")
                return
            chosen = st.selectbox("Current channel / command", options,
                                  format_func=lambda c: f'{c["channel"]}: {c["name"]} ({c["unit"]})',
                                  key=prefix+"_current")
            current = chosen["channel"]
            st.caption("Command waveforms represent programmed current; recorded ADC current represents measured current.")

    with st.expander("Detection settings"):
        height = _num("Minimum AP peak (mV)", -10., prefix+"_height")
        prominence = _num("Minimum AP prominence (mV)", 20., prefix+"_prom", 0.)
        separation = _num("Minimum AP separation (ms)", 2., prefix+"_sep", .1)
        dvdt = _num("AP onset dV/dt (mV/ms)", 20., prefix+"_dv", .1)
        ahp = _num("Maximum AHP search after AP peak (ms)", 50., prefix+"_ahp", 1.)
    detector = dict(height=height, prominence=prominence, separation_ms=separation)
    event_settings = dict(**detector, dvdt_threshold=dvdt, ahp_window_ms=ahp)

    if kind == "rmp":
        sweep_index = st.selectbox("Monitor sweep (0-based)", range(info["sweep_count"]), key=prefix+"_sweep")
        start = _num("Settling period to exclude (s)", 30., prefix+"_settle", 0.)
        use_end = st.checkbox("Set RMP window end", key=prefix+"_end_enabled")
        end_input = _num("RMP window end (s)", 120., prefix+"_rmp_end", 0.) if use_end else None
    else:
        st.caption("Set the injection window below. Entered-protocol timing defaults are placeholders; verify them against the trace.")
        start_ms = _num("Injection start (ms)", 100., prefix+"_start", 0.)
        end_ms = _num("Injection end (ms)", 600., prefix+"_end", 0.)
        automatic = st.checkbox("Detect injection window from current waveform", value=source != "manual",
                                 disabled=source == "manual", key=prefix+"_auto")
        if kind == "step":
            first = _num("First sweep current (pA)", -100., prefix+"_first")
            increment = _num("Current increment per sweep (pA)", 20., prefix+"_increment")
            st.caption("Entered protocol: sweep 0 = -100 pA; sweep 17 = +240 pA with the supplied defaults.")
        elif source == "manual":
            ramp_start = _num("Ramp current at injection start (pA)", 0., prefix+"_ramp_first")
            ramp_end = _num("Ramp current at injection end (pA)", 300., prefix+"_ramp_last")
            confirmed = st.checkbox("I have verified the entered ramp current and timing", key=prefix+"_confirm")
            if not confirmed:
                st.info("Verify ramp parameters or select an ABF current/command source before calculating rheobase.")
                return
    if not st.checkbox("Analyze recording", key=prefix+"_analyze"):
        return
    sweeps = _load(data, vm["channel"], source, current)
    stem = os.path.splitext(upload.name)[0]
    if kind == "rmp":
        sw = sweeps[sweep_index]
        t, v = sw["time"], sw["voltage"]
        end = end_input if end_input is not None else float(t[-1]+np.median(np.diff(t)))
        if end > t[-1]+np.median(np.diff(t)) or start >= end:
            st.error("RMP window is outside the recording. Reduce settling time or change the end.")
            return
        summary = rmp_summary(t, v, start, end, **detector)
        st.metric("Average RMP", f'{summary["Average RMP (mV)"]:.3f} mV')
        st.dataframe(pd.DataFrame([summary]), hide_index=True)
        st.caption("Arithmetic mean of all samples in the window, including APs in active neurons. No junction-potential correction.")
        _plot(sw, window=(start, end))
        frame = pd.DataFrame([{"File": upload.name, "Sweep": sweep_index, "Start (s)": start,
                               "End (s)": end, **summary}])
        _csv(frame, "Download RMP CSV", stem+"_rmp.csv", prefix+"_rmp_csv")
        return

    if kind == "step":
        summaries, events, windows = [], {}, {}
        for sw in sweeps:
            lo, hi = _window(sw, start_ms, end_ms, automatic)
            rows = analyze_events(sw["time"], sw["voltage"], lo, hi, **event_settings)
            idx = sw["index"]
            events[idx], windows[idx] = rows, (lo, hi)
            if sw["current"] is None:
                injected = first + idx*increment
            else:
                t, cur = sw["time"], sw["current"]
                baseline = cur[t < lo]
                if not len(baseline):
                    raise ValueError("Current delta needs pre-injection baseline; adjust injection start.")
                plateau = cur[(t >= lo+.2*(hi-lo)) & (t < hi-.2*(hi-lo))]
                injected = float(np.median(plateau)-np.median(baseline))
            summaries.append({"Sweep (0-based)": idx, "Injected current (pA)": injected,
                              "AP count": len(rows), "Window start (s)": lo, "Window end (s)": hi,
                              "Current source": source})
        frame = pd.DataFrame(summaries)
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            ordered = frame.sort_values("Injected current (pA)")
            ax.plot(ordered["Injected current (pA)"], ordered["AP count"], "o-")
            ax.set_xlabel("Injected current above baseline (pA)")
            ax.set_ylabel("AP count during injection")
            ax.grid(alpha=.2)
            st.pyplot(fig)
        finally:
            plt.close(fig)
        positive = frame[(frame["Injected current (pA)"] >= 0) & (frame["AP count"] > 0)]
        st.metric("Step rheobase (pA)", f'{positive["Injected current (pA)"].min():.3f}' if len(positive) else "Not detected")
        st.dataframe(frame, hide_index=True, use_container_width=True)
        export = frame.copy()
        export.insert(0, "File", upload.name)
        _csv(export, "Download current–AP count CSV", stem+"_fi.csv", prefix+"_fi_csv")
        idx = st.selectbox("Sweep for average AP/AHP dynamics (0-based)", range(len(sweeps)), key=prefix+"_selected")
        st.write(f"All {len(events[idx])} APs detected during injection in sweep {idx}")
        _plot(sweeps[idx], events[idx], windows[idx])
        render_event_inspector(sweeps[idx], events[idx], prefix)
        _dynamics(events[idx], prefix, stem+"_sweep"+str(idx))
    else:
        idx = st.selectbox("Ramp sweep (0-based)", range(len(sweeps)), key=prefix+"_ramp_sweep")
        sw = sweeps[idx]
        lo, hi = _window(sw, start_ms, end_ms, automatic)
        all_rows = analyze_events(sw["time"], sw["voltage"], lo, hi, **event_settings)
        rows = all_rows[:1]
        _plot(sw, rows, (lo, hi))
        if not rows:
            st.info("No AP detected during the ramp.")
            return
        peak = rows[0]["Peak sample"]
        if sw["current"] is not None:
            rheobase = float(sw["current"][peak])
        else:
            rheobase = ramp_start + (ramp_end-ramp_start)*(sw["time"][peak]-lo)/(hi-lo)
        st.metric("Ramp rheobase: current at first AP peak", f"{rheobase:.3f} pA")
        st.caption("Ramp rheobase is the absolute current at the AP peak; it is not baseline-subtracted.")
        render_event_inspector(sw, rows, prefix)
        _dynamics(rows, prefix, stem+"_ramp_sweep"+str(idx))
        _csv(pd.DataFrame([{"File": upload.name, "Sweep": idx, "Rheobase (pA)": rheobase,
                            "Current source": source}]), "Download ramp rheobase CSV",
             stem+"_ramp_rheobase.csv", prefix+"_rb")
        peak_time = sw["time"][peak]
        next_onset = all_rows[1]["SPIKE Onset (ms)"]/1000. if len(all_rows) > 1 else hi
        if not np.isfinite(next_onset):
            next_onset = sw["time"][all_rows[1]["Peak sample"]]
        phase_end = min(peak_time+ahp/1000., next_onset, hi)
        mask = ((sw["time"] >= max(lo, peak_time-.005)) & (sw["time"] < phase_end))
        t, v = sw["time"][mask], sw["voltage"][mask]
        if len(t) > 2:
            derivative = np.gradient(v, t*1000)
            fig, ax = plt.subplots(figsize=(6, 4))
            try:
                ax.plot(v, derivative)
                ax.set_xlabel("Vm (mV)")
                ax.set_ylabel("dV/dt (mV/ms)")
                ax.set_title("First ramp AP phase plot")
                ax.grid(alpha=.2)
                st.pyplot(fig)
            finally:
                plt.close(fig)
            _csv(pd.DataFrame({"Time (ms)": t*1000, "Vm (mV)": v, "dV/dt (mV/ms)": derivative}),
                 "Download phase plot data CSV", stem+"_phase.csv", prefix+"_phase")


def render_abf_workspace():
    st.title("ABF Current-Clamp Analyzer")
    st.caption("RMP monitoring, step-current excitability, and first-AP ramp analysis.")
    with st.expander("Measurement definitions"):
        st.markdown(
            "Peak voltages are absolute Vm in mV. AP onset is the upward dV/dt threshold "
            "before the peak; rise is the 10–90% amplitude interval and width is full width at half amplitude "
            "above onset Vm. AHP onset is the post-AP return below that same Vm; its peak "
            "is the subsequent voltage minimum. AHP rise is the downward 10–90% amplitude interval and width is "
            "full width at half depth below onset Vm. Locations/onsets are milliseconds "
            "from sweep start; relative marker columns subtract each event’s own location. Rise/width are durations. AHP searches stop at the next AP onset, injection end, "
            "or configured search limit. Unresolved measurements remain blank and are "
            "excluded from metric averages; valid-event counts are shown. "
            "Separate columns report AP/AHP amplitudes relative to threshold. "
            "Plots may be subsampled; detection and statistics use full-resolution data."
        )
    tabs = st.tabs(["Average RMP monitor", "Step rheobase", "Ramp protocol"])
    for tab, kind in zip(tabs, ("rmp", "step", "ramp")):
        with tab:
            try:
                _protocol(kind)
            except Exception as err:
                st.error(f"Could not analyze this recording: {err}")
