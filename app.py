import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from axgd_io import load_axgd_bytes
from models import Metadata, Recording, Sweep
from properties import analyze_fahp, compute_phase_plane, compute_properties
from template_matching import TemplateConfig


# --- Helper Functions & Plotting ---
@st.cache_data(show_spinner="Loading AxoGraph recording...")
def _load_data(file_bytes: bytes, filename: str, metadata: Metadata,
                template_cfg: TemplateConfig) -> Recording:
    rec = load_axgd_bytes(file_bytes, filename, metadata)
    rec.properties = compute_properties(rec, template_cfg=template_cfg)
    return rec


def plot_phase_plane(voltage_mv, dv_dt, xlim=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(voltage_mv, dv_dt, color="crimson", lw=1.2)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel(r"Membrane Potential ($V$, mV)")
    ax.set_ylabel(r"$\mathrm{d}V/\mathrm{d}t$ (V/s)")
    ax.set_title("Action Potential Phase Plot")
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax.grid(alpha=0.3)
    return fig


# --- App Configuration ---
st.set_page_config(page_title="AxoGraph Patch-Clamp Analyzer", layout="wide")
st.title("AxoGraph (.axgd) Electrophysiology Analyzer")

# --- Sidebar: Upload & Metadata ---
st.sidebar.header("1. Upload & Metadata")
uploaded_file = st.sidebar.file_uploader(
    "Upload .axgd or .axgx file", type=["axgd", "axgx"]
)

cell_id = st.sidebar.text_input("Cell ID", value="Cell_01")
condition = st.sidebar.text_input("Condition / Genotype", value="WT")
cell_type = st.sidebar.text_input("Cell Type", value="Pyramidal")
notes = st.sidebar.text_area("Notes", value="")

with st.sidebar:
    st.header("Plot & Calculation Settings")
    
    ahp_window_ms = st.number_input(
        "AHP Search Window (ms)",
        min_value=1.0,
        max_value=200.0,
        value=20.0,
        step=1.0
    )
    
    st.subheader("Phase Plot mV Axis")
    manual_phase_range = st.checkbox("Manual mV Range for Phase Plot", value=False)
    
    if manual_phase_range:
        phase_vmin = st.number_input("Phase V Min (mV)", value=-90.0, step=5.0)
        phase_vmax = st.number_input("Phase V Max (mV)", value=50.0, step=5.0)
        phase_xlim = (phase_vmin, phase_vmax)
    else:
        phase_xlim = None

with st.sidebar:
    st.header("2. Event Detection (Template Matching)")
    st.caption(
        "Mirrors AxoGraph's Events module. A built-in AP template "
        "('0001 Bursty Template Function') is used by default -- no need "
        "to upload a template file."
    )
    tm_threshold = st.number_input(
        "Detection threshold (scale/SE)", min_value=0.1, max_value=20.0,
        value=1.0, step=0.1,
        help="Detection-criterion cutoff. Lower = more sensitive.",
    )
    tm_min_sep = st.number_input(
        "Min separation (ms)", min_value=0.1, max_value=50.0, value=2.0, step=0.1,
    )
    tm_amp_reject = st.number_input(
        "Reject amplitude below (mV)", min_value=0.0, max_value=100.0,
        value=20.0, step=1.0,
    )
    tm_latency_start = st.number_input(
        "Search from (ms after step onset)", min_value=0.0, max_value=1000.0,
        value=20.0, step=1.0,
    )
    tm_latency_end = st.number_input(
        "Search until (ms after step onset)", min_value=1.0, max_value=5000.0,
        value=530.0, step=1.0,
    )
    tm_n_episodes = st.number_input(
        "Episodes pooled for 'Captured Aps'", min_value=1, max_value=100,
        value=10, step=1,
    )
    tm_burst_window = st.number_input(
        "Burst ISI window (ms)", min_value=1.0, max_value=200.0,
        value=20.0, step=1.0,
        help="2+ events within this window, at rheobase, counts as a burst.",
    )
    tm_voltage_thresh_pct = st.slider(
        "AP threshold (% of max dV/dt)", min_value=0.01, max_value=0.5,
        value=0.10, step=0.01,
    )
    st.caption(
        "Note: template baseline/length (currently fixed at ~1 ms / ~4 ms, "
        "matching the original AxoGraph template) aren't yet independently "
        "adjustable here -- upload a custom template in a future pass to "
        "change the AP shape itself."
    )

template_cfg = TemplateConfig(
    threshold=tm_threshold,
    min_separation_ms=tm_min_sep,
    amplitude_reject_mv=tm_amp_reject,
    latency_start_ms=tm_latency_start,
    latency_end_ms=tm_latency_end,
    n_episodes=int(tm_n_episodes),
    burst_window_ms=tm_burst_window,
    voltage_threshold_pct=tm_voltage_thresh_pct,
)

if uploaded_file is not None:
    meta = Metadata(
        cell_id=cell_id,
        condition=condition,
        cell_type=cell_type,
        notes=notes,
    )

    try:
        rec = _load_data(uploaded_file.getvalue(), uploaded_file.name, meta, template_cfg)
        props = rec.properties

        st.sidebar.success(f"Loaded: {rec.display_name}")
        st.sidebar.markdown(
            f"**Sweeps:** {len(rec.sweeps)}  \n"
            f"**Sampling Rate:** {rec.sampling_rate / 1000:.1f} kHz"
        )

        # --- Main Layout Tabs ---
        tab_raw, tab_fi, tab_summary, tab_ap = st.tabs(
            [
                "Raw Sweeps",
                "F-I Curve & Excitability",
                "Intrinsic Properties",
                "Action Potential Dynamics",
            ]
        )

        # 1. Raw Sweeps View
        with tab_raw:
            st.subheader("Voltage & Current Traces")
            sweep_indices = [s.index for s in rec.sweeps]
            selected_sweeps = st.multiselect(
                "Select sweeps to display",
                sweep_indices,
                default=sweep_indices[: min(5, len(sweep_indices))],
            )

            if selected_sweeps:
                fig, (ax_v, ax_i) = plt.subplots(
                    2,
                    1,
                    figsize=(10, 6),
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                )

                for idx in selected_sweeps:
                    sw = rec.sweeps[idx]
                    label = (
                        f"Sweep {idx} ({sw.step_amplitude:.0f} pA)"
                        if sw.step_amplitude is not None
                        else f"Sweep {idx}"
                    )
                    # Convert seconds -> ms for plotting
                    time_ms_trace = sw.time * 1000.0
                    ax_v.plot(time_ms_trace, sw.voltage, label=label)
                    if sw.current is not None:
                        ax_i.plot(time_ms_trace, sw.current)

                ax_v.set_ylabel("Voltage (mV)")
                ax_v.legend(loc="upper right", fontsize="small")
                ax_v.grid(True, linestyle="--", alpha=0.5)

                ax_i.set_ylabel("Current (pA)")
                ax_i.set_xlabel("Time (ms)")
                ax_i.grid(True, linestyle="--", alpha=0.5)

                st.pyplot(fig)
                plt.close(fig)

        # 2. F-I Curve Tab
        with tab_fi:
            st.subheader("Firing Frequency vs. Injected Current (F-I)")
            if props is not None and len(props.fi_curve) > 0:
                currents = [pt[0] for pt in props.fi_curve]
                spikes = [pt[1] for pt in props.fi_curve]

                col1, col2 = st.columns([2, 1])
                with col1:
                    fig_fi, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(currents, spikes, marker="o", color="crimson", lw=1.5)
                    ax.set_xlabel("Injected Current (pA)")
                    ax.set_ylabel("Action Potential Count")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig_fi)
                    plt.close(fig_fi)

                with col2:
                    rheobase_str = (
                        f"{props.rheobase:.1f} pA"
                        if props.rheobase is not None
                        else "N/A"
                    )
                    fi_slope_str = (
                        f"{props.fi_slope:.3f} Hz/pA"
                        if props.fi_slope is not None
                        else "N/A"
                    )
                    max_fr_str = (
                        f"{props.max_firing_rate_hz:.1f} Hz"
                        if props.max_firing_rate_hz is not None
                        else "N/A"
                    )

                    st.metric("Rheobase", rheobase_str)
                    st.metric("F-I Slope", fi_slope_str)
                    st.metric("Max Firing Rate", max_fr_str)
            else:
                st.info("No F-I curve data could be generated for this recording.")

        # 3. Intrinsic Properties Summary Tab
        with tab_summary:
            st.subheader("Intrinsic Membrane & Action Potential Properties")
            if props is not None:
                flat_props = props.to_flat_dict()
                df_props = pd.DataFrame(
                    list(flat_props.items()), columns=["Property", "Value"]
                )
                def _fmt(x):
                    if x is None:
                        return "N/A"
                    if isinstance(x, bool):
                        return "Yes" if x else "No"
                    if isinstance(x, (int, float)):
                        try:
                            if np.isnan(x):
                                return "N/A"
                        except TypeError:
                            pass
                        return f"{x:.2f}" if isinstance(x, float) else str(x)
                    return str(x)

                df_props["Value"] = df_props["Value"].apply(_fmt)

                st.dataframe(df_props, use_container_width=True, hide_index=True)

                csv = df_props.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Summary CSV",
                    data=csv,
                    file_name=f"{rec.metadata.cell_id or 'cell'}_properties.csv",
                    mime="text/csv",
                )
            else:
                st.info("No intrinsic properties available.")

        # 4. Action Potential Dynamics (Phase Plane & fAHP)
        with tab_ap:
            st.subheader("Fast AHP & Phase Plane Analysis")

            sweep_idx = st.selectbox(
                "Select Sweep for AP Analysis",
                options=[s.index for s in rec.sweeps],
                index=0,
            )
            target_sweep = rec.sweeps[sweep_idx]

            sweep_time = target_sweep.time
            sweep_voltage = target_sweep.voltage
            baseline_vrest = (
                props.resting_membrane_potential
                if props and props.resting_membrane_potential is not None
                else float(np.mean(sweep_voltage[:100]))
            )

            # Retrieve detected spikes from sweep_analyses
            analysis = props.sweep_analyses[sweep_idx] if props and sweep_idx < len(props.sweep_analyses) else None
            detected_spikes = analysis.spikes if analysis else []

            col1, col2 = st.columns(2)

            with col1:
                v_trace, dvdt_trace = compute_phase_plane(sweep_time * 1000.0, sweep_voltage)
                fig_phase = plot_phase_plane(v_trace, dvdt_trace, xlim=phase_xlim)
                st.pyplot(fig_phase)
                plt.close(fig_phase)

            with col2:
                if len(detected_spikes) > 0:
                    first_spike = detected_spikes[0]
                    first_spike_idx = int(first_spike.peak_time * target_sweep.sampling_rate)
                    second_spike_idx = int(detected_spikes[1].peak_time * target_sweep.sampling_rate) if len(detected_spikes) > 1 else None

                    fahp_results = analyze_fahp(
                        time_ms=sweep_time * 1000.0,
                        voltage_mv=sweep_voltage,
                        spike_peak_idx=first_spike_idx,
                        v_rest=baseline_vrest,
                        next_spike_idx=second_spike_idx,
                        max_fahp_window_ms=ahp_window_ms,
                    )

                    if fahp_results:
                        st.metric(
                            "fAHP Amplitude",
                            f"{fahp_results['fahp_amplitude']:.2f} mV",
                        )
                        st.metric(
                            "Time to Minimum",
                            f"{fahp_results['fahp_duration_ms']:.2f} ms",
                        )
                        st.metric(
                            "Trough Voltage (V_min)",
                            f"{fahp_results['v_min']:.2f} mV",
                        )
                    else:
                        st.warning(
                            f"Spike excluded: Subsequent spike within {ahp_window_ms:.0f} ms or no valid trough detected."
                        )
                else:
                    st.info("No action potentials detected in this sweep.")

    except Exception as err:
        st.error(f"Error processing file: {err}")
else:
    st.info("Upload an AxoGraph file (`.axgd` or `.axgx`) in the sidebar to begin analysis.")
