import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from axgd_io import load_axgd_bytes
from models import Metadata, Recording, Sweep
from properties import analyze_fahp, compute_phase_plane, compute_properties


# --- Helper Functions & Plotting ---
@st.cache_data(show_spinner="Loading AxoGraph recording...")
def _load_data(file_bytes: bytes, filename: str, metadata: Metadata) -> Recording:
    rec = load_axgd_bytes(file_bytes, filename, metadata)
    rec.properties = compute_properties(rec)
    return rec


def plot_phase_plane(voltage_mv, dv_dt, spike_ranges=None, xlim=None):
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
    
    # Pass-through parameter for calculation (e.g., threshold or filter cutoff)
    calc_param = st.number_input(
        "Calculation Parameter (e.g. Sampling Rate / Threshold)",
        min_value=1.0,
        value=10000.0,
        step=100.0
    )
    
    st.subheader("Phase Plot mV Axis")
    manual_phase_range = st.checkbox("Manual mV Range for Phase Plot", value=False)
    
    if manual_phase_range:
        phase_vmin = st.number_input("Phase V Min (mV)", value=-90.0, step=5.0)
        phase_vmax = st.number_input("Phase V Max (mV)", value=50.0, step=5.0)
        phase_xlim = (phase_vmin, phase_vmax)
    else:
        phase_xlim = None

def run_calculations(raw_time_sec, raw_voltage_mv, param):
    """
    Downstream processing using the passed sidebar parameter.
    Converts time axis from seconds to milliseconds.
    """
    # Convert seconds -> milliseconds
    time_ms = raw_time_sec * 1000.0
    
    # Calculate dV/dt (V/s or mV/ms)
    # Using sampling interval from converted time
    dt_ms = np.gradient(time_ms)
    dv_dt = np.gradient(raw_voltage_mv) / (dt_ms / 1000.0)  # in V/s or mV/ms depending on scaling
    
    # Example computation using the passed sidebar input
    metrics = {
        "max_dvdt": np.max(dv_dt),
        "min_dvdt": np.min(dv_dt),
        "adjusted_stat": np.mean(raw_voltage_mv) * (param / 1000.0)
    }
    
    return time_ms, raw_voltage_mv, dv_dt, metrics

if uploaded_file is not None:
    meta = Metadata(
        cell_id=cell_id,
        condition=condition,
        cell_type=cell_type,
        notes=notes,
    )

    try:
        rec = _load_data(uploaded_file.getvalue(), uploaded_file.name, meta)
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
                    ax_v.plot(sw.time, sw.voltage, label=label)
                    if sw.current is not None:
                        ax_i.plot(sw.time, sw.current)

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
            if props and props.fi_curve:
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

        # 3. Intrinsic Properties Summary Tab
        with tab_summary:
            st.subheader("Intrinsic Membrane & Action Potential Properties")
            if props:
                flat_props = props.to_flat_dict()
                df_props = pd.DataFrame(
                    list(flat_props.items()), columns=["Property", "Value"]
                )
                df_props["Value"] = df_props["Value"].apply(
                    lambda x: f"{x:.2f}"
                    if isinstance(x, (int, float)) and not np.isnan(x)
                    else ("N/A" if x is None else str(x))
                )

                st.dataframe(df_props, use_container_width=True, hide_index=True)

                csv = df_props.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Summary CSV",
                    data=csv,
                    file_name=f"{rec.metadata.cell_id or 'cell'}_properties.csv",
                    mime="text/csv",
                )

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
                else np.mean(sweep_voltage[:100])
            )

            # Retrieve detected spikes if available on the sweep model
            first_spike_idx = getattr(target_sweep, "first_spike_index", None)
            second_spike_idx = getattr(target_sweep, "second_spike_index", None)

            col1, col2 = st.columns(2)

            with col1:
                v_trace, dvdt_trace = compute_phase_plane(sweep_time, sweep_voltage)
                fig_phase = plot_phase_plane(v_trace, dvdt_trace, xlim=phase_xlim)
                st.pyplot(fig_phase)
                plt.close(fig_phase)

            with col2:
                if first_spike_idx is not None:
                    fahp_results = analyze_fahp(
                        time_ms=sweep_time * 1000.0,
                        voltage_mv=sweep_voltage,
                        spike_peak_idx=first_spike_idx,
                        v_rest=baseline_vrest,
                        next_spike_idx=second_spike_idx,
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
                            "Spike excluded: Subsequent spike within 20 ms or no valid trough detected."
                        )
                else:
                    st.info("No action potential peak detected in the selected sweep.")

    except Exception as err:
        st.error(f"Error processing file: {err}")
else:
    st.info("Upload an AxoGraph file (`.axgd` or `.axgx`) in the sidebar to begin analysis.")
