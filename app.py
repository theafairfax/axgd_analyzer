import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models import Metadata, Recording
from axgd_io import load_axgd_bytes
from properties import compute_properties

st.set_page_config(page_title="AxoGraph Patch-Clamp Analyzer", layout="wide")

st.title("AxoGraph (.axgd) Electrophysiology Analyzer")

# --- Sidebar: Upload & Metadata ---
st.sidebar.header("1. Upload & Metadata")
uploaded_file = st.sidebar.file_uploader("Upload .axgd or .axgx file", type=["axgd", "axgx"])

cell_id = st.sidebar.text_input("Cell ID", value="Cell_01")
condition = st.sidebar.text_input("Condition / Genotype", value="WT")
cell_type = st.sidebar.text_input("Cell Type", value="Pyramidal")
notes = st.sidebar.text_area("Notes", value="")

if uploaded_file is not None:
    # Build metadata
    meta = Metadata(
        cell_id=cell_id,
        condition=condition,
        cell_type=cell_type,
        notes=notes
    )

    # Load and cache recording
    @st.cache_data(show_spinner="Loading AxoGraph recording...")
    def _load_data(file_bytes: bytes, filename: str, metadata: Metadata):
        rec = load_axgd_bytes(file_bytes, filename, metadata)
        rec.properties = compute_properties(rec)
        return rec

    try:
        rec = _load_data(uploaded_file.getvalue(), uploaded_file.name, meta)
        props = rec.properties

        st.sidebar.success(f"Loaded: {rec.display_name}")
        st.sidebar.markdown(f"**Sweeps:** {len(rec.sweeps)}  \n**Sampling Rate:** {rec.sampling_rate / 1000:.1f} kHz")

        # --- Main Layout Tabs ---
        tab_raw, tab_fi, tab_summary = st.tabs(["Raw Sweeps", "F-I Curve & Excitability", "Intrinsic Properties"])

        # 1. Raw Sweeps View
        with tab_raw:
            st.subheader("Voltage & Current Traces")
            sweep_indices = [s.index for s in rec.sweeps]
            selected_sweeps = st.multiselect("Select sweeps to display", sweep_indices, default=sweep_indices[:min(5, len(sweep_indices))])

            if selected_sweeps:
                fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                
                for idx in selected_sweeps:
                    sw = rec.sweeps[idx]
                    ax_v.plot(sw.time, sw.voltage, label=f"Sweep {idx} ({sw.step_amplitude:.0f} pA)" if sw.step_amplitude is not None else f"Sweep {idx}")
                    if sw.current is not None:
                        ax_i.plot(sw.time, sw.current)

                ax_v.set_ylabel("Voltage (mV)")
                ax_v.legend(loc="upper right", fontsize="small")
                ax_v.grid(True, linestyle="--", alpha=0.5)

                ax_i.set_ylabel("Current (pA)")
                ax_i.set_xlabel("Time (s)")
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
                    st.metric("Rheobase", f"{props.rheobase:.1f} pA" if props.rheobase is not None else "N/A")
                    st.metric("F-I Slope", f"{props.fi_slope:.3f} Hz/pA" if props.fi_slope is not None else "N/A")
                    st.metric("Max Firing Rate", f"{props.max_firing_rate_hz:.1f} Hz" if props.max_firing_rate_hz is not None else "N/A")

        # 3. Intrinsic Properties Summary Tab
        with tab_summary:
            st.subheader("Intrinsic Membrane & Action Potential Properties")
            if props:
                flat_props = props.to_flat_dict()
                df_props = pd.DataFrame(list(flat_props.items()), columns=["Property", "Value"])
                df_props["Value"] = df_props["Value"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isnan(x) else ("N/A" if x is None else str(x)))

                st.dataframe(df_props, use_container_width=True, hide_index=True)

                csv = df_props.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Summary CSV",
                    data=csv,
                    file_name=f"{rec.metadata.cell_id or 'cell'}_properties.csv",
                    mime="text/csv"
                )

    except Exception as err:
        st.error(f"Error processing file: {err}")
else:
    st.info("Upload an AxoGraph file (`.axgd` or `.axgx`) in the sidebar to begin analysis.")
