"""Initial ABF workspace: inspect native channels before protocol analysis."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from abf_io import inspect_abf_bytes, load_abf_traces, summarize_traces


@st.cache_data(show_spinner="Reading ABF header...", max_entries=1)
def _inspect(file_bytes):
    return inspect_abf_bytes(file_bytes)


@st.cache_data(show_spinner="Loading selected ABF traces...", max_entries=1)
def _traces(file_bytes, sweeps, channels):
    return load_abf_traces(file_bytes, sweeps, channels)


def render_abf_workspace():
    st.title("ABF Electrophysiology Analyzer")
    st.caption("Initial ABF workspace: inspect recorded channels and sweeps, view signals, and export data.")
    with st.sidebar:
        st.header("1. ABF recording")
        upload = st.file_uploader("ClampX / pCLAMP recording", type=["abf"], key="abf_upload")
        st.caption("ABF1 and ABF2. Channel values retain the units stored in the file.")
    if upload is None:
        st.info("Upload an .abf recording to begin.")
        return
    file_bytes = upload.getvalue()
    try:
        info = _inspect(file_bytes)
    except Exception as err:
        st.error(f"Could not read ABF recording: {err}")
        return

    a, b, c = st.columns(3)
    a.metric("Sweeps", info["sweep_count"])
    b.metric("Channels", len(info["channels"]))
    c.metric("Sampling rate", f'{info["sampling_rate_hz"]:g} Hz')
    st.caption(f'ABF {info["version"]} | Protocol: {info["protocol"]} | Recorded: {info["recorded"]}')
    st.dataframe(pd.DataFrame(info["channels"]), hide_index=True, use_container_width=True)
    with st.sidebar:
        st.header("2. Trace selection")
        channel_map = {ch["channel"]: ch for ch in info["channels"]}
        # Widget state is reset by Streamlit when options change between files.
        channels = st.multiselect(
            "Recorded channels", list(channel_map), default=list(channel_map)[:1],
            format_func=lambda ch: f'{ch}: {channel_map[ch]["name"]} ({channel_map[ch]["unit"]})',
        )
        start = st.number_input("First sweep (0-based)", min_value=0,
                                max_value=max(0, info["sweep_count"] - 1), value=0, step=1)
        count = st.number_input("Number of sweeps", min_value=1,
                                max_value=min(20, info["sweep_count"] - int(start)), value=1, step=1)
        st.caption("Select up to 20 sweeps at a time. pyABF loads the recording into memory when traces are requested.")
        load = st.checkbox("Load selected traces", value=False, key="abf_load_traces")
    if not channels or not load:
        st.info("Select channels and sweeps, then enable Load selected traces.")
        return
    try:
        traces = _traces(file_bytes, tuple(range(int(start), int(start + count))), tuple(channels))
    except Exception as err:
        st.error(f"Could not load ABF traces: {err}")
        return

    st.subheader("Recorded signals")
    for channel in channels:
        selected = [t for t in traces if t.channel == channel]
        fig, ax = plt.subplots(figsize=(10, 3))
        try:
            for trace in selected:
                # Bound plot size only; statistics and exports retain every sample.
                stride = max(1, int(np.ceil(len(trace.time) / 20000)))
                ax.plot(trace.time[::stride], trace.values[::stride],
                        lw=0.8, label=f"Sweep {trace.sweep}")
            ax.set_xlabel("Time within sweep (s)")
            ax.set_ylabel(f'{channel_map[channel]["name"]} ({channel_map[channel]["unit"]})')
            ax.legend(fontsize="small")
            ax.grid(alpha=0.2)
            st.pyplot(fig)
        finally:
            plt.close(fig)
    st.caption("Long traces are subsampled for display; brief events may be hidden. Statistics and CSV exports use all samples.")
    st.subheader("Selected sweep statistics")
    summary = pd.DataFrame(summarize_traces(traces))
    summary.insert(0, "File", upload.name)
    st.dataframe(summary, hide_index=True, use_container_width=True)
    stem = os.path.splitext(upload.name)[0]
    st.download_button("Download sweep statistics CSV", summary.to_csv(index=False).encode(),
                       file_name=f"{stem}_abf_summary.csv", mime="text/csv")
    st.subheader("Export a full-resolution trace")
    index = st.selectbox("Trace", range(len(traces)),
                         format_func=lambda i: f"Sweep {traces[i].sweep} | {traces[i].name} | channel {traces[i].channel}")
    trace = traces[index]
    # Build a potentially large export only when explicitly requested.
    if st.button("Prepare trace CSV"):
        frame = pd.DataFrame({"Time (s)": trace.time, f"{trace.name} ({trace.unit})": trace.values})
        st.download_button("Download trace CSV", frame.to_csv(index=False).encode(),
                           file_name=f"{stem}_sweep{trace.sweep}_ch{trace.channel}.csv",
                           mime="text/csv")
    st.info("This first ABF version provides recorded-signal inspection and descriptive statistics. "
            "Rheobase, AP/AHP, and passive membrane calculations will require ABF protocol and channel mapping.")
