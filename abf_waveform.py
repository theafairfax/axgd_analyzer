"""Isolated, manually reviewable ABF event landmarks (all internal times in ms)."""
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

LANDMARKS = {
    "Location": ("#dc2626", "Peak / trough"),
    "Onset": ("#7c3aed", "Onset"),
    "Rise 10%": ("#0891b2", "Rise 10%"),
    "Rise 90%": ("#059669", "Rise 90%"),
    "Half rise": ("#d97706", "Half amplitude: leading edge"),
    "Half fall": ("#db2777", "Half amplitude: recovery edge"),
}


def validate_landmarks(values, lo, hi):
    """Allow censored crossings, but reject out-of-window or reordered markers."""
    if any(np.isfinite(x) and not lo <= x <= hi for x in values.values()):
        return False
    order = [values[k] for k in ("Onset", "Rise 10%", "Half rise", "Rise 90%", "Location", "Half fall")
             if np.isfinite(values[k])]
    return all(a <= b for a, b in zip(order, order[1:]))


def landmark_table(values, tm, voltage):
    origin = values["Location"]
    return pd.DataFrame([
        {"Marker": name, "Sweep time (ms)": value,
         "Relative to location (ms)": value-origin,
         "Vm (mV)": float(np.interp(value, tm, voltage)) if np.isfinite(value) else np.nan}
        for name, value in values.items()
    ])


def render_event_inspector(sweep, rows, prefix):
    if not rows:
        return
    st.subheader("Isolated first AP & AHP")
    st.caption("First detected AP in the selected sweep. Each event has its own location = 0. "
               "Drag the sliders to move colored lines, then apply. Adjustments are manual timing "
               "overrides; automatic detection, protocol averages and their exports remain available below.")
    row = rows[0]
    tm, voltage = np.asarray(sweep["time"])*1000., np.asarray(sweep["voltage"])
    if "_Event end (ms)" not in row:
        st.info("No onset threshold found; adjust detection settings to resolve this AP.")
        return
    # Include settings/results in the key so a new detection cannot reuse stale edits.
    signature = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:12]
    for block, tab in zip(("SPIKE", "AHP"), st.tabs(["AP spike", "AHP trough"])):
        with tab:
            origin = row[f"{block} Location (ms)"]
            if not np.isfinite(origin):
                st.info("AHP trough is unresolved within the analysis window.")
                continue
            lo = row["_Event start (ms)"] if block == "SPIKE" else row["AHP Onset (ms)"]
            hi = row["_Event end (ms)"]
            if block == "SPIKE" and np.isfinite(row["AHP Onset (ms)"]):
                hi = row["AHP Onset (ms)"]
            key = f"{prefix}_{sweep['index']}_{signature}_{block}"
            defaults = {name: row[f"{block} {name} (ms)"] for name in LANDMARKS}
            if st.button("Reset to detected markers", key=key+"_reset"):
                for name in LANDMARKS:
                    st.session_state.pop(key+name, None)
                st.session_state.pop(key+"_reviewed", None)
            with st.expander("Adjust marker times", expanded=False):
                st.caption("Times are absolute milliseconds within the sweep. Missing crossings stay blank; "
                           "increase the detection/search window to resolve them. Rise/width durations use the edited endpoints.")
                with st.form(key+"_form"):
                    edited = {}
                    for name, (_, label) in LANDMARKS.items():
                        value = defaults[name]
                        if np.isfinite(value):
                            edited[name] = st.slider(label+" (ms)", min_value=float(lo), max_value=float(hi),
                                value=float(value), step=float(np.median(np.diff(tm))), format="%.4f", key=key+name)
                        else:
                            edited[name] = np.nan
                            st.caption(label+": unresolved")
                    apply = st.form_submit_button("Apply marker adjustments")
                if apply:
                    if validate_landmarks(edited, lo, hi):
                        st.session_state[key+"_reviewed"] = edited
                    else:
                        st.error("Keep markers ordered: onset ≤ 10% ≤ half-rise ≤ 90% ≤ location ≤ half-fall. Previous valid values are retained.")
            values = st.session_state.get(key+"_reviewed", defaults)
            reviewed = key+"_reviewed" in st.session_state
            origin = values["Location"]
            relative = st.radio("Time axis", ["Relative to location", "Absolute sweep time"],
                                horizontal=True, key=key+"_axis") == "Relative to location"
            offset = origin if relative else 0.
            mask = (tm >= lo) & (tm <= hi)
            fig, ax = plt.subplots(figsize=(11, 4.5))
            try:
                ax.plot(tm[mask]-offset, voltage[mask], color="#1e293b", lw=1.8, label="Recorded Vm")
                for name, (color, label) in LANDMARKS.items():
                    value = values[name]
                    if not np.isfinite(value):
                        continue
                    vm = np.interp(value, tm, voltage)
                    ax.axvline(value-offset, color=color, lw=1.2, ls="--", label=label)
                    ax.scatter(value-offset, vm, color=color, s=35, zorder=5)
                for a, b, color in (("Rise 10%", "Rise 90%", "#059669"), ("Half rise", "Half fall", "#d97706")):
                    if np.isfinite(values[a]) and np.isfinite(values[b]):
                        ax.plot([values[a]-offset, values[b]-offset],
                                np.interp([values[a], values[b]], tm, voltage), color=color, lw=2.5)
                ax.set_xlabel("Time relative to event location (ms)" if relative else "Time within sweep (ms)")
                ax.set_ylabel("Membrane potential (mV)")
                ax.set_title(f'{block}: {"manually reviewed" if reviewed else "automatically detected"}')
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(alpha=.15)
                ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
            finally:
                plt.close(fig)
            table = landmark_table(values, tm, voltage)
            st.dataframe(table, hide_index=True, use_container_width=True)
            rise = values["Rise 90%"]-values["Rise 10%"]
            width = values["Half fall"]-values["Half rise"]
            a, b = st.columns(2)
            a.metric("Rise 10–90% (ms)", f"{rise:.4f}" if np.isfinite(rise) else "Unresolved")
            b.metric("Width at half amplitude (ms)", f"{width:.4f}" if np.isfinite(width) else "Unresolved")
            st.caption("Rise and width are durations and do not change with the time origin. "
                       "AHP uses downward 10–90% crossings from its onset Vm toward the trough; "
                       "its onset is the post-AP return through AP threshold Vm.")
            table.insert(0, "Event", block)
            table.insert(0, "Sweep", sweep["index"])
            table["Measurement source"] = "Manual override" if reviewed else "Automatic"
            table["Rise 10–90% duration (ms)"] = rise
            table["Half-width duration (ms)"] = width
            st.download_button("Download displayed markers CSV", table.to_csv(index=False).encode(),
                               file_name=f"{prefix}_sweep{sweep['index']}_{block}_markers.csv",
                               mime="text/csv", key=key+"_csv")
