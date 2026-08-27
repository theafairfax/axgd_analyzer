import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from axgd_io import load_axgd_bytes
from models import Metadata
from properties import compute_phase_plane, compute_properties
from template_matching import TemplateConfig
from test_pulse import compute_test_pulse_properties

@st.cache_data(show_spinner="Loading AxoGraph recording...")
def _load_data(file_bytes,filename,metadata,template_cfg,mode="rheobase"):
    rec=load_axgd_bytes(file_bytes,filename,metadata)
    if mode=="rheobase":rec.properties=compute_properties(rec,template_cfg=template_cfg)
    return rec

def _fmt(x):
    if x is None:return "N/A"
    try:
        if np.isnan(x):return "N/A"
    except TypeError:pass
    if isinstance(x,bool):return "Yes" if x else "No"
    return f"{x:.3f}" if isinstance(x,float) else str(x)

def plot_phase_plane(v,d):
    fig,ax=plt.subplots(figsize=(5,4));ax.plot(v,d,lw=1);ax.set_xlabel("Membrane potential (mV)");ax.set_ylabel("dV/dt (mV/ms = V/s)");ax.grid(alpha=.3);return fig

st.set_page_config(page_title="AxoGraph Patch-Clamp Analyzer",layout="wide")
st.title("AxoGraph (.axgd) Current Clamp Analyzer")
st.caption("Passive properties come from the dedicated test pulse; intrinsic AP/AHP shape uses only APs generated at rheobase.")
with st.sidebar:
    st.header("1. Recordings")
    test_file=st.file_uploader("Test pulse .axgd (usually 001)",type=["axgd","axgx"],key="test")
    rheo_file=st.file_uploader("Step current / rheobase .axgd",type=["axgd","axgx"],key="rheo")
    st.header("2. Metadata")
    cell_id=st.text_input("Cell ID","Cell_01");condition=st.text_input("Condition / treatment","");cell_type=st.text_input("Cell Type","")
    animal_id=st.text_input("Animal ID","");age=st.text_input("DPI / Age","");sex=st.text_input("Sex","");notes=st.text_area("Notes","")
    st.header("3. Test Pulse")
    pulse_mv=st.number_input("Voltage command amplitude (mV)",.1,100.,10.,.5)
    pulse_onset=st.number_input("Expected pulse onset (ms)",0.,500.,20.,1.)
    pulse_width=st.number_input("Expected pulse width (ms)",1.,500.,40.,1.,help="The supplied 0001 Test Pulse recordings show an approximately 20-60 ms pulse.")
    st.header("4. AP Detection")
    tm_threshold=st.number_input("Template detection threshold",.1,20.,1.,.1);tm_min_sep=st.number_input("Min separation (ms)",.1,50.,2.,.1)
    tm_amp_reject=st.number_input("Reject amplitude below (mV)",0.,100.,20.,1.);tm_latency_start=st.number_input("Search from (ms)",0.,1000.,20.,1.)
    tm_latency_end=st.number_input("Search until (ms)",1.,5000.,530.,1.);tm_n_episodes=st.number_input("Episodes pooled for Captured APs",1,100,10,1)
    tm_burst_window=st.number_input("Burst ISI window (ms)",1.,200.,20.,1.);tm_voltage_thresh_pct=st.slider("AP threshold (% max dV/dt)",.01,.5,.10,.01)
cfg=TemplateConfig(threshold=tm_threshold,min_separation_ms=tm_min_sep,amplitude_reject_mv=tm_amp_reject,latency_start_ms=tm_latency_start,
    latency_end_ms=tm_latency_end,n_episodes=int(tm_n_episodes),burst_window_ms=tm_burst_window,voltage_threshold_pct=tm_voltage_thresh_pct)
meta=Metadata(cell_id=cell_id,condition=condition,cell_type=cell_type,animal_id=animal_id,age=age,sex=sex,notes=notes)
if test_file is None or rheo_file is None:st.info("Upload both recordings to create a combined analysis/export.")
else:
    try:
        test_rec=_load_data(test_file.getvalue(),test_file.name,meta,cfg,"test");rheo_rec=_load_data(rheo_file.getvalue(),rheo_file.name,meta,cfg,"rheobase")
        tp=compute_test_pulse_properties(test_rec,pulse_amplitude_mv=pulse_mv,expected_onset_ms=pulse_onset,expected_width_ms=pulse_width);props=rheo_rec.properties
        props.series_resistance=tp.series_resistance;props.membrane_resistance=tp.membrane_resistance;props.membrane_capacitance=tp.membrane_capacitance
        st.success(f"Loaded test pulse: {test_file.name} | rheobase protocol: {rheo_file.name}")
        tab_tp,tab_raw,tab_fi,tab_summary,tab_ap=st.tabs(["Test Pulse QC","Rheobase Traces","F-I & Excitability","Combined Export","AP Dynamics"])
        with tab_tp:
            st.subheader("Dedicated Test-Pulse Passive Properties")
            a,b,c,d=st.columns(4);a.metric("Average Rs",f"{_fmt(tp.series_resistance)} MOhm");b.metric("Average Rm",f"{_fmt(tp.membrane_resistance)} MOhm");c.metric("Average Cm",f"{_fmt(tp.membrane_capacitance)} pF");d.metric("Accepted fits",f"{tp.n_valid_sweeps}/{tp.n_total_sweeps}")
            st.caption("Rs/Rm/Cm are averaged after robust outlier rejection. QC retains every successfully fitted sweep and marks whether it contributed to the average.")
            if tp.sweep_values:st.dataframe(pd.DataFrame(tp.sweep_values),hide_index=True,use_container_width=True)
            else:st.error("No valid test-pulse transients were detected. Verify command amplitude and the expected pulse onset/width.")
        with tab_raw:
            idxs=[s.index for s in rheo_rec.sweeps];selected=st.multiselect("Sweeps",idxs,default=idxs[:min(5,len(idxs))])
            fig,(av,ai)=plt.subplots(2,1,figsize=(10,6),sharex=True)
            for idx in selected:
                s=rheo_rec.sweeps[idx];label=f"Sweep {idx}" if s.step_amplitude is None else f"Sweep {idx} ({s.step_amplitude:.0f} pA)";av.plot(s.time*1000,s.voltage,label=label)
                if s.current is not None:ai.plot(s.time*1000,s.current)
            av.legend(fontsize="small");av.set_ylabel("Voltage (mV)");ai.set_ylabel("Current (pA)");ai.set_xlabel("Time (ms)");st.pyplot(fig);plt.close(fig)
        with tab_fi:
            if props.fi_curve:
                fig,ax=plt.subplots(figsize=(6,4));ax.plot([x[0] for x in props.fi_curve],[x[1] for x in props.fi_curve],marker="o");ax.set_xlabel("Current (pA)");ax.set_ylabel("AP count");ax.grid(alpha=.3);st.pyplot(fig);plt.close(fig)
            x,y,z=st.columns(3);x.metric("Rheobase",_fmt(props.rheobase));y.metric("F-I slope",_fmt(props.fi_slope));z.metric("Max firing rate",_fmt(props.max_firing_rate_hz))
        with tab_summary:
            combined={"Cell":cell_id,"Type":cell_type,"treatment":condition,"DPI/Age":age,"sex":sex};combined.update(props.to_flat_dict());df=pd.DataFrame([combined])
            st.subheader("Combined Test Pulse + Rheobase Output")
            st.caption("SPIKE/AHP Peak columns are event amplitudes (AxoGraph-style), not absolute Vm. They are averaged across all APs generated at rheobase. Absolute Vm remains visible in AP Dynamics.")
            st.dataframe(df,use_container_width=True,hide_index=True);st.download_button("Download combined CSV",df.to_csv(index=False).encode(),file_name=f"{cell_id or 'cell'}_combined_properties.csv",mime="text/csv")
        with tab_ap:
            if props.rheobase is None:st.info("No rheobase AP detected.")
            else:
                rb=next((s for s in rheo_rec.sweeps if np.isclose(s.step_amplitude,props.rheobase,atol=.5)),None);analysis=next((a for a in props.sweep_analyses if rb is not None and a.sweep_index==rb.index),None)
                if rb is not None and analysis is not None:
                    st.subheader(f"Rheobase sweep: {props.rheobase:.1f} pA | {analysis.n_spikes} AP(s)")
                    rows=[]
                    for i,sp in enumerate(analysis.spikes,1):rows.append({"AP":i,"Absolute peak Vm (mV)":sp.peak_voltage,"Threshold Vm (mV)":sp.threshold_voltage,"AP amplitude (mV)":sp.amplitude,"Half-width (ms)":sp.half_width,"Max rise (mV/ms)":sp.max_rise_slope,"Max fall (mV/ms)":sp.max_fall_slope,"Absolute AHP trough Vm (mV)":sp.ahp_voltage,"AHP amplitude from threshold (mV)":(sp.ahp_voltage-sp.threshold_voltage) if sp.ahp_voltage is not None else np.nan})
                    if rows:
                        apdf=pd.DataFrame(rows);st.dataframe(apdf,hide_index=True,use_container_width=True);st.write("**Rheobase AP averages**");st.dataframe(apdf.drop(columns=["AP"]).mean(numeric_only=True).to_frame("Mean").T,hide_index=True,use_container_width=True)
                    v,d=compute_phase_plane(rb.time*1000,rb.voltage);fig=plot_phase_plane(v,d);st.pyplot(fig);plt.close(fig)
    except Exception as err:st.error(f"Error processing recordings: {err}")
