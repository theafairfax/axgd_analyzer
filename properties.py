"""Computation of intrinsic electrophysiological properties."""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit
from models import Recording, Sweep, SweepAnalysis, IntrinsicProperties, SpikeEventFeatures
from protocol import detect_steps
from spikes import detect_spikes, adaptation_index as _adaptation_index, detect_spikes_template, build_spike_event_features, build_ahp_event_features
from template_matching import TemplateConfig

def _exp_decay(t,v_inf,delta,tau): return v_inf+delta*np.exp(-t/tau)

def _fit_tau(sweep: Sweep)->Optional[float]:
    if sweep.step_onset_idx is None or sweep.step_offset_idx is None:return None
    fs=sweep.sampling_rate; onset=sweep.step_onset_idx; skip=max(int(.0005*fs),1)
    fit_len=max(int(.6*(sweep.step_offset_idx-onset)),skip+10); lo=onset+skip; hi=min(onset+fit_len,len(sweep.voltage))
    if hi-lo<10:return None
    t=(np.arange(lo,hi)-onset)/fs; v=sweep.voltage[lo:hi]; v0=float(sweep.voltage[onset]); vi=float(v[-1])
    try:
        popt,_=curve_fit(_exp_decay,t,v,p0=[vi,v0-vi,.02],maxfev=5000); tau=float(popt[2])
        return tau*1000 if .0005<tau<1 else None
    except Exception:return None

def _analyze_sweep(sweep:Sweep)->SweepAnalysis:
    spikes=detect_spikes(sweep); duration=sweep.step_duration or len(sweep.voltage)/sweep.sampling_rate
    steady=deflection=peak=sag=None
    if sweep.step_onset_idx is not None and sweep.step_offset_idx is not None and not spikes:
        onset,offset=sweep.step_onset_idx,sweep.step_offset_idx; tail=max(offset-int(.1*(offset-onset)),onset)
        steady=float(np.mean(sweep.voltage[tail:offset])); deflection=steady-(sweep.baseline_voltage or 0.)
        if sweep.step_amplitude is not None and sweep.step_amplitude<0:
            w=sweep.voltage[onset:offset]
            if len(w):
                peak=float(np.min(w)); denom=peak-(sweep.baseline_voltage or 0.)
                if abs(denom)>1e-6:sag=(peak-steady)/denom
    return SweepAnalysis(sweep_index=sweep.index,step_amplitude=sweep.step_amplitude or 0.,n_spikes=len(spikes),
        firing_rate_hz=len(spikes)/duration if duration else 0.,spikes=spikes,steady_state_voltage=steady,
        voltage_deflection=deflection,peak_hyperpolarization=peak,sag_ratio=sag,
        adaptation_index=_adaptation_index(spikes) if len(spikes)>=3 else None)

def _mean_event(events):
    events=[e for e in events if e is not None]
    if not events:return None
    def m(name):
        vals=[getattr(e,name) for e in events if getattr(e,name) is not None and np.isfinite(getattr(e,name))]
        return float(np.mean(vals)) if vals else np.nan
    return SpikeEventFeatures(peak_voltage=m('peak_voltage'),location_ms=m('location_ms'),onset_ms=m('onset_ms'),
        rise_ms=m('rise_ms'),width_ms=m('width_ms'),decay_ms=m('decay_ms'))

def compute_properties(recording:Recording,template_cfg:TemplateConfig=TemplateConfig())->IntrinsicProperties:
    detect_steps(recording); analyses=[_analyze_sweep(s) for s in recording.sweeps]
    baselines=[s.baseline_voltage for s in recording.sweeps if s.baseline_voltage is not None]
    rmp=float(np.median(baselines)) if baselines else None
    sub=[(a.step_amplitude,a.voltage_deflection) for a in analyses if a.n_spikes==0 and a.voltage_deflection is not None and a.step_amplitude]
    rin=None
    if len(sub)>=2:
        cs=np.array([x[0] for x in sub]); ds=np.array([x[1] for x in sub])
        if np.ptp(cs)>0:rin=float(np.polyfit(cs,ds,1)[0])*1000.
    hyper=[s for s,a in zip(recording.sweeps,analyses) if a.n_spikes==0 and s.step_amplitude is not None and s.step_amplitude<0]
    tau=None; sag=None
    if hyper:
        h=min(hyper,key=lambda s:s.step_amplitude); tau=_fit_tau(h); aa=next((a for a in analyses if a.sweep_index==h.index),None); sag=aa.sag_ratio if aa else None
    cm=tau*1000./rin if tau is not None and rin else None
    firing=sorted([a for a in analyses if a.n_spikes>0 and a.step_amplitude>0],key=lambda a:a.step_amplitude)
    rheobase=firing[0].step_amplitude if firing else None
    rb_analysis=firing[0] if firing else None
    rb_spikes=rb_analysis.spikes if rb_analysis else []
    first_thr=float(np.mean([s.threshold_voltage for s in rb_spikes])) if rb_spikes else None
    first_amp=float(np.mean([s.amplitude for s in rb_spikes])) if rb_spikes else None
    first_hw=float(np.mean([s.half_width for s in rb_spikes])) if rb_spikes else None
    fi=sorted([(a.step_amplitude,a.n_spikes) for a in analyses],key=lambda x:x[0])
    max_fr=max((a.firing_rate_hz for a in analyses),default=None); fi_slope=None
    pos=[x for x in fi if x[0]>0]
    if len(pos)>=2:
        cs=np.array([x[0] for x in pos]); ns=np.array([x[1] for x in pos])
        if np.ptp(cs)>0:
            dur=recording.sweeps[0].step_duration if recording.sweeps else None
            if dur:fi_slope=float(np.polyfit(cs,ns,1)[0])/dur
    adaptation=rb_analysis.adaptation_index if rb_analysis and rb_analysis.n_spikes>=3 else None
    rmp_pre=recording.sweeps[0].baseline_voltage if recording.sweeps else None; rmp_post=None
    if recording.sweeps:
        last=recording.sweeps[-1]
        if last.step_offset_idx is not None:
            lo=min(last.step_offset_idx+int(.05*last.sampling_rate),len(last.voltage)-1); tail=last.voltage[lo:]
            if len(tail):rmp_post=float(np.median(tail))
        if rmp_post is None:rmp_post=last.baseline_voltage
    template_trains={}; captured=0
    for sweep in recording.sweeps[:template_cfg.n_episodes]:
        ts=detect_spikes_template(sweep,template_cfg); template_trains[sweep.index]=ts; captured+=len(ts)
    spike_event=ahp_event=None; has_burst=n_burst=None
    if rheobase is not None:
        rb_sweep=next((s for s in recording.sweeps if np.isclose(s.step_amplitude,rheobase,atol=.5)),None)
        if rb_sweep is not None:
            # Intrinsic AP/AHP summary is restricted to ALL APs generated at rheobase.
            # Prefer the robust prominence detector used to define rheobase; template
            # matching is retained for Captured APs and as a fallback.
            shape_spikes=rb_spikes
            if not shape_spikes:shape_spikes=template_trains.get(rb_sweep.index) or detect_spikes_template(rb_sweep,template_cfg)
            onset=rb_sweep.step_onset_idx or 0; v=rb_sweep.voltage; fs=rb_sweep.sampling_rate
            spike_event=_mean_event([build_spike_event_features(v,fs,sp,onset) for sp in shape_spikes])
            ahp_event=_mean_event([build_ahp_event_features(v,fs,sp,onset,baseline_v=rb_sweep.baseline_voltage) for sp in shape_spikes])
            has_burst=any(sp.isi_prev_ms is not None and sp.isi_prev_ms<=template_cfg.burst_window_ms for sp in shape_spikes[1:])
            n_burst=(1+sum(1 for sp in shape_spikes[1:] if sp.isi_prev_ms is not None and sp.isi_prev_ms<=template_cfg.burst_window_ms)) if shape_spikes else 0
    return IntrinsicProperties(resting_membrane_potential=rmp,input_resistance=rin,membrane_tau=tau,membrane_capacitance=cm,
        sag_ratio=sag,rheobase=rheobase,first_spike_threshold=first_thr,first_spike_amplitude=first_amp,first_spike_half_width=first_hw,
        max_firing_rate_hz=max_fr,fi_slope=fi_slope,adaptation_index=adaptation,fi_curve=fi,sweep_analyses=analyses,
        series_resistance=None,membrane_resistance=rin,rmp_pre=rmp_pre,rmp_post=rmp_post,captured_aps=captured,
        spike_event=spike_event,ahp_event=ahp_event,has_burst_at_rheobase=has_burst,n_burst_events_at_rheobase=n_burst)

def compute_phase_plane(time_ms,voltage_mv):
    dt_s=np.gradient(time_ms)/1000.; return voltage_mv,np.gradient(voltage_mv)/dt_s

def analyze_fahp(time_ms,voltage_mv,spike_peak_idx,v_rest,next_spike_idx=None,max_fahp_window_ms=20.):
    dt=time_ms[1]-time_ms[0]; max_pts=int(max_fahp_window_ms/dt); start=spike_peak_idx
    end=min(start+max_pts,next_spike_idx) if next_spike_idx is not None else min(start+max_pts,len(voltage_mv))
    post=voltage_mv[start:end]
    if not len(post):return None
    trough=start+int(np.argmin(post)); vmin=float(voltage_mv[trough])
    return {'v_min':vmin,'fahp_amplitude':float(v_rest-vmin),'fahp_duration_ms':float(time_ms[trough]-time_ms[start]),'trough_idx':trough}
