"""Computation of intrinsic electrophysiological properties."""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit
from models import Recording,Sweep,SweepAnalysis,IntrinsicProperties,SpikeEventFeatures
from protocol import detect_steps
from spikes import adaptation_index as _adaptation_index,detect_spikes_template,build_spike_event_features,build_ahp_event_features,build_average_rheobase_event_features
from template_matching import TemplateConfig

def _exp_decay(t,v_inf,delta,tau):return v_inf+delta*np.exp(-t/tau)
def _fit_tau(s):
    if s.step_onset_idx is None or s.step_offset_idx is None:return None
    fs=s.sampling_rate;o=s.step_onset_idx;lo=o+max(int(.0005*fs),1);hi=min(o+max(int(.6*(s.step_offset_idx-o)),10),len(s.voltage))
    if hi-lo<10:return None
    t=(np.arange(lo,hi)-o)/fs;v=s.voltage[lo:hi]
    try:
        p,_=curve_fit(_exp_decay,t,v,p0=[float(v[-1]),float(s.voltage[o]-v[-1]),.02],maxfev=5000);tau=float(p[2]);return tau*1000 if .0005<tau<1 else None
    except:return None

def _analyze_sweep(s,template_cfg):
    spikes=detect_spikes_template(s,template_cfg);dur=s.step_duration or len(s.voltage)/s.sampling_rate;steady=defl=peak=sag=None
    if s.step_onset_idx is not None and s.step_offset_idx is not None and not spikes:
        o,e=s.step_onset_idx,s.step_offset_idx;tail=max(e-int(.1*(e-o)),o);steady=float(np.mean(s.voltage[tail:e]));defl=steady-(s.baseline_voltage or 0.)
        if s.step_amplitude is not None and s.step_amplitude<0:
            w=s.voltage[o:e];peak=float(np.min(w));den=peak-(s.baseline_voltage or 0.);sag=(peak-steady)/den if abs(den)>1e-6 else None
    return SweepAnalysis(sweep_index=s.index,step_amplitude=s.step_amplitude or 0.,n_spikes=len(spikes),firing_rate_hz=len(spikes)/dur if dur else 0.,spikes=spikes,steady_state_voltage=steady,voltage_deflection=defl,peak_hyperpolarization=peak,sag_ratio=sag,adaptation_index=_adaptation_index(spikes) if len(spikes)>=3 else None)
def _mean_event(es):
    es=[e for e in es if e is not None]
    if not es:return None
    def m(n):
        x=[getattr(e,n) for e in es if getattr(e,n) is not None and np.isfinite(getattr(e,n))];return float(np.mean(x)) if x else np.nan
    return SpikeEventFeatures(peak_voltage=m('peak_voltage'),location_ms=m('location_ms'),onset_ms=m('onset_ms'),rise_ms=m('rise_ms'),width_ms=m('width_ms'),decay_ms=m('decay_ms'))
def compute_properties(recording:Recording,template_cfg:TemplateConfig=TemplateConfig())->IntrinsicProperties:
    detect_steps(recording);aa=[_analyze_sweep(s,template_cfg) for s in recording.sweeps]
    bs=[s.baseline_voltage for s in recording.sweeps if s.baseline_voltage is not None];rmp=float(np.median(bs)) if bs else None
    sub=[(a.step_amplitude,a.voltage_deflection) for a in aa if a.n_spikes==0 and a.voltage_deflection is not None and a.step_amplitude]
    rin=None
    if len(sub)>=2:
        c=np.array([x[0] for x in sub]);d=np.array([x[1] for x in sub]);rin=float(np.polyfit(c,d,1)[0])*1000 if np.ptp(c)>0 else None
    hyp=[s for s,a in zip(recording.sweeps,aa) if a.n_spikes==0 and s.step_amplitude is not None and s.step_amplitude<0];tau=sag=None
    if hyp:
        h=min(hyp,key=lambda s:s.step_amplitude);tau=_fit_tau(h);q=next((a for a in aa if a.sweep_index==h.index),None);sag=q.sag_ratio if q else None
    firing=sorted([a for a in aa if a.n_spikes>0 and a.step_amplitude>0],key=lambda a:a.step_amplitude);rb=firing[0] if firing else None;rheo=rb.step_amplitude if rb else None;spikes=rb.spikes if rb else []
    thr=float(np.mean([s.threshold_voltage for s in spikes])) if spikes else None;amp=float(np.mean([s.amplitude for s in spikes])) if spikes else None;hw=float(np.mean([s.half_width for s in spikes])) if spikes else None
    fi=sorted([(a.step_amplitude,a.n_spikes) for a in aa],key=lambda x:x[0]);maxfr=max((a.firing_rate_hz for a in aa),default=None);slope=None;pos=[x for x in fi if x[0]>0]
    if len(pos)>=2 and recording.sweeps and recording.sweeps[0].step_duration:
        c=np.array([x[0] for x in pos]);n=np.array([x[1] for x in pos]);slope=float(np.polyfit(c,n,1)[0])/recording.sweeps[0].step_duration if np.ptp(c)>0 else None
    pre=recording.sweeps[0].baseline_voltage if recording.sweeps else None;post=None
    if recording.sweeps:
        last=recording.sweeps[-1]
        if last.step_offset_idx is not None:
            tail=last.voltage[min(last.step_offset_idx+int(.05*last.sampling_rate),len(last.voltage)):];post=float(np.median(tail)) if len(tail) else last.baseline_voltage
    captured=sum(a.n_spikes for a in aa[:template_cfg.n_episodes])
    sev=aev=None;burst=nburst=None
    if rb:
        rs=next((s for s in recording.sweeps if s.index==rb.sweep_index),None)
        if rs is not None:
            # AxoGraph's captured-event workflow operates on the waveform made
            # from the detected rheobase events. Reproduce that order: capture
            # each detected AP, align on its peak, average the voltage traces,
            # then measure SPIKE and AHP shapes on that single mean waveform.
            # This is intentionally different from averaging per-AP shape
            # measurements, which is nonlinear for rise/width/decay crossings.
            sev,aev=build_average_rheobase_event_features(rs.voltage,rs.sampling_rate,rb.spikes)
            burst=any(x.isi_prev_ms is not None and x.isi_prev_ms<=template_cfg.burst_window_ms for x in rb.spikes[1:]);nburst=(1+sum(x.isi_prev_ms is not None and x.isi_prev_ms<=template_cfg.burst_window_ms for x in rb.spikes[1:])) if rb.spikes else 0
    return IntrinsicProperties(resting_membrane_potential=rmp,input_resistance=rin,membrane_tau=tau,membrane_capacitance=(tau*1000/rin if tau is not None and rin else None),sag_ratio=sag,rheobase=rheo,first_spike_threshold=thr,first_spike_amplitude=amp,first_spike_half_width=hw,max_firing_rate_hz=maxfr,fi_slope=slope,adaptation_index=rb.adaptation_index if rb and rb.n_spikes>=3 else None,fi_curve=fi,sweep_analyses=aa,series_resistance=None,membrane_resistance=rin,rmp_pre=pre,rmp_post=post,captured_aps=captured,spike_event=sev,ahp_event=aev,has_burst_at_rheobase=burst,n_burst_events_at_rheobase=nburst)
def compute_phase_plane(time_ms,voltage_mv):return voltage_mv,np.gradient(voltage_mv)/np.gradient(time_ms)
def analyze_fahp(time_ms,voltage_mv,spike_peak_idx,v_rest,next_spike_idx=None,max_fahp_window_ms=20.):
    dt=time_ms[1]-time_ms[0];end=min(spike_peak_idx+int(max_fahp_window_ms/dt),next_spike_idx if next_spike_idx is not None else len(voltage_mv),len(voltage_mv));post=voltage_mv[spike_peak_idx:end]
    if not len(post):return None
    tr=spike_peak_idx+int(np.argmin(post));vm=float(voltage_mv[tr]);return {'v_min':vm,'fahp_amplitude':float(v_rest-vm),'fahp_duration_ms':float(time_ms[tr]-time_ms[spike_peak_idx]),'trough_idx':tr}
