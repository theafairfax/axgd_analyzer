"""Action-potential detection and waveform feature extraction."""
from __future__ import annotations
from typing import List,Optional
import numpy as np
from scipy.signal import find_peaks
from models import Sweep,SpikeFeatures,SpikeEventFeatures
from template_matching import DEFAULT_TEMPLATE_MS,DEFAULT_TEMPLATE_MV,TemplateConfig,detect_events_by_template

def _interp_cross(x0,y0,x1,y1,target):
    if y1==y0:return float(x0)
    return float(x0+(target-y0)*(x1-x0)/(y1-y0))

def _spike_features_at_peak(v,dv,fs,global_idx,next_global_idx,prev_peak_time,rise_thresh_frac=.10):
    peak_time=global_idx/fs;peak_v=float(v[global_idx]);back=max(int(.005*fs),5);lo=max(global_idx-back,0)
    rise=dv[lo:global_idx+1];max_rel=int(np.argmax(rise));max_rise=float(rise[max_rel]);target=rise_thresh_frac*max_rise
    thr_idx=lo
    for j in range(max_rel,0,-1):
        if rise[j-1]<=target<rise[j]:thr_idx=lo+j-1;break
    threshold_v=float(v[thr_idx]);threshold_t=thr_idx/fs;amplitude=peak_v-threshold_v;half_v=threshold_v+amplitude/2.
    li=global_idx
    while li>thr_idx and v[li]>half_v:li-=1
    ri=global_idx;hi=min(global_idx+max(int(.006*fs),5),len(v)-1)
    while ri<hi and v[ri]>half_v:ri+=1
    lt=_interp_cross(li/fs,v[li],(li+1)/fs,v[li+1],half_v) if li+1<len(v) else li/fs
    rt=_interp_cross((ri-1)/fs,v[ri-1],ri/fs,v[ri],half_v) if ri>0 else ri/fs
    hw=(rt-lt)*1000. if rt>lt else np.nan;max_fall=float(np.min(dv[global_idx:hi+1])) if hi>global_idx else np.nan
    a0=min(global_idx+max(int(.00025*fs),1),len(v)-1);a1=min(global_idx+max(int(.020*fs),2),next_global_idx-1,len(v)-1)
    if a1>a0:
        ai=a0+int(np.argmin(v[a0:a1+1]));ahp_v=float(v[ai]);ahp_t=ai/fs
    else:ahp_v=ahp_t=None
    isi=(peak_time-prev_peak_time)*1000. if prev_peak_time is not None else None
    return SpikeFeatures(peak_time=peak_time,peak_voltage=peak_v,threshold_voltage=threshold_v,threshold_time=threshold_t,
        amplitude=amplitude,half_width=hw,max_rise_slope=max_rise,max_fall_slope=max_fall,ahp_voltage=ahp_v,ahp_time=ahp_t,isi_prev_ms=isi)

def detect_spikes(sweep:Sweep,min_peak_mv=-10.,min_prominence_mv=20.,refractory_ms=1.)->List[SpikeFeatures]:
    v=sweep.voltage;fs=sweep.sampling_rate
    if len(v)==0 or fs==0:return []
    start=sweep.step_onset_idx if sweep.step_onset_idx is not None else 0;end=sweep.step_offset_idx if sweep.step_offset_idx is not None else len(v)-1
    w=v[start:end+1];peaks,_=find_peaks(w,height=min_peak_mv,prominence=min_prominence_mv,distance=max(int(refractory_ms/1000*fs),1))
    if not len(peaks):return []
    dv=np.gradient(v)*fs/1000.;out=[];prev=None
    for k,p in enumerate(peaks):
        gi=start+int(p);nxt=start+int(peaks[k+1]) if k+1<len(peaks) else min(gi+int(.020*fs),end)
        sp=_spike_features_at_peak(v,dv,fs,gi,nxt,prev,.10);prev=sp.peak_time;out.append(sp)
    return out

def detect_spikes_template(sweep:Sweep,cfg:TemplateConfig=TemplateConfig(),template_ms=DEFAULT_TEMPLATE_MS,template_mv=DEFAULT_TEMPLATE_MV)->List[SpikeFeatures]:
    v=sweep.voltage;fs=sweep.sampling_rate
    if len(v)==0 or fs==0 or sweep.step_onset_idx is None:return []
    onset=sweep.step_onset_idx;ss=onset+int(cfg.latency_start_ms/1000*fs);se=min(onset+int(cfg.latency_end_ms/1000*fs),len(v)) if cfg.latency_end_ms is not None else (sweep.step_offset_idx or len(v))
    peaks=detect_events_by_template(v,fs,template_ms,template_mv,cfg,search_start_idx=max(ss,0),search_end_idx=se)
    if not len(peaks):return []
    dv=np.gradient(v)*fs/1000.;out=[];prev=None
    for k,gi0 in enumerate(peaks):
        gi=int(gi0);nxt=int(peaks[k+1]) if k+1<len(peaks) else min(gi+int(.020*fs),se-1)
        sp=_spike_features_at_peak(v,dv,fs,gi,nxt,prev,cfg.voltage_threshold_pct)
        if sp.amplitude>=cfg.amplitude_reject_mv:prev=sp.peak_time;out.append(sp)
    return out

def _cross_after(v,start,end,target,direction):
    end=min(end,len(v)-1)
    for i in range(max(start+1,1),end+1):
        if direction=='down' and v[i-1]>target>=v[i]:return i
        if direction=='up' and v[i-1]<target<=v[i]:return i
    return None

def build_spike_event_features(v,fs,spike,step_onset_idx):
    """Spreadsheet-compatible SPIKE event measurements.

    AxoGraph's Peak-of-detected-event value is an event amplitude, not absolute
    Vm.  For intrinsic AP shape we use the dV/dt threshold as the event baseline,
    so Peak = AP peak - threshold.  Location is relative to the AP threshold,
    matching the sub-ms spreadsheet shape columns rather than latency from the
    beginning of the current step.
    """
    pi=int(round(spike.peak_time*fs));ti=int(round(spike.threshold_time*fs));amp=float(spike.peak_voltage-spike.threshold_voltage)
    decay_cross=_cross_after(v,pi,min(pi+int(.010*fs),len(v)-1),spike.threshold_voltage,'down')
    decay_ms=((decay_cross-pi)/fs*1000.) if decay_cross is not None else np.nan
    return SpikeEventFeatures(peak_voltage=amp,location_ms=(pi-ti)/fs*1000.,onset_ms=0.,
        rise_ms=(pi-ti)/fs*1000.,width_ms=spike.half_width,decay_ms=decay_ms)

def build_ahp_event_features(v,fs,spike,step_onset_idx,baseline_v=None):
    """Spreadsheet-compatible AHP event measurements.

    Peak is the negative AHP amplitude relative to the local pre-AHP baseline,
    not the absolute trough Vm.  We use the AP threshold voltage as that local
    baseline because it is stable for rheobase singlets and avoids contaminating
    AHP amplitude with the depolarizing current-step offset.
    """
    if spike.ahp_voltage is None or spike.ahp_time is None:return None
    pi=int(round(spike.peak_time*fs));tr=int(round(spike.ahp_time*fs));base=float(spike.threshold_voltage)
    oi=_cross_after(v,pi,tr,base,'down')
    if oi is None:oi=pi
    event_amp=float(spike.ahp_voltage-base) # negative, matching spreadsheet AHP Peak
    depth=base-spike.ahp_voltage
    if depth<=0:return None
    half=base-depth/2.;li=tr
    while li>oi and v[li]<half:li-=1
    hi=min(tr+int(.020*fs),len(v)-1);ri=tr
    while ri<hi and v[ri]<half:ri+=1
    # AxoGraph shape locations are relative to event onset.
    location=(tr-oi)/fs*1000.;rise=location;width=(ri-li)/fs*1000.;decay=(ri-tr)/fs*1000.
    return SpikeEventFeatures(peak_voltage=event_amp,location_ms=location,onset_ms=0.,rise_ms=rise,width_ms=width,decay_ms=decay)

def first_singlet_or_doublet(spikes,burst_window_ms=20.):
    if not spikes:return []
    return spikes[:2] if len(spikes)>=2 and spikes[1].isi_prev_ms is not None and spikes[1].isi_prev_ms<=burst_window_ms else spikes[:1]

def adaptation_index(spikes):
    isis=[s.isi_prev_ms for s in spikes if s.isi_prev_ms is not None]
    if len(isis)<2:return None
    vals=[(isis[i+1]-isis[i])/(isis[i+1]+isis[i]) for i in range(len(isis)-1) if isis[i+1]+isis[i]!=0]
    return float(np.mean(vals)) if vals else None
