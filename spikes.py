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

def _crossing_time(v,fs,start,end,target,direction):
    """Interpolated crossing time in seconds between sample indices start/end."""
    start=max(int(start),0);end=min(int(end),len(v)-1)
    if end<=start:return None
    if direction=='up':
        for i in range(start+1,end+1):
            if v[i-1] <= target < v[i]:
                return _interp_cross((i-1)/fs,v[i-1],i/fs,v[i],target)
    else:
        for i in range(start+1,end+1):
            if v[i-1] >= target > v[i]:
                return _interp_cross((i-1)/fs,v[i-1],i/fs,v[i],target)
    return None

def _axograph_shape(v,fs,peak_idx,baseline,polarity=1,left_limit=None,right_limit=None):
    """AxoGraph Find/Measure Peaks and Shapes geometry.

    Settings reproduced from the user's AxoGraph dialog:
      onset at 5% of peak
      rise from 10% to 90% of peak
      width at 50% of peak
      decay from 100% to 50% of peak

    `polarity` is +1 for APs and -1 for negative-going AHPs. Times are
    interpolated between samples. Location and onset are expressed relative to
    the event's 5%-onset time, matching AxoGraph's event-coordinate convention.
    """
    peak_idx=int(peak_idx);left=max(0,peak_idx-int(.010*fs)) if left_limit is None else max(0,int(left_limit));right=min(len(v)-1,peak_idx+int(.020*fs)) if right_limit is None else min(len(v)-1,int(right_limit))
    peak=float(v[peak_idx]);amp=polarity*(peak-float(baseline))
    if amp<=0:return None
    def level(frac):return float(baseline)+polarity*amp*frac
    rise_dir='up' if polarity>0 else 'down';fall_dir='down' if polarity>0 else 'up'
    onset_t=_crossing_time(v,fs,left,peak_idx,level(.05),rise_dir)
    rise10_t=_crossing_time(v,fs,left,peak_idx,level(.10),rise_dir)
    rise90_t=_crossing_time(v,fs,left,peak_idx,level(.90),rise_dir)
    half_rise_t=_crossing_time(v,fs,left,peak_idx,level(.50),rise_dir)
    half_decay_t=_crossing_time(v,fs,peak_idx,right,level(.50),fall_dir)
    peak_t=peak_idx/fs
    if onset_t is None:onset_t=rise10_t if rise10_t is not None else peak_t
    location=(peak_t-onset_t)*1000.
    # AxoGraph reports onset relative to the detected event/peak-position origin;
    # for the embedded template this origin precedes the waveform peak. Preserve
    # that signed event coordinate by expressing onset relative to peak position.
    onset=(onset_t-peak_t)*1000.
    rise=((rise90_t-rise10_t)*1000.) if rise10_t is not None and rise90_t is not None else np.nan
    width=((half_decay_t-half_rise_t)*1000.) if half_rise_t is not None and half_decay_t is not None else np.nan
    decay=((half_decay_t-peak_t)*1000.) if half_decay_t is not None else np.nan
    return SpikeEventFeatures(peak_voltage=polarity*amp,location_ms=location,onset_ms=onset,rise_ms=rise,width_ms=width,decay_ms=decay)

def build_spike_event_features(v,fs,spike,step_onset_idx):
    """AxoGraph-compatible positive SPIKE event shape measurements."""
    pi=int(round(spike.peak_time*fs))
    # For Find/Measure Peaks and Shapes, peak amplitude is referenced to the
    # local pre-event baseline rather than the physiological dV/dt threshold.
    left=max(int(step_onset_idx),pi-max(int(.005*fs),5))
    baseline=float(np.median(v[left:max(left+1,pi-int(.001*fs))])) if pi-left>2 else float(spike.threshold_voltage)
    return _axograph_shape(v,fs,pi,baseline,polarity=1,left_limit=left,right_limit=min(len(v)-1,pi+int(.010*fs)))

def build_ahp_event_features(v,fs,spike,step_onset_idx,baseline_v=None):
    """AxoGraph-compatible negative-going AHP event shape measurements."""
    if spike.ahp_voltage is None or spike.ahp_time is None:return None
    pi=int(round(spike.peak_time*fs));tr=int(round(spike.ahp_time*fs))
    # AHP baseline is the local voltage immediately after the AP has repolarized,
    # bounded by the AP peak and AHP trough. This mirrors Find/Measure Shapes on
    # an inverted (negative-going) event rather than using AP threshold as base.
    pre0=min(pi+max(int(.00025*fs),1),tr)
    pre1=max(pre0+1,tr-max(int(.00025*fs),1))
    if pre1>pre0:
        base=float(np.max(v[pre0:pre1+1]))
    else:
        base=float(spike.threshold_voltage if baseline_v is None else baseline_v)
    return _axograph_shape(v,fs,tr,base,polarity=-1,left_limit=pi,right_limit=min(len(v)-1,tr+int(.020*fs)))

def first_singlet_or_doublet(spikes,burst_window_ms=20.):
    if not spikes:return []
    return spikes[:2] if len(spikes)>=2 and spikes[1].isi_prev_ms is not None and spikes[1].isi_prev_ms<=burst_window_ms else spikes[:1]

def adaptation_index(spikes):
    isis=[s.isi_prev_ms for s in spikes if s.isi_prev_ms is not None]
    if len(isis)<2:return None
    vals=[(isis[i+1]-isis[i])/(isis[i+1]+isis[i]) for i in range(len(isis)-1) if isis[i+1]+isis[i]!=0]
    return float(np.mean(vals)) if vals else None
