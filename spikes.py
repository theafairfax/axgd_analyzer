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
    return SpikeFeatures(peak_time=peak_time,peak_voltage=peak_v,threshold_voltage=threshold_v,threshold_time=threshold_t,amplitude=amplitude,half_width=hw,max_rise_slope=max_rise,max_fall_slope=max_fall,ahp_voltage=ahp_v,ahp_time=ahp_t,isi_prev_ms=isi)

def detect_spikes(sweep:Sweep,min_peak_mv=-10.,min_prominence_mv=20.,refractory_ms=1.)->List[SpikeFeatures]:
    v=sweep.voltage;fs=sweep.sampling_rate
    if len(v)==0 or fs==0:return []
    start=sweep.step_onset_idx if sweep.step_onset_idx is not None else 0;end=sweep.step_offset_idx if sweep.step_offset_idx is not None else len(v)-1
    w=v[start:end+1];peaks,_=find_peaks(w,height=min_peak_mv,prominence=min_prominence_mv,distance=max(int(refractory_ms/1000*fs),1))
    if not len(peaks):return []
    dv=np.gradient(v)*fs/1000.;out=[];prev=None
    for k,p in enumerate(peaks):
        gi=start+int(p);nxt=start+int(peaks[k+1]) if k+1<len(peaks) else min(gi+int(.020*fs),end);sp=_spike_features_at_peak(v,dv,fs,gi,nxt,prev,.10);prev=sp.peak_time;out.append(sp)
    return out

def detect_spikes_template(sweep:Sweep,cfg:TemplateConfig=TemplateConfig(),template_ms=DEFAULT_TEMPLATE_MS,template_mv=DEFAULT_TEMPLATE_MV)->List[SpikeFeatures]:
    v=sweep.voltage;fs=sweep.sampling_rate
    if len(v)==0 or fs==0 or sweep.step_onset_idx is None:return []
    onset=sweep.step_onset_idx;ss=onset+int(cfg.latency_start_ms/1000*fs);se=min(onset+int(cfg.latency_end_ms/1000*fs),len(v)) if cfg.latency_end_ms is not None else (sweep.step_offset_idx or len(v))
    peaks=detect_events_by_template(v,fs,template_ms,template_mv,cfg,search_start_idx=max(ss,0),search_end_idx=se)
    if not len(peaks):return []
    dv=np.gradient(v)*fs/1000.;out=[];prev=None
    for k,gi0 in enumerate(peaks):
        gi=int(gi0);nxt=int(peaks[k+1]) if k+1<len(peaks) else min(gi+int(.020*fs),se-1);sp=_spike_features_at_peak(v,dv,fs,gi,nxt,prev,cfg.voltage_threshold_pct)
        if sp.amplitude>=cfg.amplitude_reject_mv:prev=sp.peak_time;out.append(sp)
    return out

def _cross(v,fs,start,end,target,direction):
    start=max(int(start),0);end=min(int(end),len(v)-1)
    for i in range(start+1,end+1):
        if (direction=='up' and v[i-1]<=target<v[i]) or (direction=='down' and v[i-1]>=target>v[i]):return _interp_cross((i-1)/fs,v[i-1],i/fs,v[i],target)
    return None

def _shape(v,fs,peak_idx,baseline,polarity,left,right,time_origin_idx):
    peak=float(v[peak_idx]);signed_amp=peak-baseline;amp=polarity*signed_amp
    if amp<=0:return None
    level=lambda f:baseline+signed_amp*f;up='up' if polarity>0 else 'down';down='down' if polarity>0 else 'up'
    t05=_cross(v,fs,left,peak_idx,level(.05),up);t10=_cross(v,fs,left,peak_idx,level(.10),up);t90=_cross(v,fs,left,peak_idx,level(.90),up);t50a=_cross(v,fs,left,peak_idx,level(.50),up);t50b=_cross(v,fs,peak_idx,right,level(.50),down);pt=peak_idx/fs;origin=time_origin_idx/fs
    return SpikeEventFeatures(peak_voltage=signed_amp,location_ms=(pt-origin)*1000.,onset_ms=((t05-origin)*1000. if t05 is not None else np.nan),rise_ms=((t90-t10)*1000. if t10 is not None and t90 is not None else np.nan),width_ms=((t50b-t50a)*1000. if t50a is not None and t50b is not None else np.nan),decay_ms=((t50b-pt)*1000. if t50b is not None else np.nan))

def build_average_rheobase_event_features(v,fs,spikes,pre_ms=10.,post_ms=40.):
    """Measure AxoGraph-style SPIKE/AHP shapes on the averaged captured AP."""
    if not spikes:return None,None
    pre=max(int(round(pre_ms/1000.*fs)),1);post=max(int(round(post_ms/1000.*fs)),1);captures=[]
    for sp in spikes:
        p=int(round(sp.peak_time*fs));lo=p-pre;hi=p+post+1
        if lo>=0 and hi<=len(v):captures.append(np.asarray(v[lo:hi],dtype=float))
    if not captures:return None,None
    mean=np.mean(np.vstack(captures),axis=0);peak=pre
    template_peak_ms=float(DEFAULT_TEMPLATE_MS[int(np.argmax(DEFAULT_TEMPLATE_MV))])
    origin=peak-int(round(template_peak_ms/1000.*fs))
    # Validation against the manual AxoGraph export showed a residual common
    # +0.2375 ms shift in both SPIKE and AHP locations. At 80 kHz this is exactly
    # 19 samples, so move the AxoGraph event origin 19 samples later.
    origin += int(round(0.0002375*fs))
    b1=max(origin,1);b0=max(0,b1-int(.001*fs));base=float(np.mean(mean[b0:b1])) if b1>b0 else float(mean[origin])
    mean=mean-base;base=0.0
    spike=_shape(mean,fs,peak,base,1,max(0,origin-int(.001*fs)),min(len(mean)-1,peak+int(.006*fs)),origin)
    a0=min(peak+max(int(.00025*fs),1),len(mean)-1);a1=min(origin+int(.020*fs),len(mean)-1);ahp=None
    if a1>a0:
        trough=a0+int(np.argmin(mean[a0:a1+1]))
        # Search the entire remaining captured waveform for the 50% recovery.
        # The previous search boundary could terminate before broad AHPs crossed
        # back through half-amplitude, yielding NaN width/decay values.
        ahp=_shape(mean,fs,trough,0.0,-1,peak,len(mean)-1,origin)
    return spike,ahp

def build_spike_event_features(v,fs,spike,step_onset_idx):
    pi=int(round(spike.peak_time*fs));left=max(int(step_onset_idx),pi-max(int(.005*fs),5));baseline=float(np.median(v[left:max(left+1,pi-int(.001*fs))])) if pi-left>2 else float(spike.threshold_voltage);return _shape(v,fs,pi,baseline,1,left,min(len(v)-1,pi+int(.010*fs)),pi)
def build_ahp_event_features(v,fs,spike,step_onset_idx,baseline_v=None):
    if spike.ahp_voltage is None or spike.ahp_time is None:return None
    pi=int(round(spike.peak_time*fs));tr=int(round(spike.ahp_time*fs));base=float(spike.threshold_voltage if baseline_v is None else baseline_v);return _shape(v,fs,tr,base,-1,pi,min(len(v)-1,tr+int(.020*fs)),pi)
def first_singlet_or_doublet(spikes,burst_window_ms=20.):return spikes[:2] if len(spikes)>=2 and spikes[1].isi_prev_ms is not None and spikes[1].isi_prev_ms<=burst_window_ms else spikes[:1]
def adaptation_index(spikes):
    isis=[s.isi_prev_ms for s in spikes if s.isi_prev_ms is not None]
    if len(isis)<2:return None
    vals=[(isis[i+1]-isis[i])/(isis[i+1]+isis[i]) for i in range(len(isis)-1) if isis[i+1]+isis[i]!=0];return float(np.mean(vals)) if vals else None
