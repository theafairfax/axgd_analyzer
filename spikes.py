"""Action-potential detection and AxoGraph-style waveform feature extraction."""
from __future__ import annotations
from typing import List,Optional
import numpy as np
from scipy.signal import find_peaks,savgol_filter
from models import Sweep,SpikeFeatures,SpikeEventFeatures
from template_matching import DEFAULT_TEMPLATE_MS,DEFAULT_TEMPLATE_MV,TemplateConfig,detect_events_by_template,detect_template_events

def _interp_cross(x0,y0,x1,y1,target):
    if y1==y0:return float(x0)
    return float(x0+(target-y0)*(x1-x0)/(y1-y0))

def _smoothed_dvdt(v,fs,window_ms=.0875,polyorder=3):
    """Noise-stable dV/dt while preserving the fast AP upstroke at 80 kHz."""
    n=max(int(round(window_ms/1000.*fs)),polyorder+2)
    if n%2==0:n+=1
    if n>=len(v):n=len(v)-1 if len(v)%2==0 else len(v)
    if n<=polyorder:return np.gradient(v)*fs/1000.
    return savgol_filter(np.asarray(v,dtype=float),n,polyorder,deriv=1,delta=1000./fs)

def _spike_features_at_peak(v,dv,fs,global_idx,next_global_idx,prev_peak_time,rise_thresh_frac=.10,threshold_dvdt=20.):
    peak_time=global_idx/fs;peak_v=float(v[global_idx]);back=max(int(.005*fs),5);lo=max(global_idx-back,0)
    rise=dv[lo:global_idx+1];max_rel=int(np.argmax(rise));max_rise=float(rise[max_rel]);target=float(threshold_dvdt) if threshold_dvdt is not None else rise_thresh_frac*max_rise
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
    dv=_smoothed_dvdt(v,fs);out=[];prev=None
    for k,p in enumerate(peaks):
        gi=start+int(p);nxt=start+int(peaks[k+1]) if k+1<len(peaks) else min(gi+int(.020*fs),end);sp=_spike_features_at_peak(v,dv,fs,gi,nxt,prev,.10,20.);prev=sp.peak_time;out.append(sp)
    return out

def detect_spikes_template(sweep:Sweep,cfg:TemplateConfig=TemplateConfig(),template_ms=DEFAULT_TEMPLATE_MS,template_mv=DEFAULT_TEMPLATE_MV)->List[SpikeFeatures]:
    v=sweep.voltage;fs=sweep.sampling_rate
    if len(v)==0 or fs==0 or sweep.step_onset_idx is None:return []
    onset=sweep.step_onset_idx;ss=onset+int(cfg.latency_start_ms/1000*fs);se=min(onset+int(cfg.latency_end_ms/1000*fs),len(v)) if cfg.latency_end_ms is not None else (sweep.step_offset_idx or len(v))
    peaks=detect_events_by_template(v,fs,template_ms,template_mv,cfg,search_start_idx=max(ss,0),search_end_idx=se)
    if not len(peaks):return []
    dv=_smoothed_dvdt(v,fs);out=[];prev=None
    for k,gi0 in enumerate(peaks):
        gi=int(gi0);nxt=int(peaks[k+1]) if k+1<len(peaks) else min(gi+int(.020*fs),se-1);sp=_spike_features_at_peak(v,dv,fs,gi,nxt,prev,cfg.voltage_threshold_pct,cfg.threshold_dvdt_mv_per_ms)
        if sp.amplitude>=cfg.amplitude_reject_mv:prev=sp.peak_time;out.append(sp)
    return out

def _cross(v,fs,start,end,target,direction):
    start=max(int(start),0);end=min(int(end),len(v)-1)
    for i in range(start+1,end+1):
        if (direction=='up' and v[i-1]<=target<v[i]) or (direction=='down' and v[i-1]>=target>v[i]):return _interp_cross((i-1)/fs,v[i-1],i/fs,v[i],target)
    return None

def _shape(v,fs,peak_idx,baseline,polarity,left,right,time_origin_idx):
    """Measure event amplitude/location/onset/rise/width on an averaged capture."""
    peak=float(v[peak_idx]);signed_amp=peak-baseline;amp=polarity*signed_amp
    if amp<=0:return None
    level=lambda f:baseline+signed_amp*f;up='up' if polarity>0 else 'down';down='down' if polarity>0 else 'up'
    t05=_cross(v,fs,left,peak_idx,level(.05),up)
    t20=_cross(v,fs,left,peak_idx,level(.20),up);t80=_cross(v,fs,left,peak_idx,level(.80),up)
    t50a=_cross(v,fs,left,peak_idx,level(.50),up);t50b=_cross(v,fs,peak_idx,right,level(.50),down);pt=peak_idx/fs;origin=time_origin_idx/fs
    return SpikeEventFeatures(peak_voltage=signed_amp,location_ms=(pt-origin)*1000.,onset_ms=((t05-origin)*1000. if t05 is not None else np.nan),rise_ms=((t80-t20)*1000. if t20 is not None and t80 is not None else np.nan),width_ms=((t50b-t50a)*1000. if t50a is not None and t50b is not None else np.nan),decay_ms=((t50b-pt)*1000. if t50b is not None else np.nan))

def _lowest_recovering_trough(v,fs,start,end,min_prominence_mv=.10):
    start=max(int(start),1);end=min(int(end),len(v)-2)
    if end<=start:return None
    w=np.asarray(v[start:end+1],dtype=float);distance=max(int(round(.00015*fs)),1)
    minima,_=find_peaks(-w,prominence=max(float(min_prominence_mv),0.),distance=distance)
    valid=[]
    for rel in minima:
        idx=start+int(rel);recover_end=min(idx+max(int(round(.003*fs)),2),len(v)-1)
        if recover_end>idx and np.max(v[idx+1:recover_end+1])>v[idx]+min_prominence_mv:valid.append(idx)
    if valid:return min(valid,key=lambda idx:v[idx])
    d=np.gradient(v)
    valid=[]
    for idx in range(start+1,end):
        if d[idx-1]<0<=d[idx]:
            recover_end=min(idx+max(int(round(.002*fs)),2),len(v)-1)
            if recover_end>idx and np.max(v[idx+1:recover_end+1])>v[idx]+min_prominence_mv:valid.append(idx)
    return min(valid,key=lambda idx:v[idx]) if valid else None

def _align_captures_at_onset(captures,fs,origin_idx,template_baseline_points,onset_sd_multiple=8.0):
    """Mirror AxoGraph's separate Align at Onset step before averaging.

    Align at Onset.axtx derives one common baseline mean/SD from the capture
    baseline, scans forward until |event-baseline| exceeds baselineSD*8, and
    shifts each captured episode so those onsets coincide.  We use the median
    detected onset as the target so alignment does not impose a new arbitrary
    absolute coordinate on the capture x-axis.
    """
    if not captures:return [],origin_idx
    arrs=[np.asarray(c,dtype=float) for c in captures]
    b1=max(int(origin_idx),1);b0=max(0,b1-max(int(template_baseline_points),2))
    baseline_samples=np.concatenate([a[b0:b1] for a in arrs if len(a)>=b1]) if b1>b0 else np.array([])
    if baseline_samples.size<4:return arrs,origin_idx
    base=float(np.mean(baseline_samples));sd=float(np.std(baseline_samples,ddof=1))
    threshold=max(onset_sd_multiple*sd,0.25)
    onsets=[]
    search_start=max(b0,0);search_end=min(len(arrs[0])-1,origin_idx+max(int(round(.004*fs)),2))
    for a in arrs:
        oi=None
        for i in range(search_start,search_end+1):
            if abs(a[i]-base)>=threshold:
                oi=i;break
        onsets.append(oi)
    valid=[o for o in onsets if o is not None]
    if not valid:return arrs,origin_idx
    target=int(round(float(np.median(valid))))
    aligned=[]
    for a,o in zip(arrs,onsets):
        if o is None:continue
        shift=target-o
        out=np.full_like(a,np.nan,dtype=float)
        if shift>=0:
            if shift<len(a):out[shift:]=a[:len(a)-shift]
        else:
            k=-shift
            if k<len(a):out[:len(a)-k]=a[k:]
        aligned.append(out)
    return aligned,origin_idx

def build_average_rheobase_event_features(v,fs,spikes,pre_ms=10.,post_ms=40.,template_cfg:TemplateConfig=TemplateConfig()):
    """Detect, capture, onset-align, average, then measure rheobase AP events."""
    if not spikes or fs<=0:return None,None
    first_peak=min(int(round(s.peak_time*fs)) for s in spikes)
    last_peak=max(int(round(s.peak_time*fs)) for s in spikes)
    margin=max(int(round(.010*fs)),1)
    events=detect_template_events(v,fs,DEFAULT_TEMPLATE_MS,DEFAULT_TEMPLATE_MV,template_cfg,
                                  max(0,first_peak-margin),min(len(v),last_peak+margin))
    if not events:return None,None
    spike_peaks=np.asarray([int(round(s.peak_time*fs)) for s in spikes],dtype=int)
    matched=[];tol=max(int(round(.001*fs)),1)
    for e in events:
        if len(spike_peaks) and int(np.min(np.abs(spike_peaks-e.peak_index)))<=tol:matched.append(e)
    if not matched:return None,None

    template_baseline=max(int(round(template_cfg.baseline_ms/1000.*fs)),1)
    capture_baseline=max(int(round(template_cfg.capture_baseline_ms/1000.*fs)),template_baseline+1)
    capture_after=max(int(round(template_cfg.capture_ms/1000.*fs)),1)
    capture_total=capture_baseline+capture_after
    capture_offset=capture_baseline-template_baseline-1
    captures=[]
    for e in matched:
        first=e.detection_index-capture_offset
        last=first+capture_total
        if first>=0 and last<=len(v):captures.append(np.asarray(v[first:last],dtype=float))
    if not captures:return None,None

    origin=capture_baseline-1
    aligned,origin=_align_captures_at_onset(captures,fs,origin,template_baseline)
    if not aligned:return None,None
    stack=np.vstack(aligned)
    mean=np.nanmean(stack,axis=0)
    if not np.any(np.isfinite(mean)):return None,None
    # Fill edge NaNs introduced only by shifting with the closest finite sample;
    # the central event region remains supported by all/most captures.
    finite=np.flatnonzero(np.isfinite(mean))
    if len(finite):
        mean[:finite[0]]=mean[finite[0]];mean[finite[-1]+1:]=mean[finite[-1]]
        missing=np.flatnonzero(~np.isfinite(mean))
        if len(missing):mean[missing]=np.interp(missing,finite,mean[finite])

    b1=max(origin,1);b0=max(0,b1-template_baseline)
    base=float(np.mean(mean[b0:b1])) if b1>b0 else float(mean[origin])
    mean=mean-base;base=0.0

    spike_left=max(0,origin-template_baseline)
    spike_right=min(len(mean)-1,origin+max(int(.008*fs),1))
    peak=spike_left+int(np.argmax(mean[spike_left:spike_right+1]))
    spike=_shape(mean,fs,peak,base,1,spike_left,min(len(mean)-1,peak+int(.006*fs)),origin)

    next_peak=None
    later,_=find_peaks(mean[peak+max(int(.001*fs),1):],prominence=10.,distance=max(int(.001*fs),1))
    if len(later):next_peak=peak+max(int(.001*fs),1)+int(later[0])
    a0=min(peak+max(int(.00025*fs),1),len(mean)-1)
    # Search the full captured post-spike interval (up to the next AP). The
    # target is the lowest post-spike voltage, not only an early fixed-window
    # fAHP. A negative copy of the lab AP template supplies the AHP event onset
    # and prevents ordinary AP repolarization from being reported as AHP rise.
    a1=len(mean)-2
    if next_peak is not None:a1=min(a1,next_peak-1)
    negative_events=detect_template_events(mean,fs,DEFAULT_TEMPLATE_MS,-DEFAULT_TEMPLATE_MV,
                                           template_cfg,a0,a1+1)
    negative_events=[e for e in negative_events if a0<=e.peak_index<=a1]
    trough=_lowest_recovering_trough(mean,fs,a0,a1)
    ahp_left=peak
    if negative_events:
        selected=min(negative_events,key=lambda e:mean[e.peak_index])
        if trough is None or mean[selected.peak_index]<mean[trough]:trough=selected.peak_index
        # AxoGraph's detection coordinate follows the template's baseline.
        # Step back by that baseline interval so the 5/20% AHP crossings are
        # available for onset and rise measurements.
        ahp_left=max(peak,min(selected.detection_index-template_baseline,trough))
    ahp=_shape(mean,fs,trough,0.0,-1,ahp_left,len(mean)-1,origin) if trough is not None else None
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
