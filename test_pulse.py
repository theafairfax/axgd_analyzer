"""Passive membrane analysis that mirrors the AxoGraph Test Cell workflow.

Lab workflow:
  1. average all 20 repeats of 0001 Test Pulse 1-Ch
  2. run Electrophys -> Measure Rm, Rs, and Cm on the ensemble average

AxoGraph Test Setup defaults shown in the acquisition manual:
  - Fit Double Exponential: ON
  - After Pulse Onset Skip: 0.1 ms
  - Then Fit Over: 5 ms

The ensemble-average current is modeled as
    I(t) = Iss + A1*exp(-t/tau1) + A2*exp(-t/tau2)
with a single-exponential fallback when the second component is not warranted.
Series resistance uses the fitted total current extrapolated to pulse onset,
membrane resistance uses the steady-state current, and membrane capacitance is
computed from the *initial slope* of the fitted transient.  For a double
exponential this defines an effective early time constant

    tau_eff = (A1 + A2) / (A1/tau1 + A2/tau2)

and Cm = tau_eff / (Rs || Rm).  This preserves the effect of both fitted
components at pulse onset without allowing a slow dendritic/distributed tail to
dominate the capacitance estimate through full charge integration.  For a
single exponential tau_eff is simply tau1.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit
from models import Recording

@dataclass
class TestPulseProperties:
    series_resistance: Optional[float] = None
    membrane_resistance: Optional[float] = None
    membrane_capacitance: Optional[float] = None
    n_valid_sweeps: int = 0
    n_total_sweeps: int = 0
    sweep_values: list = field(default_factory=list)
    pulse_amplitude_mv: float = 10.0
    expected_onset_ms: float = 20.0
    expected_width_ms: float = 40.0


def _edge_near(d, center_idx, radius):
    lo=max(0,int(center_idx-radius)); hi=min(len(d),int(center_idx+radius+1))
    if hi<=lo:return None
    return lo+int(np.argmax(d[lo:hi]))


def _find_test_pulse_window(signal,fs,expected_onset_ms=20.,expected_width_ms=40.,tolerance_ms=6.):
    x=np.asarray(signal,dtype=float)
    if len(x)<30 or fs<=0:return None,None
    w=max(int(round(.00005*fs)),1)
    smooth=np.convolve(x,np.ones(w)/w,mode='same') if w>1 else x
    d=np.abs(np.diff(smooth))
    radius=max(int(tolerance_ms/1000.*fs),3)
    onset=_edge_near(d,expected_onset_ms/1000.*fs,radius)
    offset=_edge_near(d,(expected_onset_ms+expected_width_ms)/1000.*fs,radius)
    if onset is not None and offset is not None and offset>onset+max(int(.005*fs),5):
        return onset+1,offset+1
    return None,None


def _single_exp(t_ms,a,tau):
    return a*np.exp(-t_ms/tau)


def _double_exp(t_ms,a1,tau1,a2,tau2):
    return a1*np.exp(-t_ms/tau1)+a2*np.exp(-t_ms/tau2)


def _extract_current(sw):
    raw=sw.current if sw.current is not None else sw.voltage
    return None if raw is None else np.asarray(raw,dtype=float)


def _ensemble_average(recording:Recording):
    traces=[]; fs=None
    for sw in recording.sweeps:
        x=_extract_current(sw)
        if x is None or len(x)<20:continue
        this_fs=float(sw.sampling_rate)
        if fs is None:fs=this_fs
        if abs(this_fs-fs)>1e-6:continue
        traces.append(x)
    if not traces:return None,None,0
    n=min(len(x) for x in traces)
    return np.mean(np.vstack([x[:n] for x in traces]),axis=0),fs,len(traces)


def _fit_transient(t,z):
    """AxoGraph-like double exponential with automatic single-exp fallback."""
    try:
        ps,_=curve_fit(_single_exp,t,z,p0=[float(z[0]),.5],
                       bounds=([1e-9,.01],[np.inf,20.]),maxfev=30000)
        pred_s=_single_exp(t,*ps)
        rss_s=float(np.sum((z-pred_s)**2))
    except Exception:
        ps=None; rss_s=np.inf

    amp0=max(float(z[0]),1e-9)
    p0=[amp0*.65,.25,amp0*.35,1.5]
    try:
        pd,_=curve_fit(_double_exp,t,z,p0=p0,
                       bounds=([0,.01,0,.05],[np.inf,5.,np.inf,30.]),maxfev=60000)
        a1,t1,a2,t2=[float(v) for v in pd]
        if t1>t2:a1,t1,a2,t2=a2,t2,a1,t1
        pred_d=_double_exp(t,a1,t1,a2,t2)
        rss_d=float(np.sum((z-pred_d)**2))
        n=max(len(z),1)
        bic_s=n*np.log(max(rss_s/n,1e-30))+2*np.log(n) if np.isfinite(rss_s) else np.inf
        bic_d=n*np.log(max(rss_d/n,1e-30))+4*np.log(n)
        frac=min(a1,a2)/max(a1+a2,1e-12)
        separated=t2/max(t1,1e-12)>=1.5
        if bic_d+2.0<bic_s and frac>=.02 and separated:
            return {'model':'double','a1':a1,'tau1':t1,'a2':a2,'tau2':t2,'rss':rss_d,'pred':pred_d}
    except Exception:
        pass

    if ps is None:return None
    return {'model':'single','a1':float(ps[0]),'tau1':float(ps[1]),'a2':0.,'tau2':0.,'rss':rss_s,'pred':pred_s}


def _measure_average_trace(x,fs,pulse_amplitude_mv,expected_onset_ms,expected_width_ms,
                           skip_ms=.1,fit_over_ms=5.0):
    onset,offset=_find_test_pulse_window(x,fs,expected_onset_ms,expected_width_ms)
    if onset is None or offset is None:return None
    pre_n=max(int(.005*fs),5)
    baseline=float(np.mean(x[max(0,onset-pre_n):onset]))
    pulse=np.asarray(x[onset:offset],dtype=float)-baseline
    if len(pulse)<20:return None

    early=min(max(int(.002*fs),8),len(pulse))
    sign=1. if abs(np.max(pulse[:early]))>=abs(np.min(pulse[:early])) else -1.
    y=pulse*sign

    off_margin=max(int(.001*fs),1)
    ss_start=max(int(len(y)*.5),0)
    ss_end=max(len(y)-off_margin,ss_start+1)
    iss=float(np.mean(y[ss_start:ss_end]))
    if iss<=0:return None

    raw_peak=float(np.max(y[:min(len(y),max(int(.0015*fs),10))]))
    scale=1e12 if max(abs(raw_peak),abs(iss))<1e-3 else 1.
    iss_pa=iss*scale; raw_peak_pa=raw_peak*scale

    t_ms=np.arange(len(y),dtype=float)/fs*1000.
    transient=(y-iss)*scale
    fit_start=skip_ms
    fit_end=skip_ms+fit_over_ms
    mask=(t_ms>=fit_start)&(t_ms<=fit_end)&np.isfinite(transient)&(transient>0)
    if np.count_nonzero(mask)<8:return None
    tf=t_ms[mask]; z=transient[mask]
    fit=_fit_transient(tf,z)
    if fit is None:return None

    a1,t1,a2,t2=fit['a1'],fit['tau1'],fit['a2'],fit['tau2']
    transient0_pa=a1+a2
    i0_pa=iss_pa+transient0_pa
    if i0_pa<=iss_pa:return None

    rs=abs(pulse_amplitude_mv/i0_pa)*1000.
    rtotal=abs(pulse_amplitude_mv/iss_pa)*1000.
    rm=rtotal-rs
    if not (1<rs<500 and 5<rm<5000):return None
    rparallel=(rs*rm)/(rs+rm)
    if rparallel<=0:return None

    # Effective time constant defined by the fitted transient's initial slope:
    # Itr(0) / -dItr/dt|0.  This is robust to a long slow component that would
    # otherwise dominate integrated charge and grossly inflate Cm.
    slope0=a1/max(t1,1e-12)+a2/max(t2,1e-12)
    tau_effective=transient0_pa/max(slope0,1e-12)
    cm=tau_effective*1000./rparallel
    if not (1<cm<1000):return None

    # Keep the full integrated transient as a diagnostic only. It reflects slow
    # distributed charging and is intentionally not used for reported Cm.
    q_fit_fc=a1*t1+a2*t2
    divider=(rtotal/rm)**2
    cm_charge_integral=q_fit_fc*divider/abs(pulse_amplitude_mv)
    charge_weighted_tau=q_fit_fc/max(transient0_pa,1e-12)

    pred=fit['pred']; ss_res=fit['rss']; ss_tot=float(np.sum((z-np.mean(z))**2))
    r2=1.-ss_res/ss_tot if ss_tot>0 else np.nan

    return {
        'Trace':'20-sweep ensemble average','Model':fit['model'],
        'Rs (MOhm)':rs,'Rm (MOhm)':rm,'Cm (pF)':cm,'Rtotal (MOhm)':rtotal,
        'I0 fitted total (pA)':i0_pa,'I transient fitted (pA)':transient0_pa,
        'I raw peak (pA)':raw_peak_pa,'I steady (pA)':iss_pa,
        'A fast (pA)':a1,'Tau fast (ms)':t1,
        'A slow (pA)':a2 if fit['model']=='double' else np.nan,
        'Tau slow (ms)':t2 if fit['model']=='double' else np.nan,
        'Tau effective initial-slope (ms)':tau_effective,
        'Cm charge-integral diagnostic (pF)':cm_charge_integral,
        'Tau charge-weighted diagnostic (ms)':charge_weighted_tau,
        'Transient charge diagnostic (fC)':q_fit_fc,
        'Fit R2':r2,'Skip after onset (ms)':skip_ms,'Fit over (ms)':fit_over_ms,
        'Onset (ms)':onset/fs*1000.,'Offset (ms)':offset/fs*1000.
    }


def compute_test_pulse_properties(recording:Recording,pulse_amplitude_mv:float=10.,
                                  expected_onset_ms:float=20.,expected_width_ms:float=40.)->TestPulseProperties:
    avg,fs,n=_ensemble_average(recording)
    if avg is None:
        return TestPulseProperties(n_total_sweeps=len(recording.sweeps),pulse_amplitude_mv=pulse_amplitude_mv,
                                   expected_onset_ms=expected_onset_ms,expected_width_ms=expected_width_ms)
    result=_measure_average_trace(avg,fs,pulse_amplitude_mv,expected_onset_ms,expected_width_ms,
                                  skip_ms=.1,fit_over_ms=5.)
    if result is None:
        return TestPulseProperties(n_total_sweeps=len(recording.sweeps),n_valid_sweeps=n,
                                   pulse_amplitude_mv=pulse_amplitude_mv,expected_onset_ms=expected_onset_ms,
                                   expected_width_ms=expected_width_ms)
    return TestPulseProperties(series_resistance=result['Rs (MOhm)'],
        membrane_resistance=result['Rm (MOhm)'],membrane_capacitance=result['Cm (pF)'],
        n_valid_sweeps=n,n_total_sweeps=len(recording.sweeps),sweep_values=[result],
        pulse_amplitude_mv=pulse_amplitude_mv,expected_onset_ms=expected_onset_ms,
        expected_width_ms=expected_width_ms)
