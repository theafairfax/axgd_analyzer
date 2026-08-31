"""Passive membrane analysis mirroring AxoGraph's 'Measure Rs, Rm and Cm'.

The implementation follows the supplied AxoGraph AXT source rather than an
empirical capacitance approximation.  The lab averages the test-pulse repeats
first, so this module fits the ensemble-average trace as one AxoGraph episode.

Relevant AxoGraph rules from Measure and Correct Rs.axtx:
  * skipFitSamples = 1 + Round(0.2 ms / sampleInterval), minimum 2
  * fit a double exponential plus constant first
  * fall back to a single exponential plus constant when
        t2 <= 0, t1 <= 0, t2 > 0.3*t1, or a1*a2 <= 0
  * extrapolate fitted amplitudes back by (skipFitSamples-1)*dt
  * Rs = |pulseSize / (a1+a2+steadyState)|
  * Rm = |pulseSize / steadyState| - Rs
  * Cm = t2 * (1/Rs + 1/Rm)
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
    d=np.abs(np.diff(smooth));radius=max(int(tolerance_ms/1000.*fs),3)
    onset=_edge_near(d,expected_onset_ms/1000.*fs,radius)
    offset=_edge_near(d,(expected_onset_ms+expected_width_ms)/1000.*fs,radius)
    if onset is not None and offset is not None and offset>onset+max(int(.005*fs),5):return onset+1,offset+1
    return None,None


def _single_plus(t,a,tau,c):return c+a*np.exp(-t/tau)
def _double_plus(t,a1,t1,a2,t2,c):return c+a1*np.exp(-t/t1)+a2*np.exp(-t/t2)

def _extract_current(sw):
    raw=sw.current if sw.current is not None else sw.voltage
    return None if raw is None else np.asarray(raw,dtype=float)

def _ensemble_average(recording:Recording):
    traces=[];fs=None
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


def _fit_axograph_transient(t_s,y):
    """Double-exp-plus-constant with AxoGraph's explicit fallback rule.

    AxoGraph's returned t2 is the fast component in the accepted double fit
    (its source requires t2 < 0.3*t1 and uses t2 for Cm).
    """
    y=np.asarray(y,dtype=float);t_s=np.asarray(t_s,dtype=float)
    if len(y)<8:return None
    c0=float(np.mean(y[-max(3,len(y)//10):]));amp0=float(y[0]-c0)
    sign=1. if amp0>=0 else -1.
    scale=max(abs(amp0),float(np.ptp(y)),1e-15)
    try:
        # Parameter order deliberately names the slow component t1 and fast t2
        # to match AxoGraph's acceptance test and Cm=t2*(...).
        p0=[sign*.35*scale,.002,sign*.65*scale,.0003,c0]
        pd,_=curve_fit(_double_plus,t_s,y,p0=p0,
            bounds=([-np.inf,1e-5,-np.inf,1e-5,-np.inf],[np.inf,.1,np.inf,.03,np.inf]),maxfev=100000)
        a1,t1,a2,t2,c=[float(q) for q in pd]
        # Normalize component labels so t2 is the faster component, as required
        # by the AxoGraph source's t2 < 0.3*t1 test.
        if t1<t2:a1,t1,a2,t2=a2,t2,a1,t1
        if t2>0 and t1>0 and t2<=.3*t1 and a1*a2>0:
            pred=_double_plus(t_s,a1,t1,a2,t2,c)
            return {'model':'double','a1':a1,'t1':t1,'a2':a2,'t2':t2,'steady':c,'pred':pred}
    except Exception:
        pass
    try:
        ps,_=curve_fit(_single_plus,t_s,y,p0=[amp0,.0005,c0],
            bounds=([-np.inf,1e-5,-np.inf],[np.inf,.1,np.inf]),maxfev=100000)
        a2,t2,c=[float(q) for q in ps]
        pred=_single_plus(t_s,a2,t2,c)
        return {'model':'single','a1':0.,'t1':.001,'a2':a2,'t2':t2,'steady':c,'pred':pred}
    except Exception:return None


def _measure_average_trace(x,fs,pulse_amplitude_mv,expected_onset_ms,expected_width_ms):
    onset,offset=_find_test_pulse_window(x,fs,expected_onset_ms,expected_width_ms)
    if onset is None or offset is None:return None
    dt=1./fs
    # Exact Measure Rs, Rm and Cm source convention: 0.2 ms, plus one sample.
    skip_samples=max(1+int(round(.0002/dt)),2)
    ref_start=max(0,onset-(offset-onset));ref_end=max(ref_start+1,onset-skip_samples)
    fit_start=onset+skip_samples;fit_end=max(fit_start+8,offset-skip_samples)
    fit_end=min(fit_end,len(x))
    if fit_end-fit_start<8:return None
    ref=np.asarray(x[ref_start:ref_end],dtype=float)
    data=np.asarray(x[fit_start:fit_end],dtype=float)
    baseline=float(np.mean(ref)) if len(ref) else 0.
    # Fit raw data, just as FitDoubleExponential does; subtract the reference
    # from only the returned steady-state constant afterward.
    t=np.arange(len(data),dtype=float)*dt
    fit=_fit_axograph_transient(t,data)
    if fit is None:return None
    a1,t1,a2,t2,steady=fit['a1'],fit['t1'],fit['a2'],fit['t2'],fit['steady']
    steady-=baseline
    # AxoGraph fit starts at fitMin.  Its source extrapolates by
    # (skipFitSamples-1)*sampleInterval to pulse onset.
    extrap=(skip_samples-1)*dt
    if t1>0:a1*=np.exp(extrap/t1)
    if t2>0:a2*=np.exp(extrap/t2)

    # Convert current units to amperes when the parser supplies pA-scale values.
    magnitude=max(abs(a1),abs(a2),abs(steady),1e-30)
    current_scale=1e-12 if magnitude>1e-3 else 1.
    a1_a=a1*current_scale;a2_a=a2*current_scale;steady_a=steady*current_scale
    pulse_v=pulse_amplitude_mv*1e-3
    denom=a1_a+a2_a+steady_a
    if denom==0 or steady_a==0:return None
    rs=abs(pulse_v/denom)
    rm=abs(pulse_v/steady_a)-rs
    if rs<=0 or rm<=0:return None
    cm=t2*(1./rs+1./rm)
    rs_m=rs/1e6;rm_m=rm/1e6;cm_pf=cm*1e12
    if not (1<rs_m<500 and 5<rm_m<5000 and 1<cm_pf<2000):return None
    pred=np.asarray(fit['pred']);rss=float(np.sum((data-pred)**2));sst=float(np.sum((data-np.mean(data))**2));r2=1-rss/sst if sst>0 else np.nan
    return {'Trace':'20-sweep ensemble average','Model':fit['model'],'Rs (MOhm)':rs_m,'Rm (MOhm)':rm_m,'Cm (pF)':cm_pf,
            'A1 fitted':a1,'Tau1 (ms)':t1*1000.,'A2 fitted':a2,'Tau2 used for Cm (ms)':t2*1000.,
            'Steady fitted':steady,'Fit R2':r2,'Skip after onset (ms)':skip_samples*dt*1000.,
            'Onset (ms)':onset/fs*1000.,'Offset (ms)':offset/fs*1000.}


def compute_test_pulse_properties(recording:Recording,pulse_amplitude_mv:float=10.,expected_onset_ms:float=20.,expected_width_ms:float=40.)->TestPulseProperties:
    avg,fs,n=_ensemble_average(recording)
    if avg is None:return TestPulseProperties(n_total_sweeps=len(recording.sweeps),pulse_amplitude_mv=pulse_amplitude_mv,expected_onset_ms=expected_onset_ms,expected_width_ms=expected_width_ms)
    result=_measure_average_trace(avg,fs,pulse_amplitude_mv,expected_onset_ms,expected_width_ms)
    if result is None:return TestPulseProperties(n_total_sweeps=len(recording.sweeps),n_valid_sweeps=n,pulse_amplitude_mv=pulse_amplitude_mv,expected_onset_ms=expected_onset_ms,expected_width_ms=expected_width_ms)
    return TestPulseProperties(series_resistance=result['Rs (MOhm)'],membrane_resistance=result['Rm (MOhm)'],membrane_capacitance=result['Cm (pF)'],n_valid_sweeps=n,n_total_sweeps=len(recording.sweeps),sweep_values=[result],pulse_amplitude_mv=pulse_amplitude_mv,expected_onset_ms=expected_onset_ms,expected_width_ms=expected_width_ms)
