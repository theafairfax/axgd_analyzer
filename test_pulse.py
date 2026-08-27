"""Passive membrane analysis from AxoGraph test-pulse recordings.

The standard 0001 Test Pulse 1-Ch protocol applies a small voltage command
(~10 mV) beginning near 20 ms and ending near 60 ms.  Passive properties are
computed from the measured current transient using the conventional whole-cell
membrane-test relationships:

    Rs = dV / (I_peak - I_ss)
    Rm = dV / I_ss - Rs
    Cm = tau / (Rs || Rm)

The very earliest samples are excluded from the exponential fit because they
are dominated by electrode/pipette capacitance and acquisition filtering.  The
fit is then extrapolated back to t=0 to estimate the membrane transient
amplitude I_peak-I_ss without using the raw capacitive artifact.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit
from models import Recording, Sweep

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
    margin=max(int(.002*fs),2); early_hi=min(len(d)-margin,int(.5*len(d)))
    if early_hi<=margin:return None,None
    onset=margin+int(np.argmax(d[margin:early_hi]))
    target=onset+int(expected_width_ms/1000.*fs)
    offset=_edge_near(d,target,max(int(.010*fs),5))
    if offset is None or offset<=onset:return None,None
    return onset+1,offset+1


def _exp_transient(t_ms,amp,tau_ms):
    return amp*np.exp(-t_ms/tau_ms)


def _analyze_voltage_clamp_pulse(sweep:Sweep,pulse_amplitude_mv:float,expected_onset_ms:float,expected_width_ms:float,
                                  artifact_blank_ms:float=.10,fit_end_ms:float=5.0):
    raw=sweep.current if sweep.current is not None else sweep.voltage
    if raw is None:return None
    x=np.asarray(raw,dtype=float); fs=float(sweep.sampling_rate)
    onset,offset=_find_test_pulse_window(x,fs,expected_onset_ms,expected_width_ms)
    if onset is None or offset is None or offset-onset<20:return None

    pre_n=max(int(.005*fs),5)
    base=float(np.median(x[max(0,onset-pre_n):onset]))
    pulse=x[onset:offset]-base
    if len(pulse)<20:return None

    early=min(max(int(.002*fs),8),len(pulse))
    sign=1. if abs(np.max(pulse[:early]))>=abs(np.min(pulse[:early])) else -1.
    y=pulse*sign
    tail_n=min(max(int(.004*fs),5),max(len(y)//5,5))
    iss=float(np.median(y[-tail_n:]))
    if iss<=0:return None

    # Convert to pA if a one-channel fallback somehow retained amperes.
    raw_peak=float(np.max(y[:min(len(y),max(int(.0015*fs),10))]))
    scale=1e12 if max(abs(raw_peak),abs(iss))<1e-3 else 1.
    iss_pa=iss*scale; raw_peak_pa=raw_peak*scale

    # Fit only the membrane-dominated decay after blanking the earliest artifact.
    t_ms=np.arange(len(y))/fs*1000.
    transient=(y-iss)*scale
    fit_mask=(t_ms>=artifact_blank_ms)&(t_ms<=fit_end_ms)&np.isfinite(transient)&(transient>0)
    if np.count_nonzero(fit_mask)<8:return None
    tf=t_ms[fit_mask]; z=transient[fit_mask]
    amp0=float(np.max(z)); tau0=.5
    try:
        popt,_=curve_fit(_exp_transient,tf,z,p0=[amp0,tau0],
                         bounds=([1e-9,.02],[np.inf,10.]),maxfev=20000)
        transient_amp_pa=float(popt[0]); tau_ms=float(popt[1])
    except Exception:return None
    if transient_amp_pa<=0 or not (.02<tau_ms<10.):return None

    # Conventional membrane-test equations.  Rs uses the transient current
    # ABOVE steady state, not total current at t=0.
    rs=abs(pulse_amplitude_mv/transient_amp_pa)*1000.
    rtotal=abs(pulse_amplitude_mv/iss_pa)*1000.
    rm=rtotal-rs
    if not (1<rs<500 and 5<rm<5000):return None
    rparallel=(rs*rm)/(rs+rm)
    cm=tau_ms*1000./rparallel if rparallel>0 else None
    if cm is None or not (1<cm<1000):return None

    # Fit quality on the fitted interval.
    pred=_exp_transient(tf,transient_amp_pa,tau_ms)
    ss_res=float(np.sum((z-pred)**2)); ss_tot=float(np.sum((z-np.mean(z))**2))
    r2=1.-ss_res/ss_tot if ss_tot>0 else np.nan
    if np.isfinite(r2) and r2<.75:return None

    return {
        'Rs (MOhm)':rs,'Rm (MOhm)':rm,'Cm (pF)':cm,
        'Rtotal (MOhm)':rtotal,'I transient extrapolated (pA)':transient_amp_pa,
        'I raw peak (pA)':raw_peak_pa,'I steady (pA)':iss_pa,'Tau membrane (ms)':tau_ms,
        'Fit R2':r2,'Artifact blank (ms)':artifact_blank_ms,
        'Onset (ms)':onset/fs*1000.,'Offset (ms)':offset/fs*1000.
    }


def _robust_keep(values,key):
    arr=np.asarray([v[key] for v in values],dtype=float)
    if len(arr)<4:return np.ones(len(arr),dtype=bool)
    med=float(np.median(arr)); mad=float(np.median(np.abs(arr-med)))
    if mad<=1e-12:return np.ones(len(arr),dtype=bool)
    return np.abs(arr-med)/(1.4826*mad)<=3.5


def compute_test_pulse_properties(recording:Recording,pulse_amplitude_mv:float=10.,expected_onset_ms:float=20.,expected_width_ms:float=40.)->TestPulseProperties:
    vals=[]
    for sw in recording.sweeps:
        r=_analyze_voltage_clamp_pulse(sw,pulse_amplitude_mv,expected_onset_ms,expected_width_ms)
        if r is not None:vals.append({'Sweep':sw.index,**r})
    if not vals:
        return TestPulseProperties(n_total_sweeps=len(recording.sweeps),pulse_amplitude_mv=pulse_amplitude_mv,
                                   expected_onset_ms=expected_onset_ms,expected_width_ms=expected_width_ms)
    keep=np.ones(len(vals),dtype=bool)
    for key in ('Rs (MOhm)','Rm (MOhm)','Cm (pF)'):
        keep&=_robust_keep(vals,key)
    for v,k in zip(vals,keep):v['Included in average']=bool(k)
    accepted=[v for v,k in zip(vals,keep) if k] or vals
    return TestPulseProperties(
        series_resistance=float(np.mean([v['Rs (MOhm)'] for v in accepted])),
        membrane_resistance=float(np.mean([v['Rm (MOhm)'] for v in accepted])),
        membrane_capacitance=float(np.mean([v['Cm (pF)'] for v in accepted])),
        n_valid_sweeps=len(accepted),n_total_sweeps=len(recording.sweeps),sweep_values=vals,
        pulse_amplitude_mv=pulse_amplitude_mv,expected_onset_ms=expected_onset_ms,expected_width_ms=expected_width_ms)
