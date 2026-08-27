"""Passive membrane analysis that mirrors the AxoGraph workflow used in-lab.

Protocol used for these recordings:
  1. acquire 20 repeats of 0001 Test Pulse 1-Ch
  2. calculate the ensemble average of all 20 current traces
  3. run AxoGraph Electrophys -> Measure Rm, Rs, and Cm on the average trace

AxoGraph's documentation describes the passive-property measurement as measuring
the steady-state current near the end of the pulse and fitting an exponential to
the first few milliseconds of the response.  For an Rs-(Rm||Cm) equivalent
circuit:

    I(t) = Iss + A * exp(-t/tau)
    I(0+) = Iss + A = dV / Rs
    Iss = dV / (Rs + Rm)
    tau = (Rs || Rm) * Cm

The primary reported Rs/Rm/Cm values are therefore calculated from the ensemble
average, not by averaging per-sweep parameter estimates.
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


def _exp_transient(t_ms,amp,tau_ms):
    return amp*np.exp(-t_ms/tau_ms)


def _extract_current(sw):
    raw=sw.current if sw.current is not None else sw.voltage
    return None if raw is None else np.asarray(raw,dtype=float)


def _ensemble_average(recording:Recording):
    traces=[]
    fs=None
    for sw in recording.sweeps:
        x=_extract_current(sw)
        if x is None or len(x)<20:continue
        if fs is None:fs=float(sw.sampling_rate)
        if abs(float(sw.sampling_rate)-fs)>1e-6:continue
        traces.append(x)
    if not traces:return None,None,0
    n=min(len(x) for x in traces)
    stack=np.vstack([x[:n] for x in traces])
    return np.mean(stack,axis=0),fs,len(traces)


def _measure_average_trace(x,fs,pulse_amplitude_mv,expected_onset_ms,expected_width_ms,
                           fit_end_ms=3.0):
    onset,offset=_find_test_pulse_window(x,fs,expected_onset_ms,expected_width_ms)
    if onset is None or offset is None:return None
    pre_n=max(int(.005*fs),5)
    baseline=float(np.mean(x[max(0,onset-pre_n):onset]))
    pulse=np.asarray(x[onset:offset],dtype=float)-baseline
    if len(pulse)<20:return None

    early=min(max(int(.002*fs),8),len(pulse))
    sign=1. if abs(np.max(pulse[:early]))>=abs(np.min(pulse[:early])) else -1.
    y=pulse*sign

    # AxoGraph documentation describes a steady-state measurement at the end of
    # the pulse. Use the final 4 ms, safely before the OFF transition.
    tail_n=max(int(.004*fs),5)
    tail_n=min(tail_n,max(len(y)//4,5))
    iss=float(np.mean(y[-tail_n:]))
    if iss<=0:return None

    raw_peak=float(np.max(y[:min(len(y),max(int(.0015*fs),10))]))
    scale=1e12 if max(abs(raw_peak),abs(iss))<1e-3 else 1.
    iss_pa=iss*scale; raw_peak_pa=raw_peak*scale

    # Fit the FIRST few milliseconds, as AxoGraph does.  Do not discard the
    # first full acquired sample after the edge: at 16 kHz this is 0.0625 ms,
    # and excluding it substantially biases tau upward and Rs/Cm upward.
    t_ms=np.arange(len(y),dtype=float)/fs*1000.
    transient=(y-iss)*scale
    first_sample_ms=1000./fs
    fit_start_ms=first_sample_ms
    mask=(t_ms>=fit_start_ms)&(t_ms<=fit_end_ms)&np.isfinite(transient)&(transient>0)
    if np.count_nonzero(mask)<8:return None
    tf=t_ms[mask]; z=transient[mask]
    try:
        popt,_=curve_fit(_exp_transient,tf,z,p0=[float(z[0]),.5],
                         bounds=([1e-9,.01],[np.inf,10.]),maxfev=30000)
        amp_pa=float(popt[0]); tau_ms=float(popt[1])
    except Exception:return None
    if amp_pa<=0 or not (.01<tau_ms<10.):return None

    # Total extrapolated current at pulse onset, not transient amplitude alone.
    i0_pa=iss_pa+amp_pa
    rs=abs(pulse_amplitude_mv/i0_pa)*1000.
    rtotal=abs(pulse_amplitude_mv/iss_pa)*1000.
    rm=rtotal-rs
    if not (1<rs<500 and 5<rm<5000):return None
    rparallel=(rs*rm)/(rs+rm)
    cm=tau_ms*1000./rparallel if rparallel>0 else None
    if cm is None or not (1<cm<1000):return None

    pred=_exp_transient(tf,amp_pa,tau_ms)
    ss_res=float(np.sum((z-pred)**2)); ss_tot=float(np.sum((z-np.mean(z))**2))
    r2=1.-ss_res/ss_tot if ss_tot>0 else np.nan

    return {
        'Trace':'20-sweep ensemble average',
        'Rs (MOhm)':rs,'Rm (MOhm)':rm,'Cm (pF)':cm,'Rtotal (MOhm)':rtotal,
        'I0 fitted total (pA)':i0_pa,'I transient fitted (pA)':amp_pa,
        'I raw peak (pA)':raw_peak_pa,'I steady (pA)':iss_pa,
        'Tau membrane (ms)':tau_ms,'Fit R2':r2,
        'Fit start (ms)':fit_start_ms,'Fit end (ms)':fit_end_ms,
        'Onset (ms)':onset/fs*1000.,'Offset (ms)':offset/fs*1000.
    }


def compute_test_pulse_properties(recording:Recording,pulse_amplitude_mv:float=10.,
                                  expected_onset_ms:float=20.,expected_width_ms:float=40.)->TestPulseProperties:
    avg,fs,n=_ensemble_average(recording)
    if avg is None:
        return TestPulseProperties(n_total_sweeps=len(recording.sweeps),pulse_amplitude_mv=pulse_amplitude_mv,
                                   expected_onset_ms=expected_onset_ms,expected_width_ms=expected_width_ms)
    result=_measure_average_trace(avg,fs,pulse_amplitude_mv,expected_onset_ms,expected_width_ms)
    if result is None:
        return TestPulseProperties(n_total_sweeps=len(recording.sweeps),n_valid_sweeps=n,
                                   pulse_amplitude_mv=pulse_amplitude_mv,expected_onset_ms=expected_onset_ms,
                                   expected_width_ms=expected_width_ms)
    return TestPulseProperties(series_resistance=result['Rs (MOhm)'],
        membrane_resistance=result['Rm (MOhm)'],membrane_capacitance=result['Cm (pF)'],
        n_valid_sweeps=n,n_total_sweeps=len(recording.sweeps),sweep_values=[result],
        pulse_amplitude_mv=pulse_amplitude_mv,expected_onset_ms=expected_onset_ms,
        expected_width_ms=expected_width_ms)
