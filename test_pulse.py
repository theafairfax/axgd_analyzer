"""Passive membrane analysis from AxoGraph test-pulse recordings.

The standard 0001 Test Pulse 1-Ch protocol used for these recordings applies a
10 mV voltage command beginning at about 20 ms and ending about 40 ms later.
We use that protocol timing as a strong prior, then fit the measured current
transient with a fast electrode-artifact exponential plus the slower membrane
charging exponential.
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


def _double_exp(t, i_ss, a_fast, tau_fast, a_mem, tau_mem):
    return i_ss + a_fast*np.exp(-t/tau_fast) + a_mem*np.exp(-t/tau_mem)


def _edge_near(d, center_idx, radius):
    lo=max(0,int(center_idx-radius)); hi=min(len(d),int(center_idx+radius+1))
    if hi<=lo:return None
    return lo+int(np.argmax(d[lo:hi]))


def _find_test_pulse_window(signal, fs, expected_onset_ms=20., expected_width_ms=40., tolerance_ms=6.):
    """Detect ON/OFF transitions near the known protocol times.

    Earlier versions simply chose the two largest derivatives and could label
    the large OFF transient as the onset. Anchoring both edges independently
    around 20 ms and 60 ms avoids that failure while still allowing several ms
    of acquisition/protocol jitter.
    """
    x=np.asarray(signal,dtype=float)
    if len(x)<30 or fs<=0:return None,None
    w=max(int(round(.00005*fs)),1)
    smooth=np.convolve(x,np.ones(w)/w,mode='same') if w>1 else x
    d=np.abs(np.diff(smooth))
    onset_guess=expected_onset_ms/1000.*fs
    offset_guess=(expected_onset_ms+expected_width_ms)/1000.*fs
    radius=max(int(tolerance_ms/1000.*fs),3)
    onset=_edge_near(d,onset_guess,radius); offset=_edge_near(d,offset_guess,radius)
    if onset is not None and offset is not None and offset>onset+max(int(.005*fs),5):
        return onset+1,offset+1
    # Fallback for non-standard timing: choose an early edge, then a later edge
    # separated by approximately the expected pulse width.
    margin=max(int(.002*fs),2)
    early_hi=min(len(d)-margin,int(.5*len(d)))
    if early_hi<=margin:return None,None
    onset=margin+int(np.argmax(d[margin:early_hi]))
    target=onset+int(expected_width_ms/1000.*fs)
    offset=_edge_near(d,target,max(int(.010*fs),5))
    if offset is None or offset<=onset:return None,None
    return onset+1,offset+1


def _analyze_voltage_clamp_pulse(sweep:Sweep,pulse_amplitude_mv:float,expected_onset_ms:float,expected_width_ms:float):
    raw=sweep.current if sweep.current is not None else sweep.voltage
    if raw is None:return None
    x=np.asarray(raw,dtype=float);fs=float(sweep.sampling_rate)
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
    iss_obs=float(np.median(y[-tail_n:]))
    if iss_obs<=0:return None
    t=np.arange(len(y))/fs*1000.
    trim=max(int(.0004*fs),1); fit_n=max(len(y)-trim,10)
    tf=t[:fit_n];yf=y[:fit_n]
    raw_peak=float(np.max(y[:min(len(y),max(int(.0015*fs),10))]))
    amp=max(raw_peak-iss_obs,1e-9)
    try:
        popt,_=curve_fit(_double_exp,tf,yf,
            p0=[iss_obs,amp*.45,.04,amp*.55,.55],
            bounds=([0,0,.002,0,.03],[np.inf,np.inf,.5,np.inf,20.]),
            maxfev=30000)
        iss,af,tfast,am,tmem=[float(v) for v in popt]
        if tfast>tmem:af,tfast,am,tmem=am,tmem,af,tfast
    except Exception:return None
    scale=1e12 if max(abs(iss),abs(am),abs(af))<1e-3 else 1.
    iss_pa=iss*scale; imem0_pa=(iss+am)*scale; raw_peak_pa=raw_peak*scale
    if iss_pa<=0 or imem0_pa<=iss_pa:return None
    # Whole-cell voltage-clamp equivalent circuit:
    # I(0+) = dV/Rs ; I(inf) = dV/(Rs+Rm)
    rs=abs(pulse_amplitude_mv/imem0_pa)*1000.
    rtot=abs(pulse_amplitude_mv/iss_pa)*1000.
    rm=rtot-rs
    if not (1<rs<500 and 5<rm<5000):return None
    rpar=(rs*rm)/(rs+rm)
    cm=tmem*1000./rpar if rpar>0 else None
    if cm is None or not (1<cm<1000):return None
    # Reject fits where the slower membrane component is negligible or the
    # two time constants are not meaningfully separated.
    membrane_fraction=am/max(af+am,1e-12)
    if membrane_fraction<.05 or tmem/max(tfast,1e-9)<1.5:return None
    return {'Rs (MOhm)':rs,'Rm (MOhm)':rm,'Cm (pF)':cm,
        'I membrane t0 (pA)':imem0_pa,'I raw peak (pA)':raw_peak_pa,'I steady (pA)':iss_pa,
        'Tau membrane (ms)':tmem,'Tau fast artifact (ms)':tfast,'Membrane fraction':membrane_fraction,
        'Onset (ms)':onset/fs*1000.,'Offset (ms)':offset/fs*1000.}


def _robust_keep(values, key):
    arr=np.asarray([v[key] for v in values],dtype=float)
    if len(arr)<4:return np.ones(len(arr),dtype=bool)
    med=float(np.median(arr)); mad=float(np.median(np.abs(arr-med)))
    if mad<=1e-12:return np.ones(len(arr),dtype=bool)
    z=np.abs(arr-med)/(1.4826*mad)
    return z<=3.5


def compute_test_pulse_properties(recording:Recording,pulse_amplitude_mv:float=10.,expected_onset_ms:float=20.,expected_width_ms:float=40.)->TestPulseProperties:
    vals=[]
    for sw in recording.sweeps:
        r=_analyze_voltage_clamp_pulse(sw,pulse_amplitude_mv,expected_onset_ms,expected_width_ms)
        if r is not None:vals.append({'Sweep':sw.index,**r})
    if not vals:return TestPulseProperties(n_total_sweeps=len(recording.sweeps),pulse_amplitude_mv=pulse_amplitude_mv,expected_onset_ms=expected_onset_ms,expected_width_ms=expected_width_ms)
    keep=np.ones(len(vals),dtype=bool)
    for key in ('Rs (MOhm)','Rm (MOhm)','Cm (pF)'):
        keep &= _robust_keep(vals,key)
    for v,k in zip(vals,keep):v['Included in average']=bool(k)
    accepted=[v for v,k in zip(vals,keep) if k]
    if not accepted:accepted=vals
    return TestPulseProperties(series_resistance=float(np.mean([v['Rs (MOhm)'] for v in accepted])),
        membrane_resistance=float(np.mean([v['Rm (MOhm)'] for v in accepted])),
        membrane_capacitance=float(np.mean([v['Cm (pF)'] for v in accepted])),n_valid_sweeps=len(accepted),
        n_total_sweeps=len(recording.sweeps),sweep_values=vals,pulse_amplitude_mv=pulse_amplitude_mv,
        expected_onset_ms=expected_onset_ms,expected_width_ms=expected_width_ms)
