"""Passive membrane analysis from AxoGraph test-pulse recordings.

For voltage-clamp test pulses the measured current often contains a very fast
pipette/electrode-capacitance artifact followed by the slower whole-cell
membrane charging transient.  Using the absolute first current peak therefore
systematically underestimates series resistance.  We separate those components
with a biexponential fit and use the slower component for Rs/Rm/Cm.
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
    sweep_values: list = field(default_factory=list)
    pulse_amplitude_mv: float = 10.0


def _double_exp(t, i_ss, a_fast, tau_fast, a_mem, tau_mem):
    return i_ss + a_fast*np.exp(-t/tau_fast) + a_mem*np.exp(-t/tau_mem)


def _find_test_pulse_window(signal, fs):
    """Find a pair of strong transitions separated by 5-80 ms.

    Pair scoring fixes the prior failure mode where the larger pulse OFF
    transition was incorrectly labeled as the onset.
    """
    x=np.asarray(signal,dtype=float)
    if len(x)<30 or fs<=0:return None,None
    # 50-us smoothing at high sampling rates, otherwise one sample.
    w=max(int(round(0.00005*fs)),1)
    smooth=np.convolve(x,np.ones(w)/w,mode='same') if w>1 else x
    d=np.abs(np.diff(smooth))
    margin=max(int(.002*fs),2)
    lo0,hi0=margin,len(d)-margin
    if hi0<=lo0:return None,None
    n_candidates=min(20,max(4,(hi0-lo0)//10))
    idx=np.argpartition(d[lo0:hi0],-n_candidates)[-n_candidates:]+lo0
    idx=sorted(int(i) for i in idx)
    min_sep=max(int(.005*fs),2); max_sep=max(int(.080*fs),min_sep+1)
    best=None
    for i in idx:
        for j in idx:
            if j<=i:continue
            sep=j-i
            if min_sep<=sep<=max_sep:
                score=float(d[i]+d[j])
                # Small preference for earlier valid pairs when scores are similar.
                score-=1e-6*i
                if best is None or score>best[0]:best=(score,i,j)
    if best is None:return None,None
    return best[1]+1,best[2]+1


def _analyze_voltage_clamp_pulse(sweep:Sweep,pulse_amplitude_mv:float):
    raw=sweep.current if sweep.current is not None else sweep.voltage
    if raw is None:return None
    x=np.asarray(raw,dtype=float);fs=float(sweep.sampling_rate)
    onset,offset=_find_test_pulse_window(x,fs)
    if onset is None or offset is None or offset-onset<20:return None
    pre_n=max(int(.005*fs),5)
    base=float(np.median(x[max(0,onset-pre_n):onset]))
    pulse=x[onset:offset]-base
    if len(pulse)<20:return None
    early=min(max(int(.002*fs),8),len(pulse))
    sign=1. if abs(np.max(pulse[:early]))>=abs(np.min(pulse[:early])) else -1.
    y=pulse*sign
    tail_n=min(max(int(.003*fs),5),max(len(y)//5,5))
    iss_obs=float(np.median(y[-tail_n:]))
    if iss_obs<=0:return None
    t=np.arange(len(y))/fs*1000.
    # Fit only the pulse, excluding the last 0.25 ms before the OFF transient.
    trim=max(int(.00025*fs),1); fit_n=max(len(y)-trim,10)
    tf=t[:fit_n];yf=y[:fit_n]
    peak=float(np.max(y[:min(len(y),max(int(.0015*fs),10))]))
    amp=max(peak-iss_obs,1e-9)
    # Fast term captures pipette capacitance; slow term is membrane charging.
    p0=[iss_obs,amp*.45,.04,amp*.55,.5]
    try:
        popt,_=curve_fit(_double_exp,tf,yf,p0=p0,
            bounds=([0,0,.002,0,.03],[np.inf,np.inf,.5,np.inf,20.]),maxfev=30000)
        iss,af,tfst,am,tmem=[float(v) for v in popt]
        if tfst>tmem: af,tfst,am,tmem=am,tmem,af,tfst
    except Exception:
        return None
    # Current should already be pA from axgd_io. Retain defensive A->pA scaling.
    scale=1e12 if max(abs(iss),abs(am),abs(af))<1e-3 else 1.
    iss_pa=iss*scale; imem0_pa=(iss+am)*scale; raw_peak_pa=peak*scale
    if iss_pa<=0 or imem0_pa<=iss_pa:return None
    rs=abs(pulse_amplitude_mv/imem0_pa)*1000.
    rtot=abs(pulse_amplitude_mv/iss_pa)*1000.
    rm=rtot-rs
    if not (1<rs<500 and 5<rm<5000):return None
    # For a voltage-clamped Rs-(Rm||Cm) circuit, the membrane exponential is
    # tau = Cm * (Rs || Rm), NOT Cm*Rm.
    r_parallel=(rs*rm)/(rs+rm)
    cm=tmem*1000./r_parallel if r_parallel>0 else None
    if cm is None or not (1<cm<1000):return None
    return {
        'Rs (MOhm)':rs,'Rm (MOhm)':rm,'Cm (pF)':cm,
        'I membrane t0 (pA)':imem0_pa,'I raw peak (pA)':raw_peak_pa,
        'I steady (pA)':iss_pa,'Tau membrane (ms)':tmem,
        'Tau fast artifact (ms)':tfst,'Onset (ms)':onset/fs*1000.,
        'Offset (ms)':offset/fs*1000.,'Mode':'voltage-clamp biexponential'
    }


def compute_test_pulse_properties(recording:Recording,pulse_amplitude_mv:float=10.)->TestPulseProperties:
    vals=[]
    for sw in recording.sweeps:
        r=_analyze_voltage_clamp_pulse(sw,pulse_amplitude_mv)
        if r is not None:
            r={'Sweep':sw.index,**r};vals.append(r)
    if not vals:return TestPulseProperties(pulse_amplitude_mv=pulse_amplitude_mv)
    # Mean matches the historical spreadsheet convention; QC table exposes
    # individual sweeps so obvious failed fits can be identified.
    return TestPulseProperties(
        series_resistance=float(np.mean([v['Rs (MOhm)'] for v in vals])),
        membrane_resistance=float(np.mean([v['Rm (MOhm)'] for v in vals])),
        membrane_capacitance=float(np.mean([v['Cm (pF)'] for v in vals])),
        n_valid_sweeps=len(vals),sweep_values=vals,pulse_amplitude_mv=pulse_amplitude_mv)
