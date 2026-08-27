"""Dedicated test-pulse analysis for patch-clamp Rs/Rm/Cm measurements."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit
from models import Recording, Sweep
from protocol import detect_steps

@dataclass
class TestPulseProperties:
    series_resistance: Optional[float] = None
    membrane_resistance: Optional[float] = None
    membrane_capacitance: Optional[float] = None
    n_valid_sweeps: int = 0
    sweep_values: list = None

    def __post_init__(self):
        if self.sweep_values is None:
            self.sweep_values = []

def _double_exp(t, v_inf, a1, tau1, a2, tau2):
    return v_inf + a1*np.exp(-t/tau1) + a2*np.exp(-t/tau2)

def _analyze_sweep(sweep: Sweep):
    if sweep.current is None or sweep.step_onset_idx is None or sweep.step_offset_idx is None:
        return None
    onset, offset, fs = sweep.step_onset_idx, sweep.step_offset_idx, sweep.sampling_rate
    pre = max(onset - max(int(0.002*fs), 5), 0)
    i0 = float(np.median(sweep.current[pre:onset])) if onset > pre else float(sweep.current[0])
    pulse_end = min(onset + max(int(0.010*fs), 20), offset)
    i1 = float(np.median(sweep.current[max(onset+2, onset):pulse_end]))
    di = i1 - i0
    if abs(di) < 1e-6:
        return None
    v = sweep.voltage[onset:min(offset, onset + max(int(0.030*fs), 30))]
    if len(v) < 20:
        return None
    t = np.arange(len(v))/fs*1000.0
    v0, vend = float(v[0]), float(np.median(v[-max(3, len(v)//10):]))
    delta = v0-vend
    try:
        p0=[vend, delta*0.35, 0.15, delta*0.65, 5.0]
        bounds=([-250,-500,0.005,-500,0.05],[250,500,5,500,100])
        popt,_=curve_fit(_double_exp,t,v,p0=p0,bounds=bounds,maxfev=20000)
        vinf,a1,t1,a2,t2=popt
        if t1 > t2: a1,t1,a2,t2=a2,t2,a1,t1
        # mV/pA = 1000 MOhm
        rs=abs(a1/di)*1000.0
        rm=abs(a2/di)*1000.0
        cm=t2*1000.0/rm if rm > 0 else None
        if not (0 < rs < 1000 and 0 < rm < 10000 and cm is not None and 0 < cm < 10000):
            return None
        return rs,rm,cm
    except Exception:
        return None

def compute_test_pulse_properties(recording: Recording) -> TestPulseProperties:
    detect_steps(recording)
    values=[]
    for sweep in recording.sweeps:
        result=_analyze_sweep(sweep)
        if result is not None:
            values.append((sweep.index,*result))
    if not values:
        return TestPulseProperties()
    arr=np.asarray([x[1:] for x in values],dtype=float)
    return TestPulseProperties(
        series_resistance=float(np.median(arr[:,0])),
        membrane_resistance=float(np.median(arr[:,1])),
        membrane_capacitance=float(np.median(arr[:,2])),
        n_valid_sweeps=len(values), sweep_values=values)
