"""Loading of AxoGraph binary files (.axgd / .axgx) into our data model."""
from __future__ import annotations
import os
import tempfile
import numpy as np
from models import Recording, Sweep, Metadata
try:
    import neo
except ImportError as e:
    raise ImportError("The 'neo' package is required to read .axgd files. Install with pip install neo.") from e

def _is_current_channel(name: str, units) -> bool:
    name_l=(name or "").lower(); units_s=str(units).lower()
    if "curr" in name_l or "pa" in units_s.split() or "amp" in units_s: return True
    try:
        import quantities as pq
        return units.dimensionality == pq.A.dimensionality
    except Exception: return False

def _is_voltage_channel(name: str, units) -> bool:
    name_l=(name or "").lower(); units_s=str(units).lower()
    if "voltage" in name_l or "vm" in name_l or "volt" in units_s: return True
    try:
        import quantities as pq
        return units.dimensionality == pq.V.dimensionality
    except Exception: return False

def load_axgd_bytes(file_bytes: bytes, original_filename: str, metadata: Metadata) -> Recording:
    suffix=os.path.splitext(original_filename)[1] or ".axgd"
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as tmp:
        tmp.write(file_bytes); tmp_path=tmp.name
    try: return load_axgd_path(tmp_path,original_filename,metadata)
    finally:
        try: os.remove(tmp_path)
        except OSError: pass

def load_axgd_path(path: str, display_filename: str, metadata: Metadata) -> Recording:
    reader=neo.io.AxographIO(path); block=reader.read_block()
    protocol_notes=""
    if block.annotations:
        protocol_notes=block.annotations.get("comment","") or ""
        notes=block.annotations.get("notes","")
        if notes: protocol_notes=f"{protocol_notes}\n\n{notes}".strip()
    sweeps=[]; sampling_rate=0.0; has_current_channel=False
    for seg_idx,seg in enumerate(block.segments):
        if not seg.analogsignals: continue
        voltage_sig=None; current_sig=None
        for sig in seg.analogsignals:
            name=sig.name or ""
            if current_sig is None and _is_current_channel(name,sig.units): current_sig=sig
            if voltage_sig is None and _is_voltage_channel(name,sig.units): voltage_sig=sig
        # Important: one-channel Test Pulse 1-Ch recordings contain Current-1 only.
        # Do not relabel that signal as voltage. For the Sweep model we keep a
        # harmless copy in voltage while preserving the correctly scaled current.
        if voltage_sig is None and current_sig is None:
            voltage_sig=seg.analogsignals[0]
        if current_sig is None and len(seg.analogsignals)>1:
            for candidate in seg.analogsignals:
                if candidate is not voltage_sig:
                    current_sig=candidate; break
        reference_sig=voltage_sig if voltage_sig is not None else current_sig
        fs=float(reference_sig.sampling_rate.rescale("Hz").magnitude); sampling_rate=fs
        time=reference_sig.times.rescale("s").magnitude.flatten()
        current=None
        if current_sig is not None:
            has_current_channel=True
            try: current=current_sig.rescale("pA").magnitude.flatten().astype(np.float64)
            except Exception: current=current_sig.magnitude.flatten().astype(np.float64)*1e12
        if voltage_sig is not None:
            try: voltage=voltage_sig.rescale("mV").magnitude.flatten()
            except Exception: voltage=voltage_sig.magnitude.flatten()*1000.0
            voltage=voltage.astype(np.float64)
        elif current is not None:
            # Test-pulse-only recording: analysis uses sweep.current.
            voltage=np.zeros_like(current,dtype=np.float64)
        else:
            continue
        sweeps.append(Sweep(index=seg_idx,time=time,voltage=voltage,current=current,sampling_rate=fs))
    return Recording(filename=display_filename,metadata=metadata,sweeps=sweeps,sampling_rate=sampling_rate,
                     protocol_notes=protocol_notes,has_current_channel=has_current_channel)
