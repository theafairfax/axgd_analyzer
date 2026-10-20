# Initial ABF support

Choose **AXGD** or **ABF** using the File format control at the top of the sidebar.
AXGD retains the existing test-pulse/rheobase workflow.

ABF accepts one .abf recording and uses pyABF to read ABF1/ABF2 headers and
scaled ADC signals. Select recorded channels and up to 20 consecutive sweeps,
then enable **Load selected traces**. Sweep numbers and channel numbers are
zero-based. Inspect per-channel plots, download full-resolution sweep statistics,
or prepare a full-resolution CSV for an individual trace. Time is seconds
relative to each sweep; signal values retain the file's native units.

This initial workspace does not calculate rheobase, AP/AHP features, Rs, Rm,
or Cm for ABF. Those require explicit channel/protocol mapping. It never treats a
DAC command as a measured ADC signal or applies the AXGD assumed current series.

## Memory and plotting

Header inspection uses loadData=False. Trace loading uses pyABF setSweep, which
loads its full scaled data array internally. Only selected traces are retained;
the header and trace caches each retain at most one entry. Large recordings may
still exceed hosted memory limits. Plots are subsampled to at most 20,000 points
per trace, which can hide brief events. Exports/statistics use every sample.
Trace CSV generation occurs only on request.

## References

- [pyABF](https://github.com/swharden/pyABF): binary parsing, channel metadata,
  scaling, and sweep-relative time via ABF/setSweep/sweepX/sweepY.
- [ABF Analytics](https://github.com/Ciwiruk/abf_analytics): consulted for the
  channel-selection and raw-signal inspection workflow. This Rust desktop
  application is not a runtime dependency; its parser and analytics were not copied.

## Verification

Install requirements, then run:

```sh
python -m compileall -q .
python -m unittest discover -s tests
streamlit run app.py
```

The tests write real ABF1 recordings with pyABF and verify units, scaling, time,
sweep selection, invalid input handling, temporary-file cleanup, and statistics.
ABF2 and representative user recordings still need validation.
