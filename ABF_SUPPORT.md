# ABF current-clamp protocols

Select ABF at the top of the sidebar. There are three tabs, each with its own
file upload, channel choices, analysis settings, and exports. Existing AXGD
analysis is unchanged.

## Average RMP monitor (example 0000)

Select a voltage channel and monitor sweep. The default settling exclusion is
30 seconds; change it for shorter recordings or select a window end.
Average RMP is the arithmetic mean of every sample in that window, including
spikes in active neurons. SD, AP count, actual analyzed duration and spike
frequency are also reported. No liquid-junction-potential correction is applied.

## Step rheobase (example 0002)

The entered protocol defaults to -100 pA plus 20 pA per zero-based sweep
(-100 through +240 pA for 18 sweeps). Injection start/end defaults of
100/600 ms are placeholders: verify them against the recording.
Alternatively select a measured ADC current channel or an ABF DAC command.
Current from a waveform is reported as the median plateau minus pre-step
baseline. Automatic injection windows use deviations from the initial baseline;
verify these on the shaded trace, especially for noisy or complex protocols.
Zero-current sweeps use the entered timing.

The current-vs-AP-count plot counts peaks during injection only. Step rheobase
is the lowest nonnegative tested current that elicited an AP. Select any sweep
to view all detected APs and per-event dynamics plus metric averages.
Rebound APs outside injection are not counted in this plot.

## Ramp (example 0001)

Select a current channel/command, or explicitly verify manually entered linear
ramp endpoints and injection timing. The first detected AP during injection
provides AP/AHP dynamics. Ramp rheobase is the absolute current at that AP's
peak, following the supplied methods passage; it is not baseline-subtracted.
A phase plot shows Vm versus dV/dt near that AP. CSV exports include the
first-event metrics, ramp rheobase, and phase-plot data.

DAC waveforms represent programmed stimuli, not recorded current. ADC/DAC
units are checked and converted to pA; voltage is converted to mV. Missing
command waveforms cause an error rather than an assumed injected current.
pyABF associates each command with the corresponding channel index.

## Measurement definitions

- AP detection: adjustable peak height, prominence and refractory separation.
- AP onset: first sample at or above the chosen dV/dt threshold before
  the maximum upstroke within 5 ms of the peak, after the preceding peak.
- SPIKE peak: absolute Vm at peak. Separate AP amplitude equals peak minus
  onset Vm. Rise is the interpolated 10–90% interval; width is full width at half that amplitude.
- AHP onset: repolarizing AP's downward return through the same onset Vm.
  AHP peak: absolute minimum Vm thereafter. Separate AHP amplitude equals
  trough minus AP onset Vm (negative for hyperpolarization).
- AHP rise: downward 10–90% interval from onset Vm toward the trough. Width: full width at half depth below AP onset Vm.
- Locations/onsets and fractional crossings: absolute sweep ms, plus relative
  columns centered on each AP peak or AHP trough (location = 0). Rise and
  width are durations, independent of time origin.
- AHP windows stop at the next AP onset, injection end or configurable maximum
  post-peak duration (default 50 ms). Truncated/unresolved troughs and widths
  remain missing; missing values never contribute zero to averages.
- Averages are arithmetic means of valid per-event metrics, not metrics of an
  averaged waveform. Valid-event counts accompany every mean.

These are explicit initial definitions, not a claim of exact Clampfit or
AxoGraph event-template equivalence. Adjust detection settings and compare
representative events manually. Short interspike intervals may make some AHP
measurements unavailable.

## Dependencies, memory and validation

Uses [pyABF](https://github.com/swharden/pyABF) for ABF1/ABF2 parsing, scaling,
sweeps and commands. [ABF Analytics](https://github.com/Ciwiruk/abf_analytics)
was consulted for signal/channel inspection; its Rust parser was not copied.

Header reads avoid loading signal arrays. Protocol loading uses pyABF setSweep,
which loads a full scaled array, then copies selected Vm/current sweeps.
Caches are bounded to three entries; large recordings can still exceed memory.
Plots are subsampled to approximately 30,000 points and can hide brief events.
All detection and statistics use full-resolution samples.

Install requirements and run:

```sh
python -m compileall -q .
python -m unittest discover -s tests
streamlit run app.py
```

Tests cover generated ABF1 input/scaling and synthetic AP/AHP trains, window
selection, missing AHPs, averaging and RMP settling exclusion. Automated syntax and test checks run in GitHub Actions, including a Streamlit
navigation check for the three ABF tabs. The uploaded 0000/0001/0002 recordings
could not be inspected because the local execution environment was unavailable.
ABF2 sample validation and manual Clampfit comparisons remain outstanding.

## Isolated waveform review

Step and ramp protocols show the first detected AP in the selected sweep in
separate, always-visible AP spike dynamics and AHP dynamics sections.
The AHP section includes trough voltage, signed amplitude from onset, AP-peak
to AHP-trough delay, 10–90% rise and half-width. When detection is unresolved,
it still displays the post-spike search trace, detection status and guidance. Colored vertical lines intersect the
full-resolution waveform at location, onset, 10%, 90%, and both half-amplitude
crossings. The time axis switches between absolute sweep time and time relative
to each event's own location. Missing crossings remain unresolved.

Expand **Adjust marker times**, drag the time sliders, and apply to review
manual landmarks; durations and relative times update from these endpoints.
These are explicit manual timing overrides, not a new automatic fit. They do
not change detection or protocol averages. Download the displayed markers CSV
for the reviewed values, or reset to detected markers. Changes to the file,
sweep, channel or detection results start a fresh review.
