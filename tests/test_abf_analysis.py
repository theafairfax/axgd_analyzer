import unittest
import numpy as np

from abf_analysis import analyze_events, event_averages, rmp_summary


class ActiveProtocolTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(0., .1, .0001)
        self.v = np.interp(self.t*1000,
            [0, 8, 9, 10, 11, 14, 22, 38, 39, 40, 41, 44, 52, 100],
            [-65, -65, -60, 30, -65, -75, -65, -65, -60, 35, -65, -78, -65, -65])

    def test_all_events_and_average_preserve_absolute_peaks(self):
        rows = analyze_events(self.t, self.v, 0., .09)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["SPIKE Peak Membrane Voltage-1 (mV)"], 30.)
        self.assertAlmostEqual(rows[0]["AHP Peak Membrane Voltage-1 (mV)"], -75.)
        self.assertAlmostEqual(rows[1]["AHP Peak Membrane Voltage-1 (mV)"], -78.)
        self.assertLess(rows[0]["SPIKE Onset (ms)"], rows[0]["SPIKE Location (ms)"])
        self.assertGreater(rows[0]["SPIKE Width (ms)"], 0.)
        self.assertGreater(rows[0]["AHP Width (ms)"], 0.)
        averages = {r["Metric"]: r for r in event_averages(rows)}
        self.assertAlmostEqual(averages["SPIKE Peak Membrane Voltage-1 (mV)"]["Mean"], 32.5)
        self.assertAlmostEqual(averages["AHP Peak Membrane Voltage-1 (mV)"]["Mean"], -76.5)
        self.assertEqual(averages["AHP Width (ms)"]["Valid events"], 2)

    def test_window_excludes_other_sweeps_and_first_ramp_event(self):
        rows = analyze_events(self.t, self.v, .02, .09)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["SPIKE Location (ms)"], 40.)
        first = analyze_events(self.t, self.v, 0., .09)[:1]
        self.assertAlmostEqual(first[0]["SPIKE Location (ms)"], 10.)

    def test_missing_ahp_stays_missing(self):
        rows = analyze_events(self.t, self.v, 0., .011)
        self.assertEqual(len(rows), 1)
        self.assertTrue(np.isnan(rows[0]["AHP Location (ms)"]))
        averages = {r["Metric"]: r for r in event_averages(rows)}
        self.assertEqual(averages["AHP Width (ms)"]["Valid events"], 0)

    def test_rmp_settling_window_and_frequency(self):
        t = np.arange(0., 60., .01)
        v = np.where(t < 30., -50., -70.)
        result = rmp_summary(t, v, 30., 60.)
        self.assertAlmostEqual(result["Average RMP (mV)"], -70.)
        self.assertAlmostEqual(result["Window duration (s)"], 30.)
        self.assertEqual(result["AP count"], 0)
        self.assertEqual(result["Spiking frequency (Hz)"], 0.)
        with self.assertRaises(ValueError):
            rmp_summary(t, v, 61., 62.)


if __name__ == "__main__":
    unittest.main()
