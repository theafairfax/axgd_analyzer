"""Run with python -m unittest discover -s tests."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pyabf.abfWriter import writeABF1

from abf_io import ABFTrace, inspect_abf_bytes, load_abf_traces, open_abf, summarize_traces


class ABFInputTests(unittest.TestCase):
    def make_abf(self, units):
        expected = np.array([[-70., -60., -50., -40.], [10., 20., 30., 40.]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.abf"
            writeABF1(expected, str(path), 10000, units=units)
            return path.read_bytes(), expected

    def test_real_abf1_units_scaling_and_sweep_relative_time(self):
        for units in ("mV", "pA"):
            with self.subTest(units=units):
                data, expected = self.make_abf(units)
                info = inspect_abf_bytes(data)
                self.assertEqual(info["sweep_count"], 2)
                self.assertEqual(info["channels"][0]["unit"], units)
                self.assertEqual(info["sampling_rate_hz"], 10000)
                traces = load_abf_traces(data, (1, 0), (0,))
                self.assertEqual([t.sweep for t in traces], [1, 0])
                np.testing.assert_allclose(traces[0].values, expected[1], atol=0.01)
                np.testing.assert_allclose(traces[1].values, expected[0], atol=0.01)
                np.testing.assert_allclose(traces[0].time, np.arange(4) / 10000)
                self.assertEqual(traces[0].unit, units)

    def test_invalid_header_and_selection(self):
        with self.assertRaisesRegex(ValueError, "header"):
            inspect_abf_bytes(b"not an ABF")
        data, _ = self.make_abf("mV")
        for sweeps, channels in [((2,), (0,)), ((0,), (1,))]:
            with self.assertRaises(ValueError):
                load_abf_traces(data, sweeps, channels)

    def test_temporary_file_removed_on_parser_failure(self):
        paths = []
        def fail(path, **kwargs):
            paths.append(path)
            self.assertTrue(os.path.isfile(path))
            raise ValueError("broken recording")
        with patch("pyabf.ABF", side_effect=fail):
            with self.assertRaisesRegex(ValueError, "broken recording"):
                with open_abf(b"ABF2broken"):
                    pass
        self.assertEqual(len(paths), 1)
        self.assertFalse(os.path.exists(paths[0]))

    def test_statistics_preserve_units_and_ignore_nonfinite_samples(self):
        trace = ABFTrace(0, 1, "Current", "nA", np.arange(4) / 1000,
                         np.array([-2., np.nan, 4., np.inf]))
        row = summarize_traces([trace])[0]
        self.assertEqual(row["Unit"], "nA")
        self.assertEqual(row["Finite samples"], 2)
        self.assertEqual(row["Mean"], 1.)
        self.assertEqual(row["Peak-to-peak"], 6.)


if __name__ == "__main__":
    unittest.main()
