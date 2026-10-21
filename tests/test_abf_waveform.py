import unittest
import numpy as np
from streamlit.testing.v1 import AppTest
from abf_analysis import analyze_events
from abf_waveform import validate_landmarks, landmark_table


class WaveformTests(unittest.TestCase):
    def setUp(self):
        self.tm = np.arange(0., 30., .01)
        self.v = np.interp(self.tm, [0, 8, 10, 12, 16, 24, 30], [-65, -65, 35, -65, -85, -65, -65])
        self.row = analyze_events(self.tm/1000, self.v, 0, .029)[0]

    def test_fractional_rises_width_and_relative_times(self):
        r = self.row
        self.assertAlmostEqual(r['SPIKE Rise (ms)'], 1.6, places=2)
        self.assertAlmostEqual(r['AHP Rise (ms)'], 3.2, places=2)
        for block in ('SPIKE', 'AHP'):
            self.assertEqual(r[f'{block} Location relative (ms)'], 0.)
            self.assertLess(r[f'{block} Onset relative (ms)'], 0.)
            self.assertAlmostEqual(r[f'{block} Width (ms)'], r[f'{block} Half fall (ms)']-r[f'{block} Half rise (ms)'])

    def test_invalid_edits_and_relative_table(self):
        vals = {name: self.row[f'SPIKE {name} (ms)'] for name in
                ('Location', 'Onset', 'Rise 10%', 'Rise 90%', 'Half rise', 'Half fall')}
        self.assertTrue(validate_landmarks(vals, 5., 16.))
        table = landmark_table(vals, self.tm, self.v).set_index('Marker')
        self.assertEqual(table.loc['Location', 'Relative to location (ms)'], 0.)
        vals['Location'] = 7.
        self.assertFalse(validate_landmarks(vals, 5., 16.))

    def test_inspector_apply_reset_and_axis(self):
        app = AppTest.from_string('''
import numpy as np
from abf_analysis import analyze_events
from abf_waveform import render_event_inspector
t = np.arange(0., .03, .00001)
v = np.interp(t*1000, [0,8,10,12,16,24,30], [-65,-65,35,-65,-85,-65,-65])
sw = dict(time=t, voltage=v, index=0)
render_event_inspector(sw, analyze_events(t,v,0,.029), 'test')
''').run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn('AP spike dynamics', [h.value for h in app.subheader])
        self.assertIn('AHP dynamics', [h.value for h in app.subheader])
        self.assertIn('AHP trough Vm (mV)', [m.label for m in app.metric])
        self.assertIn('AHP amplitude from onset (mV)', [m.label for m in app.metric])
        self.assertEqual(len(app.slider), 12)
        app.slider[0].set_value(10.01)
        next(b for b in app.button if b.label == 'Apply marker adjustments').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertAlmostEqual(app.dataframe[0].value.iloc[0]['Sweep time (ms)'], 10.01)
        app.slider[6].set_value(16.01)
        [b for b in app.button if b.label == 'Apply marker adjustments'][1].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertAlmostEqual(app.dataframe[1].value.iloc[0]['Sweep time (ms)'], 16.01)
        self.assertAlmostEqual(app.dataframe[1].value.iloc[0]['Relative to location (ms)'], 0.)
        app.radio[0].set_value('Absolute sweep time').run()
        self.assertEqual(len(app.exception), 0)
        next(b for b in app.button if b.label == 'Reset to detected markers').click().run()
        self.assertAlmostEqual(app.dataframe[0].value.iloc[0]['Sweep time (ms)'], 10.)

    def test_unresolved_ahp_keeps_visible_section_and_trace(self):
        app = AppTest.from_string('''
import numpy as np
from abf_analysis import analyze_events
from abf_waveform import render_event_inspector
t = np.arange(0., .03, .00001)
v = np.interp(t*1000, [0,8,10,12,30], [-65,-65,35,-65,-80])
render_event_inspector(dict(time=t, voltage=v, index=0), analyze_events(t,v,0,.029), 'missing')
''').run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn('AHP dynamics', [h.value for h in app.subheader])
        self.assertIn('unresolved', app.warning[0].value)
        self.assertEqual(len(app.get('image')), 2)


if __name__ == '__main__':
    unittest.main()
