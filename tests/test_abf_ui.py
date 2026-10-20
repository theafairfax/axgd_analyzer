import unittest
from streamlit.testing.v1 import AppTest


class ABFNavigationTests(unittest.TestCase):
    def test_abf_has_three_protocol_tabs(self):
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        app.sidebar.radio[0].set_value("ABF").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([tab.label for tab in app.tabs],
                         ["Average RMP monitor", "Step rheobase", "Ramp protocol"])
        self.assertEqual(len(app.get("file_uploader")), 3)


if __name__ == "__main__":
    unittest.main()
