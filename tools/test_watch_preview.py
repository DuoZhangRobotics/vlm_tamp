"""Stdlib regression checks: python -m unittest discover -s tools -p 'test_*.py'."""

import json
from pathlib import Path
import tempfile
import unittest

from export_watch_preview import asset_path, load_study, render, script_json


class WatchPreviewTests(unittest.TestCase):
    def test_analysis_strings_cannot_end_data_script(self):
        text = '</script><script>throw Error("injected")</script> & \u2028'
        encoded = script_json({"explanation": text})
        self.assertNotIn("</script>", encoded)
        self.assertEqual(json.loads(encoded)["explanation"], text)

    def test_referenced_assets_cannot_escape_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study"
            study.mkdir()
            (root / "outside.mp4").write_bytes(b"private")
            (study / "escape.mp4").symlink_to(root / "outside.mp4")
            for path in ("../outside.mp4", "escape.mp4", str(root / "outside.mp4")):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    asset_path(study, path)

    def test_trial_publication_race_keeps_outcome_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            status = {"checked_at": 1234, "status_text": "Running", "runs": [
                {"id": "baseline-r1-close", "label": "close", "outcome": "In progress",
                 "detail": "not-published-yet.json", "videos": []}]}
            (study / "watch-status.json").write_text(json.dumps(status))
            data, videos = load_study(study)
            self.assertEqual(data["runs"][0]["outcome"], "In progress")
            self.assertIn("not available", data["runs"][0]["detail_error"])
            self.assertEqual(videos, [])
            self.assertIn("Running", render(data))


if __name__ == "__main__":
    unittest.main()
