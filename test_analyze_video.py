"""Offline export checks with known geometry and actual video encode/decode."""

import csv
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import analyze_video


class OfflineAnalysisCheck(unittest.TestCase):
    def test_csv_geometry_missing_detection_and_review_video(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "eye.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.full((120, 160, 3), 80, np.uint8))
            writer.release()
            mask = np.zeros((60, 80), np.uint8)
            cv2.circle(mask, (30, 20), 8, 255, -1)
            area = int(np.count_nonzero(mask))
            model = MagicMock()
            model.predict.side_effect = [
                SimpleNamespace(features={"pupil": {"valid": True, "area": area,
                                                      "centroid": (20., 30.)}}, masks={"pupil": mask}),
                SimpleNamespace(features={"pupil": {"valid": False}}, masks={}),
                SimpleNamespace(features={"pupil": {"valid": True, "area": 1,
                                                      "centroid": (20., 30.)}}, masks={})]
            output = video.with_suffix(".pupil.csv")
            self.assertEqual(analyze_video.analyze(video, output, model, (10, 15, 80, 60)), 3)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(list(rows[0]), analyze_video.FIELDS)
            self.assertEqual([row["frame_id"] for row in rows], ["0", "1", "2"])
            self.assertAlmostEqual(float(rows[2]["timestamp"]), .08)
            self.assertEqual(float(rows[0]["pupil_x"]), 40)
            self.assertEqual(float(rows[0]["pupil_y"]), 35)
            self.assertAlmostEqual(float(rows[0]["pupil_radius"]), math.sqrt(area/math.pi))
            self.assertAlmostEqual(float(rows[0]["pupil_diameter"]), 2*math.sqrt(area/math.pi))
            self.assertEqual(rows[0]["confidence"], "")
            for row in rows[1:]:
                self.assertEqual(row["detected"], "0")
                for key in ["pupil_x", "pupil_y", "pupil_radius", "pupil_diameter", "pupil_area"]:
                    self.assertEqual(row[key], "")
            self.assertEqual(model.predict.call_args.args[0].shape, (60, 80, 3))
            cap = cv2.VideoCapture(str(output.with_suffix(".annotated.avi")))
            try:
                decoded = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    decoded.append(frame)
                self.assertEqual(len(decoded), 3)
                self.assertEqual(decoded[0].shape, (120, 160, 3))
                self.assertGreater(int(decoded[0][35, 40, 2]), 150)
            finally:
                cap.release()
            with self.assertRaises(FileExistsError):
                analyze_video.analyze(video, output, model)
            self.assertEqual(model.predict.call_count, 3)
            with self.assertRaisesRegex(RuntimeError, "different paths"):
                analyze_video.analyze(video, video, model)

    def test_timing_fallback_and_error_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "eye.avi"
            video.touch()
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: 20 if prop == cv2.CAP_PROP_FPS else 0
            frame = np.zeros((120, 160, 3), np.uint8)
            cap.read.side_effect = [(True, frame), (True, frame), (False, None)]
            model = MagicMock()
            model.predict.return_value = SimpleNamespace(features={"pupil": {"valid": False}})
            output = video.with_suffix(".csv")
            with patch.object(analyze_video.cv2, "VideoCapture", return_value=cap):
                analyze_video.analyze(video, output, model)
            cap.release.assert_called_once()
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[1]["timestamp"]), .05)
            self.assertEqual(rows[1]["timestamp_source"], "fps_estimate")
            cap.reset_mock()
            cap.read.side_effect = [(True, frame)]
            with patch.object(analyze_video.cv2, "VideoCapture", return_value=cap):
                with self.assertRaisesRegex(RuntimeError, "exceeds"):
                    analyze_video.analyze(video, video.with_suffix(".bad.csv"), model, (150, 0, 80, 60))
            cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
