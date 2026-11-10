"""Check synthetic frames, video decoding, failures, and resource cleanup without real cameras."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

import camera
from pupil import annotate


class CaptureCheck(unittest.TestCase):
    def test_recording_requires_manual_toggle_and_saves_original_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(camera, "ROOT", root):
                camera.main(["--simulate", "--headless", "--frames", "2"])
                self.assertFalse((root / "captures").exists())
                with patch.object(camera.cv2, "namedWindow"), \
                        patch.object(camera.cv2, "imshow"), \
                        patch.object(camera.cv2, "destroyAllWindows"), \
                        patch.object(camera.cv2, "getWindowProperty", return_value=1), \
                        patch.object(camera.cv2, "waitKey", side_effect=[ord("v"), -1, ord("v"), -1,
                                                                      ord("v"), -1, ord("q")]):
                    self.assertEqual(camera.main(["--simulate", "--mirror", "--frames", "10"]), 0)
            videos = sorted((root / "captures").glob("*.avi"))
            self.assertEqual(len(videos), 2)
            self.assertEqual(len(list((root / "captures").iterdir())), 2)
            for video, first_index in zip(videos, [1, 5]):
                cap = cv2.VideoCapture(str(video))
                try:
                    ok, frame = cap.read()
                    self.assertTrue(ok)
                    expected = camera.simulation_frame(first_index, 640, 480)
                    self.assertLess(np.abs(frame.astype(float) - expected).mean(), 3)
                    self.assertTrue(cap.read()[0])
                    self.assertFalse(cap.read()[0])
                finally:
                    cap.release()

    def test_capture_overlaps_inference(self):
        from threading import Event
        next_read = Event()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30.0
        reads = 0

        def read():
            nonlocal reads
            reads += 1
            if reads == 2:
                next_read.set()
            return True, np.zeros((480, 640, 3), np.uint8)

        def inference(frame, *args):
            self.assertTrue(next_read.wait(1), "Capture should overlap the first inference")
            return frame.copy(), None, "Pupil: NOT DETECTED", (0, 180, 255)

        cap.read.side_effect = read
        with patch.object(camera.cv2, "VideoCapture", return_value=cap), \
                patch("pupil.load_model", return_value=MagicMock()), \
                patch("pupil.annotate", side_effect=inference):
            self.assertEqual(camera.main(["--pupil", "--headless", "--frames", "2"]), 0)
        self.assertEqual(reads, 2)
        cap.release.assert_called_once()

    def test_pupil_coordinates_and_empty_mask(self):
        from types import SimpleNamespace
        mask = np.zeros((60, 80), np.uint8)
        cv2.circle(mask, (30, 20), 8, 255, -1)
        model = MagicMock()
        model.predict.return_value = SimpleNamespace(
            features={"pupil": {"valid": True, "area": 197, "centroid": (20., 30.)}},
            masks={"pupil": mask})
        frame = np.zeros((120, 160, 3), np.uint8)
        image, center, _, _ = annotate(frame, model, (10, 15, 80, 60))
        self.assertEqual(center, (40., 35.))
        self.assertEqual(model.predict.call_args.args[0].shape, (60, 80, 3))
        self.assertTrue(image[35, 40, 2] > 0)
        self.assertTrue(cv2.flip(image, 1)[35, 119, 2] > 0)
        self.assertFalse(frame.any())
        model.predict.return_value.features["pupil"]["valid"] = False
        _, center, label, _ = annotate(frame, model, (10, 15, 80, 60))
        self.assertIsNone(center)
        self.assertIn("NOT DETECTED", label)
        with self.assertRaises(RuntimeError):
            annotate(frame, model, (150, 100, 80, 60))

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "simulation.png"
            self.assertEqual(camera.main(["--simulate", "--headless", "--frames", "3",
                                          "--snapshot", str(image)]), 0)
            frame = cv2.imread(str(image))
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertFalse(np.array_equal(camera.simulation_frame(0, 640, 480),
                                           camera.simulation_frame(20, 640, 480)))

            video = root / "fixture.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
            self.assertTrue(writer.isOpened())
            try:
                for i in range(3):
                    writer.write(camera.simulation_frame(i, 640, 480))
            finally:
                writer.release()
            snapshot = root / "video.png"
            self.assertEqual(camera.main(["--video", str(video), "--headless", "--mirror",
                                          "--snapshot", str(snapshot)]), 0)
            self.assertEqual(cv2.imread(str(snapshot)).shape, (480, 640, 3))
            self.assertEqual(camera.main(["--video", str(root / "missing.avi"), "--headless"]), 1)

    def test_camera_failure_and_disconnect(self):
        cap = MagicMock()
        cap.isOpened.return_value = False
        with patch.object(camera.cv2, "VideoCapture", return_value=cap):
            self.assertEqual(camera.main(["--headless", "--frames", "1"]), 1)
        cap.release.assert_called_once()

        cap.reset_mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30.0
        cap.read.side_effect = [(True, camera.simulation_frame(0, 640, 480)), (False, None)]
        with patch.object(camera.cv2, "VideoCapture", return_value=cap):
            self.assertEqual(camera.main(["--headless", "--frames", "3"]), 1)
        cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
