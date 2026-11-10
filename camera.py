"""HOSA capture: USB/UVC cameras, local videos, or labeled simulation frames."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WINDOW = "HOSA Camera | Q: quit | S: snapshot | M: mirror | V: record | R: eye ROI"
BACKENDS = {
    "auto": cv2.CAP_ANY,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be a nonnegative integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=nonnegative_int, default=0, help="Camera index (default: 0)")
    source.add_argument("--simulate", action="store_true", help="Generate synthetic frames without a camera")
    source.add_argument("--video", type=Path, help="Read a local video file")
    source.add_argument("--list-cameras", action="store_true", help="Probe readable camera indices")
    parser.add_argument("--scan-count", type=positive_int, default=5, help="Probe indices 0 through N-1")
    parser.add_argument("--backend", choices=BACKENDS, default="auto")
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--fps", type=positive_int, default=30, help="Requested FPS; actual hardware rate may differ")
    parser.add_argument("--fourcc", choices=["YUY2", "MJPG"], help="Optional; negotiated by the driver by default")
    parser.add_argument("--mirror", action="store_true", help="Mirror horizontally")
    parser.add_argument("--pupil", action="store_true", help="Enable MEYELens pupil inference")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="auto")
    parser.add_argument("--cpu-threads", type=positive_int, default=1)
    parser.add_argument("--sync-capture", action="store_true", help="Disable live capture prefetch")
    parser.add_argument("--roi", nargs=4, type=nonnegative_int, metavar=("X", "Y", "W", "H"),
                        help="Single-eye region in the original unmirrored image; press R to reselect")
    parser.add_argument("--min-pupil-area", type=positive_int, default=20,
                        help="Minimum pupil area in pixels; a noise filter, not a confidence score")
    parser.add_argument("--headless", action="store_true", help="Run without a preview window")
    parser.add_argument("--frames", type=positive_int, help="Exit after reading this many frames")
    parser.add_argument("--snapshot", type=Path, help="Save the final frame to this PNG path")
    args = parser.parse_args(argv)
    if args.snapshot and args.snapshot.suffix.lower() != ".png":
        parser.error("--snapshot requires a .png extension")
    if args.roi and (args.roi[2] == 0 or args.roi[3] == 0):
        parser.error("ROI width and height must be positive")
    return args


def simulation_frame(index, width, height):
    """Test the image pipeline only; not a real eye or MEYELens output. Returns uint8 BGR."""
    frame = np.full((height, width, 3), 35, dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.ellipse(frame, center, (width // 3, height // 4), 0, 0, 360, (205, 205, 205), -1)
    pupil = (int(width * (0.5 + 0.10 * math.sin(index / 25))), height // 2)
    cv2.circle(frame, pupil, max(1, min(width, height) // 10), (85, 100, 70), -1)
    cv2.circle(frame, pupil, max(1, min(width, height) // 22), (8, 8, 8), -1)
    cv2.putText(frame, "SIMULATION - NOT A CAMERA", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return frame


def open_capture(args):
    if args.video:
        if not args.video.is_file():
            raise RuntimeError(f"Video not found: {args.video}")
        cap = cv2.VideoCapture(str(args.video))
    else:
        cap = cv2.VideoCapture(args.camera, BACKENDS[args.backend])
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open input. Check the index, USB/UVC connection, and camera permissions; "
                               "close other apps using the camera. Use --simulate without hardware.")
        if not args.video:
            settings = [(cv2.CAP_PROP_FRAME_WIDTH, args.width),
                        (cv2.CAP_PROP_FRAME_HEIGHT, args.height),
                        (cv2.CAP_PROP_FPS, args.fps)]
            if args.fourcc:
                settings.insert(0, (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc)))
            for prop, value in settings:
                if not cap.set(prop, value):
                    print(f"Note: the driver rejected {prop}={value}; actual frames take precedence.")
        print(f"Capture backend: {cap.getBackendName()}; device-reported FPS: {cap.get(cv2.CAP_PROP_FPS):g}")
        return cap
    except Exception:
        cap.release()
        raise


def scan_cameras(args):
    found = []
    for index in range(args.scan_count):
        cap = cv2.VideoCapture(index, BACKENDS[args.backend])
        try:
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if ok and frame is not None and frame.size:
                found.append(index)
                print(f"Index {index}: readable, {frame.shape[1]}x{frame.shape[0]}")
        finally:
            cap.release()
    if not found:
        print("No readable cameras found. Check USB/UVC and permissions, or increase --scan-count.")
    return 0 if found else 1


def save_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Cannot save snapshot: {path}")
    print(f"Saved: {path.resolve()}")


def run(args):
    cap = None
    worker = None
    pending = None
    recording = None
    recorded = 0
    count = 0
    last_frame = None
    start = time.monotonic()
    previous = start
    mirror = args.mirror
    roi = args.roi
    try:
        model = None
        if args.pupil:
            from pupil import load_model, annotate
            model = load_model(args.device, args.cpu_threads)
        if not args.simulate:
            cap = open_capture(args)
            if not args.video and not args.sync_capture:
                worker = ThreadPoolExecutor(max_workers=1)
                pending = worker.submit(cap.read)
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        print("Input: " + ("Simulation (not a real camera)" if args.simulate else
                         str(args.video) if args.video else f"Camera index {args.camera}"))
        while args.frames is None or count < args.frames:
            if args.simulate:
                frame = simulation_frame(count, args.width, args.height)
            else:
                ok, frame = pending.result() if pending is not None else cap.read()
                if not ok or frame is None or not frame.size:
                    if args.video and count:
                        print("Video reading finished.")
                        break
                    raise RuntimeError("Cannot read a frame: the device may be disconnected, busy, or missing permission; "
                                       "the video may also be empty or undecodable.")
            count += 1
            if worker is not None and (args.frames is None or count < args.frames):
                pending = worker.submit(cap.read)
            if count == 1:
                print(f"Actual frame: {frame.shape[1]}x{frame.shape[0]}; Q/Esc: quit, S: snapshot, M: mirror, V: record.")
                if not args.video and not args.simulate and frame.shape[:2] != (args.height, args.width):
                    print(f"Warning: requested {args.width}x{args.height}; the camera returned a different size.")
            if recording is not None:
                if frame.shape[:2] != recording_size:
                    raise RuntimeError("Frame dimensions changed during recording")
                recording.write(frame)
                recorded += 1
            annotated = frame
            if model is not None:
                annotated, center, label, color = annotate(frame, model, roi, args.min_pupil_area)
                if count == 1 or args.headless:
                    print(label)
            last_frame = cv2.flip(annotated, 1) if mirror else annotated.copy()
            if model is not None:
                cv2.putText(last_frame, label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            now = time.monotonic()
            measured_fps = 1 / max(now - previous, 1e-9) if count > 1 else 0
            previous = now
            if not args.headless:
                display = last_frame.copy()  # Snapshots and model input exclude the FPS overlay.
                cv2.putText(display, f"{display.shape[1]}x{display.shape[0]} | {measured_fps:.1f} FPS",
                            (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 1)
                cv2.putText(display, f"REC {recorded} frames" if recording is not None else "PREVIEW | V: record",
                            (10, display.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 255) if recording is not None else (0, 255, 0), 1)
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("m"):
                    mirror = not mirror
                if key == ord("v"):
                    if recording is not None:
                        recording.release()
                        recording = None
                        print(f"Recording stopped: {recorded} frames.")
                    else:
                        directory = ROOT / "captures"
                        directory.mkdir(parents=True, exist_ok=True)
                        prefix = "simulation" if args.simulate else "recording"
                        path = directory / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.avi"
                        rate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else args.fps
                        if not math.isfinite(rate) or rate <= 0:
                            rate = args.fps
                        recording_size = frame.shape[:2]
                        recording = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                                                   rate, (frame.shape[1], frame.shape[0]))
                        if not recording.isOpened():
                            raise RuntimeError(f"Cannot create recording: {path}")
                        recorded = 0
                        print(f"Recording started: {path}")
                if key == ord("r") and model is not None:
                    selection = cv2.selectROI("Select one eye (original image)", frame,
                                              showCrosshair=True, fromCenter=False)
                    cv2.destroyWindow("Select one eye (original image)")
                    if selection[2] and selection[3]:
                        roi = selection
                        print(f"ROI (original unmirrored image): {roi}")
                if key == ord("s"):
                    save_frame(ROOT / "captures" / f"{datetime.now():%Y%m%d_%H%M%S_%f}.png", last_frame)
            # Pace simulations and videos at the target rate; hardware controls live capture timing.
            if args.simulate or args.video:
                rate = cap.get(cv2.CAP_PROP_FPS) if args.video else args.fps
                if not math.isfinite(rate) or rate <= 0:
                    rate = args.fps
                time.sleep(max(0, 1 / rate - (time.monotonic() - now)))
        if args.snapshot and last_frame is not None:
            save_frame(args.snapshot, last_frame)
        print(f"Finished: {count} frames in {time.monotonic() - start:.2f} seconds.")
        return 0
    finally:
        if recording is not None:
            recording.release()
        if worker is not None:
            worker.shutdown(wait=True, cancel_futures=True)
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


def main(argv=None):
    args = parse_args(argv)
    try:
        return scan_cameras(args) if args.list_cameras else run(args)
    except KeyboardInterrupt:
        print("\nCapture stopped.")
        return 130
    except (RuntimeError, OSError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
