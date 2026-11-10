"""Offline single-camera/single-eye pupil measurements and annotated review video."""

import argparse
import csv
import math
from pathlib import Path
import sys

import cv2

from camera import nonnegative_int, positive_int
from pupil import load_model, measure

FIELDS = ["timestamp", "frame_id", "pupil_x", "pupil_y", "pupil_radius",
          "pupil_diameter", "confidence", "detected", "pupil_area", "timestamp_source"]


def analyze(video, output, model, roi=None, min_area=20, annotated=None):
    """One row per frame. Times are video-relative, never sensor capture timestamps."""
    video, output = Path(video), Path(output)
    annotated = Path(annotated) if annotated is not None else output.with_suffix(".annotated.avi")
    paths = [video.resolve(), output.resolve(), annotated.resolve()]
    if len(set(paths)) != 3:
        raise RuntimeError("Input video, CSV, and annotated video must have different paths")
    if not video.is_file():
        raise RuntimeError(f"Video not found: {video}")
    for path in (output, annotated):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}; choose a different output path")
    cap = cv2.VideoCapture(str(video))
    writer = None
    count = 0
    previous_time = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Video has no valid FPS; cannot establish frame timing")
        expected = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        output.parent.mkdir(parents=True, exist_ok=True)
        annotated.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", newline="", encoding="utf-8") as stream:
            rows = csv.DictWriter(stream, fieldnames=FIELDS)
            rows.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if math.isfinite(pts) and pts >= 0 and (previous_time is None or pts > previous_time):
                    timestamp, source = pts, "video_pts"
                else:
                    timestamp = 0.0 if previous_time is None else previous_time + 1 / fps
                    source = "fps_estimate"
                previous_time = timestamp
                data, result = measure(frame, model, roi, min_area)
                radius = math.sqrt(data["pupil_area_px2"] / math.pi) if data["detected"] else None
                row = {"timestamp": timestamp, "frame_id": count,
                       "pupil_x": data["pupil_x_px"], "pupil_y": data["pupil_y_px"],
                       "pupil_radius": radius, "pupil_diameter": 2 * radius if radius is not None else None,
                       "confidence": None, "detected": data["detected"],
                       "pupil_area": data["pupil_area_px2"], "timestamp_source": source}
                # Segmentation probabilities are per-pixel, not calibrated detection confidence.
                display = frame.copy()
                x, y, w, h = (data[key] for key in ("roi_x", "roi_y", "roi_w", "roi_h"))
                color = (0, 255, 0) if data["detected"] else (0, 180, 255)
                label = "NOT DETECTED"
                if data["detected"]:
                    mask = result.masks["pupil"]
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display[y:y+h, x:x+w], contours, -1, color, 1)
                    center = (round(row["pupil_x"]), round(row["pupil_y"]))
                    cv2.drawMarker(display, center, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
                    label = f"x={row['pupil_x']:.1f} y={row['pupil_y']:.1f} eq. diameter={2*radius:.1f}px"
                if roi is not None:
                    cv2.rectangle(display, (x, y), (x+w-1, y+h-1), (255, 180, 0), 1)
                cv2.putText(display, f"MODEL PREDICTION | frame {count} | {timestamp:.3f}s", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.putText(display, label, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                size = (frame.shape[1], frame.shape[0])
                if writer is None:
                    video_size = size
                    writer = cv2.VideoWriter(str(annotated), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create annotated video: {annotated}")
                if size != video_size:
                    raise RuntimeError("Video frame dimensions changed during analysis")
                writer.write(display)
                rows.writerow(row)
                count += 1
                if count % 100 == 0:
                    print(f"Analyzed {count} frames...", flush=True)
            if count == 0:
                raise RuntimeError("Video contains no decodable frames")
            if math.isfinite(expected) and expected > 0 and count < round(expected):
                raise RuntimeError(f"Decoded only {count} of {round(expected)} reported frames")
        print(f"Exported {count} frames\nCSV: {output.resolve()}\nReview video: {annotated.resolve()}")
        return count
    finally:
        if writer is not None:
            writer.release()
        cap.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Recorded video containing one eye, or select an ROI")
    parser.add_argument("--output", type=Path, help="CSV path; default: VIDEO.pupil.csv; must not exist")
    parser.add_argument("--roi", nargs=4, type=nonnegative_int, metavar=("X", "Y", "W", "H"),
                        help="Fixed single-eye region in original video pixels")
    parser.add_argument("--select-roi", action="store_true", help="Select one eye on the first frame")
    parser.add_argument("--min-pupil-area", type=positive_int, default=20)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="auto")
    parser.add_argument("--cpu-threads", type=positive_int, default=1)
    args = parser.parse_args(argv)
    if args.roi and args.select_roi:
        parser.error("Use either --roi or --select-roi")
    if args.roi and (args.roi[2] == 0 or args.roi[3] == 0):
        parser.error("ROI width and height must be positive")
    try:
        if not args.video.is_file():
            raise RuntimeError(f"Video not found: {args.video}")
        roi = args.roi
        if args.select_roi:
            cap = cv2.VideoCapture(str(args.video))
            try:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Cannot read first frame for ROI selection")
                roi = cv2.selectROI("Select one eye; Enter to confirm, C to cancel", frame,
                                    showCrosshair=True, fromCenter=False)
            finally:
                cap.release()
                cv2.destroyAllWindows()
            if not roi[2] or not roi[3]:
                print("ROI selection canceled; no files exported.")
                return 0
            print(f"Selected ROI: {roi}")
        model = load_model(args.device, args.cpu_threads)
        analyze(args.video, args.output or args.video.with_suffix(".pupil.csv"), model,
                roi, args.min_pupil_area)
        return 0
    except KeyboardInterrupt:
        print("Analysis stopped. Any output files are incomplete.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}. Any files created by this run are incomplete.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
