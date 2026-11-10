# HOSA Eye Tracking

Camera capture, single-eye pupil segmentation, and offline video analysis for the HOSA project. This program measures pupil position and size in image pixels; it does not estimate calibrated screen gaze.
Python and OpenCV handle capture, with the pretrained model bundled in MEYELens 2.0.2 available through `--pupil` for live pupil segmentation.

## Environment and startup

Install Python 3.11, then clone the repository and install its dependencies:

```bash
git clone https://github.com/S1an0028/hosa-eye-tracking.git
cd hosa-eye-tracking
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python camera.py --simulate
```

On Windows, use `py -3.11 -m venv .venv` to create the environment and
`.venv\Scripts\Activate.ps1` to activate it in PowerShell.
The first installation includes PyTorch and can take several minutes.
The pretrained weights are supplied by the installed MEYELens package.

The preview shows a synthetic eye labeled `SIMULATION - NOT A CAMERA`.
This tests the image pipeline, not detection accuracy.
Focus the preview window to use these keys:

| Key | Action |
| --- | --- |
| `Q` / `Esc` | Quit |
| `S` | Save a PNG snapshot in `captures/` |
| `M` | Toggle horizontal mirroring |
| `V` | Start or stop recording original frames |
| `R` | Select one eye when `--pupil` is enabled |

You can also close the window or press Ctrl+C in the terminal.
Recordings, snapshots, analysis CSVs, and the local environment are excluded from Git.

## Connecting the GC0308

**Use a complete GC0308 module with a USB interface and UVC support.** GC0308 identifies the sensor, not the connection interface. Bare sensors and DVP/FPC modules require a suitable capture or bridge board and its drivers or SDK. This application reads through the operating system's video capture interface; it does not implement a DVP bus or sensor register driver.

1. Connect the USB/UVC module to the computer.
2. Allow camera access when prompted. On macOS, enable the terminal or application that launches the program under System Settings → Privacy & Security → Camera. Restart that application if necessary after changing permissions.
3. Close other applications using the camera, probe readable indices, and start capture:

```bash
python camera.py --list-cameras
python camera.py --camera 0
# If index 0 is the built-in camera, use another index from the probe results.
python camera.py --camera 1 --width 640 --height 480 --fps 30
```

Probing opens and reads each device and may trigger permission prompts. An index does not identify the hardware model and can change after reconnecting devices. Check the preview to confirm that you selected the GC0308.
Connection or capture failures produce an error; the program does not automatically switch to simulation.

The driver negotiates the pixel format by default. If the module specifies YUY2 only, try:

```bash
python camera.py --camera 1 --fourcc YUY2
```

Drivers may reject some parameter changes. The program reports rejected settings, the actual first-frame dimensions, and device-reported FPS. Preview FPS measures application throughput, not the sensor's hardware frame rate.
If needed, select `--backend avfoundation` on macOS, `dshow` on Windows, or `v4l2` on Linux. USB bridge compatibility still requires testing with the actual camera.

## Verification without a camera

```bash
# Read 60 frames without a window and save the final frame.
python camera.py --simulate --headless --frames 60 --snapshot captures/simulation.png

# Check decoding and display with an existing local video.
python camera.py --video '/absolute/path/eye.mp4'

# Run automated checks without accessing real cameras.
python -m unittest -v
```

`--headless` works with any input. Add `--frames N` to limit the frame count; otherwise videos run to the end and live inputs continue until interrupted.
`--snapshot` saves the final frame, mirrored if enabled, without the FPS overlay. Simulation labels remain visible.
OpenCV generally cannot distinguish a normal video ending from a decoding failure partway through a file. Once at least one frame has been read, subsequent video read failures are treated as the end of the video.

## Live pupil detection with MEYELens

After installing `requirements.txt`, the inference dependencies and bundled pretrained weights are available locally. No separate model download is required.

```bash
python camera.py --camera 0 --pupil
# Select a single-eye ROI: original-image x y width height. Adjust to your image.
python camera.py --camera 0 --pupil --roi 100 100 240 240
# Run the real model without hardware to check inference, not accuracy.
python camera.py --simulate --pupil
# Run inference on an existing real eye video.
python camera.py --video '/absolute/path/eye.mp4' --pupil
```

The preview displays a green pupil contour, a red center cross, original-image center coordinates, and segmented area. If the detection does not pass the minimum area filter, it displays `Pupil: NOT DETECTED` and does not reuse a previous position.
Press `R` to select a single-eye region on the original unmirrored image, then Enter/Space to confirm. Canceling keeps the previous ROI. The selection is printed in the terminal for reuse with `--roi`.
The whole frame is processed by default, suitable for a close-up of one eye. Select a single-eye ROI when multiple eyes are visible.
`--mirror` flips the displayed image and markers; reported coordinates still refer to the original unmirrored image. With inference enabled, snapshots include the contour, center, and detection status, but exclude the FPS overlay.

The default is `--device auto`, which selects Apple MPS when available, otherwise CUDA or CPU as supported by MEYELens. Use `--device cpu` to force CPU inference; `--cpu-threads` defaults to 1; benchmark other values on your computer if needed.

Live capture prefetches one frame in a worker while inference processes the preceding frame. Each pupil marker is drawn on the exact frame used for that prediction. There is no unbounded frame queue or reused prediction. Local videos retain sequential reading. Use `--sync-capture` to disable prefetch for comparison.

The overlay reports the rate between processed frames. The exit summary reports processed frames and total elapsed time, including model initialization. These timings do not verify distinct sensor exposures or detection accuracy.
`--min-pupil-area 20` filters small noise regions; it is not an accuracy or confidence guarantee. Closed eyes, reflections, and poor focus can still produce incorrect detections. Positions are model predictions and require review against real video.
`meye_config.toml` stores the thresholds and portable bundled-model filename. MEYELens resolves the weights from its installed package. A temporary config copy prevents it from writing machine-specific paths into the shared file.

Automated checks cover simulation, video decoding, manual recording, capture/inference overlap, failure cleanup, ROI coordinate mapping, mirrored markers, and offline CSV/video exports using constructed predictions. Detection accuracy on real eye footage has not yet been validated.

## Manual video recording

Recording is **off on every startup**. Focus the preview window and press `V` to
start recording. Press `V` again to stop; pressing it again creates a new session.
`Q`, Escape, closing the window, or Ctrl+C closes an active recording. The preview
shows `PREVIEW` or `REC` and the recorded frame count. Headless mode does not record.

```bash
python camera.py --camera 1
# Optional live pupil preview; recording still starts only when you press V.
python camera.py --camera 1 --pupil
```

Each session saves one `recording_TIMESTAMP.avi` in `captures/`. It contains
original, unmirrored frames without pupil annotations or preview overlays. MJPEG
is lossy compression. Simulation recordings use a `simulation_` prefix and retain
their synthetic label. No CSV or timestamp sidecar is created. Recording and
later data analysis are separate tasks; `pupil.py` only provides live inference
and annotation and does not export files.

AVI playback uses nominal constant FPS; actual per-frame capture times are not
stored. An immediately stopped session can contain zero frames and should be
discarded.

## Separate offline analysis: analyze_video.py

Recording remains in `camera.py`; `pupil.py` does not export files. Run the
independent `analyze_video.py` after recording. It processes one video from one
camera and tracks one eye per frame. These are automatic model annotations for
review, not manually verified ground-truth labels.

```bash
# Close-up video of one eye:
python analyze_video.py '/absolute/path/recording.avi'
# If both eyes or background are visible, select one fixed eye region first:
python analyze_video.py '/absolute/path/recording.avi' --select-roi
# Or supply a known region in original, unmirrored image pixels:
python analyze_video.py '/absolute/path/recording.avi' --roi 100 100 240 240 --output captures/trial01.csv
```

ROI selection uses the first frame: draw a rectangle, then press Enter/Space.
Canceling creates no outputs. The region stays fixed throughout the video; keep
the eye inside it. Analysis runs without a preview unless selecting a region.

Default outputs beside the input video:

- `recording.pupil.csv`: one row per decoded frame, including missed detections.
- `recording.pupil.annotated.avi`: review video with pupil contour, center cross,
  frame number, video time, detection status, and equivalent diameter.

`--output` changes the CSV path; the review video uses that path's base name plus
`.annotated.avi`. Existing files are never overwritten. Failed or interrupted
analysis can leave incomplete files; choose a new output name for a rerun.

| CSV field | Meaning |
| --- | --- |
| `timestamp` | Video presentation time in seconds; estimated using FPS if unavailable/non-increasing |
| `frame_id` | Zero-based frame number |
| `pupil_x`, `pupil_y` | Original full-frame coordinates, pixels; origin at top left, x right, y down |
| `pupil_radius` | Equivalent-circle radius `sqrt(pupil_area / pi)`, pixels |
| `pupil_diameter` | Twice the equivalent radius, pixels; not an ellipse axis or physical millimeters |
| `confidence` | Blank: this model exposes pixel probabilities, not a calibrated detection-level score |
| `detected` | 1 if the prediction passes validity, minimum area, and coordinate checks; otherwise 0 |
| `pupil_area` | Segmented pupil area in pixels squared |
| `timestamp_source` | `video_pts` or `fps_estimate` |

Missing detections leave coordinates, size, and area blank; no last-known values
are reused. `detected=1` does not establish that a prediction is correct. Review
the annotated video against the original footage before using the measurements.
The size is based on the visible segmented pupil; occlusion can reduce it.

The recorder stores constant-FPS video without acquisition timestamps. Therefore,
even `video_pts` represents the saved video timeline, not the true sensor exposure
time or wall-clock acquisition time. The review AVI also plays at nominal FPS.
No multi-camera synchronization or gaze calibration is performed. Decoder read
failures can mean EOF or corruption; a short decode is reported when the file's
reported frame count allows it to be detected.

## References

- [MEYELens installation and models](https://meyelens.com/software.html)
- [MEYELens Python API](https://meyelens.com/docs/)
- [OpenCV video capture tutorial](https://docs.opencv.org/4.13.0/dd/d43/tutorial_py_video_display.html)
- [Supertek GC0308 USB/UVC specifications: 640×480, VGA 30 FPS](https://www.supertekmodule.com/wp-content/uploads/2018/11/ST-GC0308-Datasheet-V2.1.pdf)
- [Kailap GC0308 DVP module example](https://www.kailaptech.com/product.aspx?id=2890&l1=799)

## Validation limits

Simulation and local videos exercise the software image pipeline. Actual GC0308 recognition, USB bandwidth, exposure, focus, and hardware frame rate require testing after the camera arrives.

## Targeting 30 FPS

```bash
# Replace 1 with the actual external camera index.
python camera.py --camera 1 --width 640 --height 480 --fps 30 --pupil
# Run without window overhead (prints frame count and total elapsed time).
python camera.py --camera 1 --width 640 --height 480 --fps 30 --pupil --headless --frames 150
```

Select one eye with `R` when needed. A warning is printed if the actual frame dimensions differ from the requested dimensions. Requested FPS remains a driver request, not a guarantee. A camera delivering 13 distinct frames per second cannot provide 30 distinct tracked frames per second through inference optimization alone.

Actual throughput depends on the camera, exposure, computer, inference device, and preview overhead. Benchmark your own setup; 30 FPS is a request, not a guarantee.
