"""MEYELens inference and original-image coordinate annotation for a single-eye ROI."""

from pathlib import Path
import math
import shutil
import tempfile

import cv2
import numpy as np


def load_model(device="auto", cpu_threads=1):
    try:
        from meyelens import Meye
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the MEYELens inference dependencies from requirements.txt first.") from exc
    try:
        torch.set_num_threads(cpu_threads)
        # MEYELens rewrites its config with local paths; keep the shared file portable.
        with tempfile.TemporaryDirectory(prefix="hosa-meye-") as directory:
            config = Path(directory) / "meye_config.toml"
            shutil.copyfile(Path(__file__).with_name("meye_config.toml"), config)
            model = Meye(config_path=config, gpu_device=device, verbose=False)
        model.compute_features = True
        model.keep_biggest = True
        print(f"MEYELens loaded: {model.model_path}; device: {model.device}")
        return model
    except Exception as exc:
        raise RuntimeError(f"MEYELens model loading failed: {exc}") from exc


def measure(frame, model, roi=None, min_area=20):
    """Return structured measurements and the model result, in original-image pixels."""
    height, width = frame.shape[:2]
    x, y, w, h = roi if roi is not None else (0, 0, width, height)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise RuntimeError(f"ROI {x,y,w,h} exceeds the actual frame bounds {width}x{height}")
    try:
        result = model.predict(frame[y:y+h, x:x+w])
    except Exception as exc:
        raise RuntimeError(f"MEYELens inference failed: {exc}") from exc
    features = result.features.get("pupil", {})
    center = None
    area = None
    if features.get("valid") and math.isfinite(features.get("area", 0)) and features.get("area", 0) >= min_area:
        row, col = features["centroid"]
        if np.isfinite([row, col]).all() and 0 <= col < w and 0 <= row < h:
            center = (float(col + x), float(row + y))
            area = float(features["area"])
    data = {"pupil_x_px": center[0] if center else None,
            "pupil_y_px": center[1] if center else None,
            "pupil_area_px2": area, "detected": int(center is not None),
            "roi_x": x, "roi_y": y, "roi_w": w, "roi_h": h}
    return data, result


def annotate(frame, model, roi=None, min_area=20):
    """ROI is x,y,w,h in the original unmirrored frame; return annotations and center (x,y) or None."""
    data, result = measure(frame, model, roi, min_area)
    x, y, w, h = (data[key] for key in ("roi_x", "roi_y", "roi_w", "roi_h"))
    display = frame.copy()
    center = None
    text = "Pupil: NOT DETECTED"
    color = (0, 180, 255)
    if data["detected"]:
        center = (data["pupil_x_px"], data["pupil_y_px"])
        mask = result.masks["pupil"]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display[y:y+h, x:x+w], contours, -1, (0, 255, 0), 1)
        cv2.drawMarker(display, tuple(round(v) for v in center), (0, 0, 255),
                       cv2.MARKER_CROSS, 16, 2)
        text = f"Pupil x={center[0]:.1f} y={center[1]:.1f} area={data['pupil_area_px2']:g}px"
        color = (0, 255, 0)
    if roi is not None:
        cv2.rectangle(display, (x, y), (x+w-1, y+h-1), (255, 180, 0), 1)
    # Mirror outside this function; draw text afterward to keep it readable.
    return display, center, text, color
