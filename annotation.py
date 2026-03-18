#!/usr/bin/env python3
"""
annotation.py — Video annotation runner (frame-level + optional annotated video)

Key features:
- Frame-level probabilities via a Keras model (frame model or sequence model).
- Smoothing for short clips.
- Label post-processing: majority-vote + fill short islands (glitch fixing).
- Min-prob threshold FIXED correctly:
    After ALL label post-processing:
      pred_prob[i] = probs[i, preds[i]] (probability of the FINAL label)
      if pred_prob[i] < --min-prob -> route to --lowconf-label (default "No action")
    In overlay timing/progress:
      only add time to a step if that frame's pred_prob >= --min-prob
      otherwise add to lowconf label.
- Annotated MP4 overlay:
    - show all steps list + checkbox indicator
    - per-step progress bars (default 3s target per step)
    - end summary screen: per-step times + completion %
- --continue-on-error to skip failing videos
- Better MP4 writing on macOS: try 'avc1' then 'mp4v'

Outputs (per video):
- <stem>_frame_predictions.csv
- <stem>_segments.csv
- <stem>_annotated.mp4 (optional)

Example:
  python annotation.py --video-dir events --glob "*.mp4" --model model.keras --out out --write-video --min-prob 0.6
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import tensorflow as tf  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "TensorFlow is required. Ensure you're running inside your environment "
        "where TensorFlow is installed.\n"
        f"Original import error: {exc}"
    )


# ----------------------------
# Config discovery (optional)
# ----------------------------
def find_config(start: Path) -> Tuple[Optional[Path], Optional[str]]:
    start = start.resolve()
    for p in [start, *start.parents]:
        root_cfg = p / "config.py"
        inf_cfg = p / "inference" / "config.py"
        if root_cfg.exists():
            return root_cfg, "root"
        if inf_cfg.exists():
            return inf_cfg, "inference"
    return None, None


def load_cfg(explicit_config: Optional[Path]) -> Optional[object]:
    cfg_path, style = (explicit_config, "explicit") if explicit_config else find_config(Path.cwd())
    if not cfg_path or not cfg_path.exists():
        return None

    if style == "explicit":
        if cfg_path.name == "config.py" and cfg_path.parent.name == "inference":
            style = "inference"
        else:
            style = "root"

    import importlib

    if style == "root":
        sys.path.insert(0, str(cfg_path.parent))
        return importlib.import_module("config")
    if style == "inference":
        sys.path.insert(0, str(cfg_path.parent.parent))
        return importlib.import_module("inference.config")

    return None


# ----------------------------
# Video + preprocessing
# ----------------------------
def preprocess_frame_bgr(frame_bgr: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for video decoding/encoding.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    return x


def collect_video_frames(
    video_path: Path,
    img_size: Tuple[int, int],
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
) -> Tuple[float, np.ndarray, List[np.ndarray], np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for video decoding/encoding.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    idx = 0
    kept = 0
    inputs: List[np.ndarray] = []
    frames_bgr: List[np.ndarray] = []
    timestamps: List[float] = []

    stride = max(1, int(frame_stride))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames_bgr.append(frame)
            inputs.append(preprocess_frame_bgr(frame, img_size))
            timestamps.append(idx / float(fps))
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        idx += 1

    cap.release()
    if inputs:
        inputs_arr = np.stack(inputs, axis=0)
    else:
        inputs_arr = np.zeros((0, img_size[0], img_size[1], 3), np.float32)

    return float(fps), inputs_arr, frames_bgr, np.asarray(timestamps, dtype=np.float32)


# ----------------------------
# Model inference helpers
# ----------------------------
def is_sequence_model(model: "tf.keras.Model") -> bool:
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        return len(shape) == 5
    except Exception:
        return False


def make_sequences(frames: np.ndarray, seq_len: int, seq_stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    n = frames.shape[0]
    if n < seq_len:
        pad = (
            np.repeat(frames[-1:, ...], repeats=(seq_len - n), axis=0)
            if n > 0
            else np.zeros((seq_len, *frames.shape[1:]), np.float32)
        )
        frames = np.concatenate([frames, pad], axis=0)
        n = frames.shape[0]

    starts = list(range(0, n - seq_len + 1, max(1, int(seq_stride))))
    seqs = np.stack([frames[s : s + seq_len] for s in starts], axis=0)
    centers = np.asarray([s + (seq_len // 2) for s in starts], dtype=np.int32)
    return seqs, centers


def load_model_robust(model_path: Path) -> "tf.keras.Model":
    custom_objects: Dict[str, object] = {}
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess  # type: ignore
        custom_objects["preprocess_input"] = mobilenet_v2_preprocess
    except Exception:
        pass

    return tf.keras.models.load_model(
        str(model_path),
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )


def predict_probs(
    model: "tf.keras.Model",
    inputs_arr: np.ndarray,
    sequence_length: int,
    sequence_stride: int,
    batch_size: int,
) -> np.ndarray:
    if inputs_arr.shape[0] == 0:
        return np.zeros((0, 1), np.float32)

    if is_sequence_model(model):
        seqs, centers = make_sequences(inputs_arr, int(sequence_length), int(sequence_stride))
        probs_seq = np.asarray(model.predict(seqs, batch_size=int(batch_size), verbose=0))

        if probs_seq.ndim == 3:
            probs_seq = probs_seq[:, probs_seq.shape[1] // 2, :]

        n = inputs_arr.shape[0]
        c = probs_seq.shape[-1]
        probs = np.zeros((n, c), np.float32)
        filled = np.zeros((n,), np.bool_)

        for i, center in enumerate(centers):
            if 0 <= center < n:
                probs[center] = probs_seq[i]
                filled[center] = True

        last = None
        for i in range(n):
            if filled[i]:
                last = probs[i].copy()
            elif last is not None:
                probs[i] = last

        last = None
        for i in reversed(range(n)):
            if filled[i]:
                last = probs[i].copy()
            elif last is not None and probs[i].sum() == 0:
                probs[i] = last

        return probs

    probs = np.asarray(model.predict(inputs_arr, batch_size=int(batch_size), verbose=0))
    if probs.ndim == 1:
        probs = probs[:, None]
    return probs


# ----------------------------
# Smoothing + glitch fixing
# ----------------------------
def moving_average_probs(probs: np.ndarray, window: int) -> np.ndarray:
    if probs is None or probs.shape[0] == 0:
        return probs

    w = int(window)
    if w <= 1:
        return probs

    n = int(probs.shape[0])
    if w > n:
        w = n

    kernel = np.ones((w,), dtype=np.float32) / float(w)
    out = np.empty_like(probs, dtype=np.float32)
    for c in range(probs.shape[1]):
        out[:, c] = np.convolve(probs[:, c], kernel, mode="same")

    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def majority_vote(preds: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or preds.size == 0:
        return preds
    w = int(window)
    half = w // 2
    out = preds.copy()
    for i in range(preds.size):
        a = max(0, i - half)
        b = min(preds.size, i + half + 1)
        vals, counts = np.unique(preds[a:b], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def _runs(preds: np.ndarray) -> List[Tuple[int, int, int]]:
    if preds.size == 0:
        return []
    runs: List[Tuple[int, int, int]] = []
    cur = int(preds[0])
    s = 0
    for i in range(1, preds.size):
        if int(preds[i]) != cur:
            runs.append((cur, s, i - 1))
            cur = int(preds[i])
            s = i
    runs.append((cur, s, preds.size - 1))
    return runs


def fill_short_islands(preds: np.ndarray, timestamps: np.ndarray, max_island_s: float = 0.5) -> np.ndarray:
    if preds.size == 0 or timestamps.size != preds.size:
        return preds

    out = preds.copy()
    runs = _runs(out)

    def dur(a: int, b: int) -> float:
        return float(timestamps[b] - timestamps[a]) if b > a else 0.0

    for i in range(1, len(runs) - 1):
        lbl, a, b = runs[i]
        left_lbl, _, _ = runs[i - 1]
        right_lbl, _, _ = runs[i + 1]
        if left_lbl == right_lbl and lbl != left_lbl and dur(a, b) <= float(max_island_s):
            out[a : b + 1] = left_lbl

    return out


# ----------------------------
# Label utilities + min-prob routing
# ----------------------------
def _label_index(labels: Sequence[str], name: str) -> Optional[int]:
    want = str(name).strip().lower()
    for i, lbl in enumerate(labels):
        if str(lbl).strip().lower() == want:
            return i
    return None


def ensure_label(labels: List[str], name: str) -> int:
    idx = _label_index(labels, name)
    if idx is not None:
        return idx
    labels.append(name)
    return len(labels) - 1


def compute_pred_prob_of_final_label(probs: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """
    pred_prob[i] = probs[i, preds[i]] (probability of the FINAL label).
    If preds[i] is out of range (e.g., virtual label), pred_prob is 0.
    """
    if probs is None or probs.size == 0 or preds.size == 0:
        return np.zeros((preds.size,), np.float32)

    n = min(probs.shape[0], preds.shape[0])
    pred_prob = np.zeros((preds.shape[0],), np.float32)

    valid = (preds[:n] >= 0) & (preds[:n] < probs.shape[1])
    idx = np.arange(n, dtype=np.int32)
    pred_prob[:n][valid] = probs[idx[valid], preds[:n][valid].astype(np.int32)].astype(np.float32)
    # invalid stays 0
    return pred_prob


def route_low_confidence_frames(
    probs: np.ndarray,
    preds: np.ndarray,
    labels_for_use: List[str],
    min_prob: float,
    lowconf_label: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    IMPORTANT: Call this AFTER majority-vote and island-fill, so preds is final post-processed label.
      pred_prob[i] = probs[i, preds[i]]
      if pred_prob[i] < min_prob -> preds[i] = lowconf_idx
    Returns:
      preds_routed, pred_prob_final_label, lowconf_idx
    """
    thr = float(min_prob)
    low_idx = ensure_label(labels_for_use, lowconf_label)

    pred_prob = compute_pred_prob_of_final_label(probs, preds)

    if thr <= 0.0:
        return preds, pred_prob, low_idx

    out = preds.copy()
    # Any frame whose FINAL label prob is below threshold becomes lowconf
    out[pred_prob < thr] = int(low_idx)

    # Recompute pred_prob for routed preds (virtual lowconf has prob 0; that's OK for display/gating)
    pred_prob2 = compute_pred_prob_of_final_label(probs, out)
    return out, pred_prob2, low_idx


# ----------------------------
# Segmentation
# ----------------------------
@dataclass
class Segment:
    label: str
    start_s: float
    end_s: float
    mean_pred_prob: float


def build_segments(labels: Sequence[str], preds: np.ndarray, pred_prob: np.ndarray, timestamps: np.ndarray, min_duration_s: float) -> List[Segment]:
    if preds.size == 0:
        return []
    segs: List[Segment] = []
    runs = _runs(preds)

    for lbl, a, b in runs:
        start_s = float(timestamps[a])
        end_s = float(timestamps[b]) if b < timestamps.size else float(timestamps[-1])
        mean_pp = float(np.mean(pred_prob[a : b + 1])) if pred_prob.size else 0.0
        name = labels[lbl] if 0 <= lbl < len(labels) else str(lbl)
        if (end_s - start_s) >= float(min_duration_s):
            segs.append(Segment(label=name, start_s=start_s, end_s=end_s, mean_pred_prob=mean_pp))
    return segs


# ----------------------------
# CSV writers
# ----------------------------
def write_frame_csv(out_csv: Path, labels: Sequence[str], timestamps: np.ndarray, probs: np.ndarray, preds: np.ndarray, pred_prob: np.ndarray) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp_s", "pred_label", "pred_prob"] + [f"prob_{lbl}" for lbl in labels]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, t in enumerate(timestamps.tolist()):
            pi = int(preds[i]) if i < preds.size else 0
            pl = labels[pi] if 0 <= pi < len(labels) else str(pi)
            pp = float(pred_prob[i]) if i < pred_prob.size else 0.0

            row = [f"{t:.6f}", pl, f"{pp:.6f}"]

            if probs.size:
                row += [f"{float(p):.6f}" for p in probs[i].tolist()]
                # If we appended virtual labels (e.g., "No action"), pad missing prob columns with 0
                extra = len(labels) - probs.shape[1]
                if extra > 0:
                    row += ["0.000000"] * extra

            w.writerow(row)


def write_segments_csv(out_csv: Path, segments: Sequence[Segment]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "start_s", "end_s", "duration_s", "mean_pred_prob"])
        for s in segments:
            w.writerow([s.label, f"{s.start_s:.6f}", f"{s.end_s:.6f}", f"{(s.end_s - s.start_s):.6f}", f"{s.mean_pred_prob:.6f}"])


# ----------------------------
# Video overlay + summary
# ----------------------------
def _safe_video_writer(out_path: Path, fps: float, size_wh: Tuple[int, int]):
    if cv2 is None:
        raise RuntimeError("cv2 required for video writing.")

    w, h = size_wh
    fps = float(fps) if fps and fps > 1e-6 else 30.0

    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if vw is not None and vw.isOpened():
            return vw, fourcc_str

    raise RuntimeError("Could not open VideoWriter for mp4 (tried avc1, mp4v).")


def _is_other_label(name: str) -> bool:
    n = name.strip().lower()
    return n in {"other", "others", "unknown", "background", "none"}


def _is_lowconf_label(name: str) -> bool:
    n = name.strip().lower()
    return n in {"no action", "noaction", "nothing", "idle", "lowconf", "low confidence"}


def _draw_step_ui(
    frame_bgr: np.ndarray,
    labels: Sequence[str],
    current_label: str,
    step_times_so_far: Dict[str, float],
    target_s: float,
    show_all_steps: bool,
    exclude_other_from_completion: bool,
    exclude_lowconf_from_completion: bool,
    font_scale: float = 0.6,
) -> np.ndarray:
    if cv2 is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    pad = 12
    panel_w = int(min(520, w * 0.48))
    panel_h = int(min(h - 2 * pad, max(220, int(h * 0.60))))
    x0 = pad
    y0 = pad

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0)

    # checkbox (visual only)
    cb_size = 18
    cb_x = x0 + 12
    cb_y = y0 + 14
    cv2.rectangle(frame, (cb_x, cb_y), (cb_x + cb_size, cb_y + cb_size), (255, 255, 255), 2)
    if show_all_steps:
        cv2.line(frame, (cb_x + 3, cb_y + cb_size // 2), (cb_x + cb_size // 2, cb_y + cb_size - 4), (255, 255, 255), 2)
        cv2.line(frame, (cb_x + cb_size // 2, cb_y + cb_size - 4), (cb_x + cb_size - 3, cb_y + 4), (255, 255, 255), 2)
    cv2.putText(frame, "Show all steps", (cb_x + cb_size + 10, cb_y + cb_size - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Current: {current_label}", (x0 + 12, cb_y + cb_size + 30),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.15, (255, 255, 255), 2, cv2.LINE_AA)

    # Determine which labels to show
    if show_all_steps:
        show_labels = list(labels)
    else:
        show_labels = [lbl for lbl in labels if not _is_other_label(lbl) and not _is_lowconf_label(lbl)]

    # Completion calc
    eligible = []
    for lbl in labels:
        if exclude_other_from_completion and _is_other_label(lbl):
            continue
        if exclude_lowconf_from_completion and _is_lowconf_label(lbl):
            continue
        eligible.append(lbl)

    done = sum(1 for lbl in eligible if step_times_so_far.get(lbl, 0.0) >= float(target_s))
    completion = (done / max(1, len(eligible))) * 100.0

    cv2.putText(frame, f"Completion: {completion:4.1f}%", (x0 + 12, y0 + panel_h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    list_y = cb_y + cb_size + 60
    bar_x = x0 + 12
    bar_w = panel_w - 24
    bar_h = 14
    row_h = 32

    max_rows = max(1, int((panel_h - (list_y - y0) - 44) / row_h))
    show_labels = show_labels[:max_rows]

    for i, lbl in enumerate(show_labels):
        y = list_y + i * row_h

        is_cur = (lbl == current_label)
        color = (255, 255, 255) if is_cur else (220, 220, 220)
        cv2.putText(frame, lbl, (bar_x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

        bar_y = y + 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

        t = float(step_times_so_far.get(lbl, 0.0))
        frac = min(1.0, t / max(0.001, float(target_s)))
        fill_w = int(bar_w * frac)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (255, 255, 255), -1)

        cv2.putText(frame, f"{t:4.1f}s", (bar_x + bar_w - 70, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

    return frame


def _make_summary_frames(
    base_frame_bgr: np.ndarray,
    labels: Sequence[str],
    total_times: Dict[str, float],
    target_s: float,
    exclude_other_from_completion: bool,
    exclude_lowconf_from_completion: bool,
    fps: float,
    seconds: float,
) -> List[np.ndarray]:
    if cv2 is None:
        return []
    h, w = base_frame_bgr.shape[:2]
    n_frames = int(max(1, round(float(seconds) * float(fps))))

    eligible = []
    for lbl in labels:
        if exclude_other_from_completion and _is_other_label(lbl):
            continue
        if exclude_lowconf_from_completion and _is_lowconf_label(lbl):
            continue
        eligible.append(lbl)

    done = sum(1 for lbl in eligible if float(total_times.get(lbl, 0.0)) >= float(target_s))
    completion = (done / max(1, len(eligible))) * 100.0

    frames: List[np.ndarray] = []
    for _ in range(n_frames):
        frame = base_frame_bgr.copy()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)

        cv2.putText(frame, "Summary", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Completion: {completion:4.1f}%  (>= {target_s:.1f}s each)",
                    (24, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        y = 130
        for lbl in labels:
            t = float(total_times.get(lbl, 0.0))
            mark = ""
            if (exclude_other_from_completion and _is_other_label(lbl)) or (exclude_lowconf_from_completion and _is_lowconf_label(lbl)):
                mark = " (excluded)"
            elif t >= float(target_s):
                mark = " ✓"
            cv2.putText(frame, f"{lbl:20s}  {t:5.1f}s{mark}", (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            y += 30
            if y > h - 24:
                break

        frames.append(frame)

    return frames


def write_annotated_video(
    out_mp4: Path,
    frames_bgr: List[np.ndarray],
    fps_out: float,
    labels: Sequence[str],
    preds: np.ndarray,
    pred_prob: np.ndarray,
    timestamps: np.ndarray,
    min_prob: float,
    lowconf_label: str,
    step_target_s: float,
    show_all_steps: bool,
    exclude_other_from_completion: bool,
    exclude_lowconf_from_completion: bool,
    summary_seconds: float,
) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for video encoding.")
    if not frames_bgr:
        return

    h, w = frames_bgr[0].shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    fps_out = float(fps_out) if fps_out and fps_out > 1e-6 else 30.0
    vw, codec = _safe_video_writer(out_mp4, fps_out, (w, h))

    # accumulate times
    step_times = {lbl: 0.0 for lbl in labels}
    lowconf_idx = _label_index(labels, lowconf_label)
    if lowconf_idx is None:
        # should not happen (we ensure it earlier), but keep safe
        lowconf_idx = 0

    dts = np.diff(timestamps) if timestamps.size == preds.size else np.array([], dtype=np.float32)
    dt_default = float(np.median(dts)) if dts.size else (1.0 / fps_out)

    thr = float(min_prob)

    for i, frame in enumerate(frames_bgr):
        cls = int(preds[i]) if i < preds.size else 0
        cls = int(np.clip(cls, 0, len(labels) - 1))
        lbl = labels[cls]

        dt = float(dts[i]) if i < dts.size else dt_default
        if dt < 0:
            dt = 0.0

        # FINAL FIX FOR TIMER:
        # Only count time towards the current step if that step's pred_prob >= min_prob.
        # Otherwise route that dt to lowconf_label.
        pp = float(pred_prob[i]) if i < pred_prob.size else 0.0
        if thr > 0.0 and pp < thr and cls != int(lowconf_idx):
            step_times[labels[int(lowconf_idx)]] += dt
            lbl_for_ui = labels[int(lowconf_idx)]
        else:
            step_times[lbl] = step_times.get(lbl, 0.0) + dt
            lbl_for_ui = lbl

        frame2 = _draw_step_ui(
            frame_bgr=frame.copy(),
            labels=labels,
            current_label=lbl_for_ui,
            step_times_so_far=step_times,
            target_s=float(step_target_s),
            show_all_steps=bool(show_all_steps),
            exclude_other_from_completion=bool(exclude_other_from_completion),
            exclude_lowconf_from_completion=bool(exclude_lowconf_from_completion),
            font_scale=0.6,
        )

        txt = f"{lbl_for_ui}  {pp * 100.0:5.1f}%"
        cv2.putText(frame2, txt, (w - 420, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame2, txt, (w - 420, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        vw.write(frame2)

    summary_frames = _make_summary_frames(
        base_frame_bgr=frames_bgr[-1],
        labels=labels,
        total_times=step_times,
        target_s=float(step_target_s),
        exclude_other_from_completion=bool(exclude_other_from_completion),
        exclude_lowconf_from_completion=bool(exclude_lowconf_from_completion),
        fps=float(fps_out),
        seconds=float(summary_seconds),
    )
    for sf in summary_frames:
        vw.write(sf)

    vw.release()
    print(f"  [video] wrote mp4 using codec: {codec}")


# ----------------------------
# CLI
# ----------------------------
def parse_labels(labels_arg: Optional[str], cfg: Optional[object]) -> List[str]:
    if labels_arg:
        parts = [p.strip() for p in labels_arg.split(",")]
        return [p for p in parts if p]
    if cfg is not None:
        for attr in ("CLASS_NAMES", "CLASSES", "LABELS"):
            if hasattr(cfg, attr):
                v = getattr(cfg, attr)
                if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
                    return list(v)
    return []


def parse_img_size(img_size_arg: Optional[str], cfg: Optional[object]) -> Tuple[int, int]:
    if img_size_arg:
        m = img_size_arg.lower().replace(" ", "")
        if "x" in m:
            a, b = m.split("x", 1)
            return int(a), int(b)
        if "," in m:
            a, b = m.split(",", 1)
            return int(a), int(b)
        raise ValueError(f"Invalid --img-size '{img_size_arg}'. Use '224x224' or '224,224'.")
    if cfg is not None and hasattr(cfg, "IMG_SIZE"):
        v = getattr(cfg, "IMG_SIZE")
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return int(v[0]), int(v[1])
    return (224, 224)


def main() -> int:
    ap = argparse.ArgumentParser(description="Annotate videos with a handwash model.")
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--video", type=str, help="Path to a single input video.")
    g_in.add_argument("--video-dir", type=str, help="Directory containing videos.")
    ap.add_argument("--glob", type=str, default="*.mp4", help="Glob inside --video-dir (default: *.mp4)")

    ap.add_argument("--model", type=str, required=True, help="Path to a Keras model (.keras/.h5/SavedModel dir).")
    ap.add_argument("--out", type=str, required=True, help="Output directory.")
    ap.add_argument("--config", type=str, default=None, help="Optional explicit path to config.py.")
    ap.add_argument("--labels", type=str, default=None, help="Comma-separated labels if config is missing.")
    ap.add_argument("--img-size", type=str, default=None, help="Resize frames to HxW (e.g. 224x224).")

    ap.add_argument("--frame-stride", type=int, default=1, help="Sample every Nth frame from input video.")
    ap.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames to process.")

    ap.add_argument("--sequence-length", type=int, default=None, help="Sequence model length T (default from config or 16).")
    ap.add_argument("--sequence-stride", type=int, default=1, help="Sequence window stride.")
    ap.add_argument("--batch-size", type=int, default=16, help="Predict batch size.")

    ap.add_argument("--smooth-probs-window", type=int, default=9, help="Moving average window on probabilities (0/1 disables).")
    ap.add_argument("--smooth-vote-window", type=int, default=0, help="Majority vote window on labels (0/1 disables).")
    ap.add_argument("--min-seg-duration", type=float, default=0.0, help="Drop segments shorter than this duration (seconds).")

    # Correct min-prob threshold behavior
    ap.add_argument("--min-prob", type=float, default=0.0, help="Minimum prob of FINAL label; else route to lowconf label.")
    ap.add_argument("--lowconf-label", type=str, default="No action", help="Label for low-confidence frames (appended if missing).")

    ap.add_argument("--fill-islands-max-s", type=float, default=0.5, help="Fill short isolated label islands (seconds).")

    ap.add_argument("--write-video", action="store_true", help="Also write annotated mp4 with overlay + summary.")
    ap.add_argument("--show-all-steps", action="store_true", help="Show all steps list (checkbox checked). Default ON.")
    ap.add_argument("--hide-all-steps", action="store_true", help="Hide list (checkbox unchecked).")
    ap.add_argument("--step-target-s", type=float, default=3.0, help="Target seconds per step for completion/progress bars.")
    ap.add_argument("--exclude-others", action="store_true", help="Exclude 'Others' from completion (default ON).")
    ap.add_argument("--include-others", action="store_true", help="Include 'Others' in completion.")
    ap.add_argument("--exclude-lowconf", action="store_true", help="Exclude lowconf label from completion (default ON).")
    ap.add_argument("--include-lowconf", action="store_true", help="Include lowconf label in completion.")
    ap.add_argument("--summary-seconds", type=float, default=3.0, help="How long to show summary screen at end (seconds).")

    ap.add_argument("--continue-on-error", action="store_true", help="Skip failing videos and continue.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config)) if args.config else load_cfg(None)
    labels = parse_labels(args.labels, cfg)
    img_size = parse_img_size(args.img_size, cfg)

    seq_len = args.sequence_length
    if seq_len is None:
        if cfg is not None and hasattr(cfg, "SEQUENCE_LENGTH"):
            try:
                seq_len = int(getattr(cfg, "SEQUENCE_LENGTH"))
            except Exception:
                seq_len = 16
        else:
            seq_len = 16

    model_path = Path(args.model).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_robust(model_path)

    if args.video:
        videos = [Path(args.video).expanduser().resolve()]
    else:
        vdir = Path(args.video_dir).expanduser().resolve()
        videos = sorted(vdir.glob(args.glob))

    if not videos:
        raise SystemExit("No videos matched your input arguments.")

    for vp in videos:
        stem = vp.stem
        print(f"[annotate] {vp.name}")
        try:
            fps, inputs_arr, frames_bgr, timestamps = collect_video_frames(
                vp, img_size=img_size, frame_stride=args.frame_stride, max_frames=args.max_frames
            )

            probs = predict_probs(
                model,
                inputs_arr,
                sequence_length=int(seq_len),
                sequence_stride=int(args.sequence_stride),
                batch_size=int(args.batch_size),
            )

            # Decide labels
            if labels and probs.shape[1] != len(labels):
                print(f"  [warn] model outputs {probs.shape[1]} classes but you provided {len(labels)} labels. Using numeric ids.")
                labels_for_use: List[str] = [str(i) for i in range(probs.shape[1])]
            elif labels:
                labels_for_use = list(labels)
            else:
                labels_for_use = [str(i) for i in range(probs.shape[1])]

            # Smooth probabilities
            if args.smooth_probs_window and args.smooth_probs_window > 1:
                probs = moving_average_probs(probs, window=int(args.smooth_probs_window))

            # Base label = argmax
            preds = probs.argmax(axis=1) if probs.size else np.zeros((timestamps.size,), np.int32)

            # Label-only post-processing
            if args.smooth_vote_window and args.smooth_vote_window > 1:
                preds = majority_vote(preds, window=int(args.smooth_vote_window))

            preds = fill_short_islands(preds, timestamps, max_island_s=float(args.fill_islands_max_s))

            # CRITICAL FIX: compute pred_prob of FINAL label, then route low confidence to "No action"
            preds, pred_prob, lowconf_idx = route_low_confidence_frames(
                probs=probs,
                preds=preds,
                labels_for_use=labels_for_use,
                min_prob=float(args.min_prob),
                lowconf_label=str(args.lowconf_label),
            )

            segments = build_segments(
                labels=labels_for_use,
                preds=preds,
                pred_prob=pred_prob,
                timestamps=timestamps,
                min_duration_s=float(args.min_seg_duration),
            )

            write_frame_csv(out_dir / f"{stem}_frame_predictions.csv", labels_for_use, timestamps, probs, preds, pred_prob)
            write_segments_csv(out_dir / f"{stem}_segments.csv", segments)

            if args.write_video:
                if cv2 is None:
                    print("  [warn] cv2 not available; skipping annotated video.")
                else:
                    stride = max(1, int(args.frame_stride))
                    fps_out = float(fps) / float(stride)

                    show_all = True
                    if args.hide_all_steps:
                        show_all = False
                    elif args.show_all_steps:
                        show_all = True

                    exclude_others = True
                    if args.include_others:
                        exclude_others = False
                    elif args.exclude_others:
                        exclude_others = True

                    exclude_lowconf = True
                    if args.include_lowconf:
                        exclude_lowconf = False
                    elif args.exclude_lowconf:
                        exclude_lowconf = True

                    write_annotated_video(
                        out_mp4=out_dir / f"{stem}_annotated.mp4",
                        frames_bgr=[f.copy() for f in frames_bgr],
                        fps_out=fps_out,
                        labels=labels_for_use,
                        preds=preds,
                        pred_prob=pred_prob,
                        timestamps=timestamps,
                        min_prob=float(args.min_prob),
                        lowconf_label=str(args.lowconf_label),
                        step_target_s=float(args.step_target_s),
                        show_all_steps=show_all,
                        exclude_other_from_completion=exclude_others,
                        exclude_lowconf_from_completion=exclude_lowconf,
                        summary_seconds=float(args.summary_seconds),
                    )

            print(f"  wrote: {stem}_frame_predictions.csv, {stem}_segments.csv" + (" + annotated.mp4" if args.write_video else ""))

        except Exception as e:
            print(f"  [error] failed on {vp.name}: {e}")
            if args.continue_on_error:
                continue
            raise

    print(f"[done] outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())