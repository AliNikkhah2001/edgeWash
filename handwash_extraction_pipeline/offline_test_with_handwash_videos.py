"""
Camera D — Offline Video Inference Pipeline (GPU-only TFLite variant)

Runs INT8 TFLite model on GPU delegate only, logs GPU availability,
saves annotated frames as images and per-frame detection data as JSON files,
and crops detected handwash boxes into separate padded videos.

Also shows tqdm progress with current processed time / full video length.
"""

import os
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ===================== CONFIGURATION =====================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

MODEL_PATH = BASE_DIR / "camera_d_handwash_320_int8.tflite"
VIDEO_PATH = BASE_DIR / "record_2_CAM_D.mkv"

FRAME_DIR = OUTPUT_DIR / "offline_frame_test_2"
JSON_DIR = OUTPUT_DIR / "offline_json_test_2"
HANDWASH_VIDEO_DIR = OUTPUT_DIR / "offline_handwash_clips_gpu_padded"

INPUT_SIZE = 320
TARGET_FPS = 15
SCORE_THRESH = 0.30

HANDWASH_CLASS_ID = 1
HANDWASH_IOU_THRESH = 0.30
HANDWASH_MAX_MISSES = 3
HANDWASH_PADDING_PX = 12
HANDWASH_CROP_SIZE = (224, 224)   # (width, height)
HANDWASH_VIDEO_CODEC = "mp4v"

CLASS_NAMES = {1: "handwash", 2: "head", 3: "human"}
CLASS_COLORS = {
    1: (0, 0, 255),
    2: (255, 100, 0),
    3: (0, 165, 255),
}

# Set this to True if you want to hard-fail when TensorFlow itself sees no GPU.
# Note: TFLite delegate availability is the real requirement here.
REQUIRE_TF_VISIBLE_GPU = False

# Candidate GPU delegate library names / paths
GPU_DELEGATE_CANDIDATES = [
    "libtensorflowlite_gpu_delegate.so",   # Linux common
    "libtensorflowlite_gpu_delegate.dylib",# macOS if manually installed
    "tensorflowlite_gpu_delegate.dll",     # Windows possible naming
]

# =========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)
HANDWASH_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

print("Base dir                 :", BASE_DIR)
print("Output dir               :", OUTPUT_DIR)
print("Saving frames to         :", FRAME_DIR)
print("Saving JSON to           :", JSON_DIR)
print("Saving padded clips to   :", HANDWASH_VIDEO_DIR)

# ── GPU availability logging ──────────────────────────────
print("\nChecking GPU availability ...")
tf_gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow visible GPUs  :", tf_gpus)

if REQUIRE_TF_VISIBLE_GPU and not tf_gpus:
    raise RuntimeError(
        "No GPU is visible to TensorFlow. Refusing to run because GPU-only mode is enabled."
    )

for gpu in tf_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Could not enable memory growth for {gpu}: {e}")

# ── Load TFLite GPU delegate (required) ───────────────────
print("\nLoading TFLite GPU delegate ...")

gpu_delegate = None
delegate_errors = []

for candidate in GPU_DELEGATE_CANDIDATES:
    try:
        gpu_delegate = tf.lite.experimental.load_delegate(candidate)
        print(f"Loaded GPU delegate      : {candidate}")
        break
    except Exception as e:
        delegate_errors.append(f"{candidate}: {repr(e)}")

if gpu_delegate is None:
    raise RuntimeError(
        "GPU-only execution requested, but no TFLite GPU delegate could be loaded.\n\n"
        "Tried:\n- " + "\n- ".join(delegate_errors) + "\n\n"
        "This usually means the GPU delegate library is not installed or not supported on this platform.\n"
        "Your current TFLite Python setup would otherwise fall back to CPU, but this script refuses to do that."
    )

# ── Load model with GPU delegate only ─────────────────────
print("\nLoading TFLite model with GPU delegate only ...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH),
    experimental_delegates=[gpu_delegate],
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  Input  : {input_details[0]['shape']}  dtype={input_details[0]['dtype']}")
print(f"  Outputs: {len(output_details)} tensors")

# ── Resolve output tensor indices ─────────────────────────
idx_num = idx_boxes = idx_scores = idx_classes = None

for det in output_details:
    shape = tuple(det["shape"])
    name = det["name"]

    if shape == (1,):
        idx_num = det["index"]
    elif len(shape) == 3 and shape[2] == 4:
        idx_boxes = det["index"]
    elif len(shape) == 2 and shape[1] > 1:
        if name.endswith(":1"):
            idx_scores = det["index"]
        elif name.endswith(":2"):
            idx_classes = det["index"]
        else:
            if idx_scores is None:
                idx_scores = det["index"]
            else:
                idx_classes = det["index"]

print(
    f"  Tensor map  -> num={idx_num}  boxes={idx_boxes}  "
    f"scores={idx_scores}  classes={idx_classes}"
)

assert all(v is not None for v in [idx_num, idx_boxes, idx_scores, idx_classes]), (
    "Could not resolve all output tensors — check model output shapes."
)


def seconds_to_hms(seconds):
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:05.2f}"


def run_inference(frame_bgr):
    orig_h, orig_w = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    inp = np.expand_dims(resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    num_det = int(interpreter.get_tensor(idx_num)[0])
    boxes = interpreter.get_tensor(idx_boxes)[0]
    scores = interpreter.get_tensor(idx_scores)[0]
    classes = interpreter.get_tensor(idx_classes)[0]

    detections = []
    for i in range(min(num_det, 100)):
        score = float(scores[i])
        if score < SCORE_THRESH:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        cls_id = int(round(float(classes[i]))) + 1

        if cls_id not in CLASS_NAMES:
            continue

        detections.append(
            {
                "cls_id": cls_id,
                "cls_name": CLASS_NAMES[cls_id],
                "score": round(score, 4),
                "box": {
                    "x1": round(float(xmin) * orig_w, 1),
                    "y1": round(float(ymin) * orig_h, 1),
                    "x2": round(float(xmax) * orig_w, 1),
                    "y2": round(float(ymax) * orig_h, 1),
                },
            }
        )

    return detections


def draw_detections(frame_bgr, detections):
    vis = frame_bgr.copy()
    for det in detections:
        b = det["box"]
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        color = CLASS_COLORS.get(det["cls_id"], (255, 255, 255))
        label = f"{det['cls_name']} {det['score']:.0%}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(0, y1 - th - 6)
        cv2.rectangle(vis, (x1, y_top), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def box_to_int_xyxy(box, frame_shape):
    h, w = frame_shape[:2]
    x1 = max(0, min(w - 1, int(round(box["x1"]))))
    y1 = max(0, min(h - 1, int(round(box["y1"]))))
    x2 = max(0, min(w, int(round(box["x2"]))))
    y2 = max(0, min(h, int(round(box["y2"]))))

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def crop_handwash_patch(
    frame_bgr,
    box,
    padding_px=HANDWASH_PADDING_PX,
    out_size=HANDWASH_CROP_SIZE,
):
    """
    Keep aspect ratio.
    Resize crop to fit inside out_size, then pad with black borders.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box_to_int_xyxy(box, frame_bgr.shape)

    x1 = max(0, x1 - padding_px)
    y1 = max(0, y1 - padding_px)
    x2 = min(w, x2 + padding_px)
    y2 = min(h, y2 + padding_px)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    target_w, target_h = out_size
    crop_h, crop_w = crop.shape[:2]

    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def create_handwash_writer(track_id, fps):
    out_path = HANDWASH_VIDEO_DIR / f"handwash_track_{track_id:04d}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*HANDWASH_VIDEO_CODEC)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, HANDWASH_CROP_SIZE)

    if not writer.isOpened():
        raise RuntimeError(f"Could not open handwash clip writer: {out_path}")

    return writer, out_path


def close_track(track):
    writer = track.get("writer")
    if writer is not None:
        writer.release()
        track["writer"] = None


# ── Open video ────────────────────────────────────────────
print(f"\nOpening video: {VIDEO_PATH}")
if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_duration_sec = total_frames / video_fps if video_fps > 0 else 0.0

frame_step = max(1, round(video_fps / TARGET_FPS))
processed_fps = video_fps / frame_step
num_processed_steps = (total_frames + frame_step - 1) // frame_step

print(f"  Resolution            : {vid_w}x{vid_h}")
print(f"  Source FPS            : {video_fps:.2f}")
print(f"  Target FPS            : {TARGET_FPS}  (keep every {frame_step} source frames)")
print(f"  Clip FPS              : {processed_fps:.2f}")
print(f"  Total frames          : {total_frames}")
print(f"  Total duration        : {seconds_to_hms(video_duration_sec)}")
print()

saved_count = 0
source_idx = 0

next_track_id = 1
active_tracks = {}
finished_tracks = []

pbar = tqdm(total=num_processed_steps, desc="Extracting handwash clips", unit="step")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if source_idx % frame_step == 0:
        current_sec = source_idx / video_fps if video_fps > 0 else 0.0

        detections = run_inference(frame)
        vis_frame = draw_detections(frame, detections)

        stem = f"frame_{source_idx:07d}"
        frame_path = FRAME_DIR / f"{stem}.jpg"
        json_path = JSON_DIR / f"{stem}.json"

        cv2.imwrite(str(frame_path), vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        payload = {
            "source_video": VIDEO_PATH.name,
            "source_frame": source_idx,
            "timestamp_sec": round(current_sec, 4),
            "frame_w": vid_w,
            "frame_h": vid_h,
            "num_detections": len(detections),
            "detections": detections,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        handwash_detections = [d for d in detections if d["cls_id"] == HANDWASH_CLASS_ID]
        det_boxes = [box_to_int_xyxy(d["box"], frame.shape) for d in handwash_detections]

        matched_track_ids = set()
        matched_det_indices = set()
        candidate_pairs = []

        for track_id, track in active_tracks.items():
            for det_idx, det_box in enumerate(det_boxes):
                iou = compute_iou(track["last_box"], det_box)
                if iou >= HANDWASH_IOU_THRESH:
                    candidate_pairs.append((iou, track_id, det_idx))

        for iou, track_id, det_idx in sorted(candidate_pairs, reverse=True):
            if track_id in matched_track_ids or det_idx in matched_det_indices:
                continue

            det = handwash_detections[det_idx]
            crop = crop_handwash_patch(frame, det["box"])
            if crop is None:
                continue

            track = active_tracks[track_id]
            track["writer"].write(crop)
            track["last_box"] = det_boxes[det_idx]
            track["last_score"] = det["score"]
            track["last_source_frame"] = source_idx
            track["num_frames"] += 1
            track["misses"] = 0

            matched_track_ids.add(track_id)
            matched_det_indices.add(det_idx)

        for det_idx, det in enumerate(handwash_detections):
            if det_idx in matched_det_indices:
                continue

            crop = crop_handwash_patch(frame, det["box"])
            if crop is None:
                continue

            writer, out_path = create_handwash_writer(next_track_id, processed_fps)
            writer.write(crop)

            active_tracks[next_track_id] = {
                "writer": writer,
                "out_path": out_path,
                "last_box": det_boxes[det_idx],
                "last_score": det["score"],
                "last_source_frame": source_idx,
                "start_source_frame": source_idx,
                "num_frames": 1,
                "misses": 0,
            }

            matched_track_ids.add(next_track_id)
            next_track_id += 1

        tracks_to_close = []
        for track_id, track in active_tracks.items():
            if track_id in matched_track_ids:
                continue

            track["misses"] += 1
            if track["misses"] > HANDWASH_MAX_MISSES:
                close_track(track)
                finished_tracks.append(
                    {
                        "track_id": track_id,
                        "out_path": track["out_path"],
                        "start_source_frame": track["start_source_frame"],
                        "end_source_frame": track["last_source_frame"],
                        "num_frames": track["num_frames"],
                    }
                )
                tracks_to_close.append(track_id)

        for track_id in tracks_to_close:
            del active_tracks[track_id]

        saved_count += 1

        pbar.update(1)
        pbar.set_postfix(
            current=seconds_to_hms(current_sec),
            total=seconds_to_hms(video_duration_sec),
            detections=len(detections),
            active_tracks=len(active_tracks),
            clips=len(finished_tracks),
        )

    source_idx += 1

pbar.close()
cap.release()

for track_id in list(active_tracks.keys()):
    track = active_tracks[track_id]
    close_track(track)
    finished_tracks.append(
        {
            "track_id": track_id,
            "out_path": track["out_path"],
            "start_source_frame": track["start_source_frame"],
            "end_source_frame": track["last_source_frame"],
            "num_frames": track["num_frames"],
        }
    )
    del active_tracks[track_id]

print("\nDone.")
print(f"  Frames saved          : {saved_count}  ->  {FRAME_DIR}")
print(f"  JSONs saved           : {saved_count}  ->  {JSON_DIR}")
print(f"  Handwash clips saved  : {len(finished_tracks)}  ->  {HANDWASH_VIDEO_DIR}")

if finished_tracks:
    print("\nSaved handwash clips:")
    for item in finished_tracks:
        print(
            f"  track={item['track_id']:04d}  "
            f"frames={item['num_frames']:>4}  "
            f"range={item['start_source_frame']}..{item['end_source_frame']}  "
            f"file={Path(item['out_path']).name}"
        )