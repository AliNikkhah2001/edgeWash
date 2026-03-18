"""
Camera D — Offline Video Inference Pipeline
Loads INT8 TFLite model, runs frame-by-frame detection on a video file,
saves annotated frames as images and per-frame detection data as JSON files.
"""

import os
import json
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# ===================== CONFIGURATION =====================
MODEL_PATH   = "/home/aaa/tmp/camera_D_offline/camera_d_handwash_320_int8.tflite"
VIDEO_PATH   = "/home/aaa/tmp/camera_D_offline/record_2_CAM_D.mkv"
FRAME_DIR    = "/home/aaa/tmp/camera_D_offline/offline_frame_test_2"
JSON_DIR     = "/home/aaa/tmp/camera_D_offline/offline_json_test_2"

INPUT_SIZE   = 320
TARGET_FPS   = 15        # Process / save only N frames per second of video
SCORE_THRESH = 0.30      # Minimum confidence to keep a detection

# Per-class config  (1-indexed after +1 offset applied to TFLite 0-indexed output)
CLASS_NAMES  = {1: "handwash", 2: "head", 3: "human"}
CLASS_COLORS = {1: (0, 0, 255), 2: (255, 100, 0), 3: (0, 165, 255)}   # BGR

# =========================================================

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(JSON_DIR,  exist_ok=True)

# ── Load model ────────────────────────────────────────────
print("Loading TFLite model ...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"  Input  : {input_details[0]['shape']}  dtype={input_details[0]['dtype']}")
print(f"  Outputs: {len(output_details)} tensors")

# ── Resolve output tensor indices (same logic as training pipeline) ──
idx_num = idx_boxes = idx_scores = idx_classes = None

for det in output_details:
    shape = tuple(det['shape'])
    name  = det['name']
    if shape == (1,):
        idx_num = det['index']
    elif len(shape) == 3 and shape[2] == 4:
        idx_boxes = det['index']
    elif len(shape) == 2 and shape[1] > 1:
        if name.endswith(':1'):
            idx_scores = det['index']
        elif name.endswith(':2'):
            idx_classes = det['index']
        else:
            if idx_scores is None:
                idx_scores = det['index']
            else:
                idx_classes = det['index']

print(f"  Tensor map  -> num={idx_num}  boxes={idx_boxes}  "
      f"scores={idx_scores}  classes={idx_classes}")
assert all(v is not None for v in [idx_num, idx_boxes, idx_scores, idx_classes]), \
    "Could not resolve all output tensors — check model output shapes."


# ── Helper: run one frame through the model ───────────────
def run_inference(frame_bgr):
    """
    Args:
        frame_bgr: H×W×3 uint8 numpy array (OpenCV native format)
    Returns:
        list of dicts  {cls_id, cls_name, score, box:{x1,y1,x2,y2}}
        box coordinates are in original pixel space.
    """
    orig_h, orig_w = frame_bgr.shape[:2]

    # Pre-process
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    inp     = np.expand_dims(resized, axis=0).astype(np.uint8)   # (1,320,320,3)

    # Infer
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()

    num_det = int(interpreter.get_tensor(idx_num)[0])
    boxes   = interpreter.get_tensor(idx_boxes)[0]    # (N,4) normalised [ymin,xmin,ymax,xmax]
    scores  = interpreter.get_tensor(idx_scores)[0]   # (N,)
    classes = interpreter.get_tensor(idx_classes)[0]  # (N,)

    detections = []
    for i in range(min(num_det, 100)):
        score = float(scores[i])
        if score < SCORE_THRESH:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        cls_id = int(round(float(classes[i]))) + 1   # 0-indexed -> 1-indexed

        if cls_id not in CLASS_NAMES:
            continue

        detections.append({
            "cls_id":   cls_id,
            "cls_name": CLASS_NAMES[cls_id],
            "score":    round(score, 4),
            "box": {
                "x1": round(float(xmin) * orig_w, 1),
                "y1": round(float(ymin) * orig_h, 1),
                "x2": round(float(xmax) * orig_w, 1),
                "y2": round(float(ymax) * orig_h, 1),
            }
        })

    return detections


# ── Helper: draw bounding boxes on a frame copy ───────────
def draw_detections(frame_bgr, detections):
    vis = frame_bgr.copy()
    for det in detections:
        b     = det["box"]
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        color = CLASS_COLORS.get(det["cls_id"], (255, 255, 255))
        label = f"{det['cls_name']} {det['score']:.0%}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label background pill for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ── Open video ────────────────────────────────────────────
print(f"\nOpening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# How many source frames to skip between saved frames
frame_step = max(1, round(video_fps / TARGET_FPS))

print(f"  Resolution  : {vid_w}x{vid_h}")
print(f"  Source FPS  : {video_fps:.2f}")
print(f"  Target FPS  : {TARGET_FPS}  (keep every {frame_step} source frames)")
print(f"  Total frames: {total_frames}")
print()

# ── Main loop ─────────────────────────────────────────────
saved_count = 0
source_idx  = 0     # absolute frame counter in the video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only process frames that fall on our target cadence
    if source_idx % frame_step == 0:

        detections = run_inference(frame)
        vis_frame  = draw_detections(frame, detections)

        # File names — zero-padded by source frame index for correct sort order
        stem       = f"frame_{source_idx:07d}"
        frame_path = os.path.join(FRAME_DIR, f"{stem}.jpg")
        json_path  = os.path.join(JSON_DIR,  f"{stem}.json")

        # Save annotated frame
        cv2.imwrite(frame_path, vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Build and save detection JSON
        payload = {
            "source_video":   os.path.basename(VIDEO_PATH),
            "source_frame":   source_idx,
            "timestamp_sec":  round(source_idx / video_fps, 4),
            "frame_w":        vid_w,
            "frame_h":        vid_h,
            "num_detections": len(detections),
            "detections":     detections,
        }
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)

        saved_count += 1

        # Progress log every 100 saved frames
        if saved_count % 100 == 0:
            pct = (source_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"  [{pct:5.1f}%] saved frame {saved_count:>5}  "
                  f"(source #{source_idx})  "
                  f"detections={len(detections)}", flush=True)

    source_idx += 1

cap.release()

print(f"\nDone.")
print(f"  Frames saved : {saved_count}  ->  {FRAME_DIR}/")
print(f"  JSONs  saved : {saved_count}  ->  {JSON_DIR}/")
