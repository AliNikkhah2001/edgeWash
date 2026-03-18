import cv2
import json
import os
import random
import copy
import numpy as np
from glob import glob
import av
from utils.tflitedetectors import DetectorCamerasD,DetectionFisheye

# =============================
# CONFIG
# =============================

VIDEO_F_PATH = "videos/record_2_CAM_F.mkv"
VIDEO_B_PATH = "videos/record_2_CAM_D.mkv"

OUTPUT_DIR = "annotate_DF_out_2"

# start from a specific saved frame name instead of from beginning
# set to None if you want normal behavior
START_FROM_SAVED_FRAME = None

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# =============================
# INITIALIZATION
# =============================

detectorb = DetectorCamerasD(detect_model_path="./storages/camera_d_handwash_320_int8.tflite",cost_lime_ann=0.77)
detectorf = DetectionFisheye(model_path="./storages/camf_detector.tflite")

container_f = av.open(VIDEO_F_PATH, mode='r')
stream_f = container_f.streams.video[0]
container_b = av.open(VIDEO_B_PATH, mode='r')
stream_b = container_b.streams.video[0]

FRAME_NUM_F = 0
FRAME_NUM_B = 0

files_ = glob(OUTPUT_DIR + "/*.json")
for f in files_:
    basename = os.path.basename(f)
    vv = basename[:-5]
    numbers_ = vv.split("_")
    frame_nm_f = int(numbers_[-1])
    FRAME_NUM_F = max(FRAME_NUM_F, frame_nm_f)

    frame_nm_b = int(numbers_[-2])
    FRAME_NUM_B = max(FRAME_NUM_B, frame_nm_b)

# if explicit start frame is given, use it
if START_FROM_SAVED_FRAME is not None:
    try:
        vv = START_FROM_SAVED_FRAME
        if vv.endswith(".json") or vv.endswith(".png"):
            vv = os.path.splitext(vv)[0]
        numbers_ = vv.split("_")
        FRAME_NUM_B = int(numbers_[-2])
        FRAME_NUM_F = int(numbers_[-1])
    except Exception:
        print(f"[ERROR] invalid START_FROM_SAVED_FRAME format: {START_FROM_SAVED_FRAME}")
        FRAME_NUM_F = 0
        FRAME_NUM_B = 0

all_detections = {"b": [], "f": []}
paused_f = False
paused_b = False
convas_dim = 680
canvas_np = np.zeros(shape=[convas_dim + 30, convas_dim * 2, 3], dtype=np.uint8)
msg = "[space]: play both, [F]: forward CAM_F, [B]: forward CAM_B, [N]: stop both, [U]: reset frame ann, [Z]: undo last ann, [Q]: exit"
cv2.putText(canvas_np, str(msg), org=(5, convas_dim + 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(255, 255, 255),
            thickness=1)

saved_annotations = []
last_point_selected = None

# =============================
# HELPER FUNCTIONS
# =============================

def random_color():
    return tuple(np.random.randint(40, 255, 3).tolist())

def parse_saved_frame_name(name_str):
    vv = name_str
    if vv.endswith(".json") or vv.endswith(".png"):
        vv = os.path.splitext(vv)[0]
    numbers_ = vv.split("_")
    frame_nm_b = int(numbers_[-2])
    frame_nm_f = int(numbers_[-1])
    return frame_nm_b, frame_nm_f

def seek_to_frame(container, target_frame_num):
    """
    Decode sequentially until target_frame_num is reached.
    Returns the target frame as ndarray.
    target_frame_num is 1-based to match FRAME_NUM_* logic in main loop.
    """
    last_frame = None
    current_frame_num = 0
    for frame_av in container.decode(video=0):
        frame_np = frame_av.to_ndarray(format='bgr24')
        if frame_np is None or not any(frame_np.shape):
            continue
        current_frame_num += 1
        last_frame = frame_np
        if current_frame_num >= target_frame_num:
            break
    return last_frame

def detect_persons_f(frame: np.ndarray):
    detectionsf, scoresf, extra_data_F = detectorf.detect(frame)
    boxes = []
    for i in range(len(detectionsf)):
        y1_ = detectionsf[i][1]
        if y1_ < 0.3:
            continue
        boxes.append({
            "assigned": False,
            "box": detectionsf[i],
            "conf": scoresf[i],
            "color": random_color()
        })
    return boxes

def detect_persons_b(frame):
    result_out = detectorb.detect_human(frame)
    boxes = []
    for det in result_out:
        x1_head = det["head"][0]
        x1_human = det["human"][0]
        y1_head = det["head"][1]
        if x1_head > 0.799:
            continue
        if x1_human > 0.799:
            continue
        if y1_head > 0.722:
            continue
        boxes.append({
            "assigned": False,
            "head": det["head"],
            "human": det["human"],
            "head_conf": det["conf_head"],
            "human_conf": det["conf_human"],
            "color": random_color()
        })
    return boxes

def draw_boxes(frame: np.ndarray, detections: dict, cam_name="fb"):
    ## TODO : x,y is normalized
    convas_w2 = frame.shape[1] // 2
    if "b" in cam_name:
        for det in detections["b"]:
            bboxb_ = det["head"]
            c_ = det["color"]
            bx1 = int(bboxb_[0] * convas_w2) + convas_w2
            by1 = int(bboxb_[1] * convas_w2)
            bx2 = int(bboxb_[2] * convas_w2) + convas_w2
            by2 = int(bboxb_[3] * convas_w2)
            cv2.rectangle(img=frame, pt1=(bx1, by1), pt2=(bx2, by2), color=c_, thickness=2)

            bboxb2_ = det["human"]
            bx12 = int(bboxb2_[0] * convas_w2) + convas_w2
            by12 = int(bboxb2_[1] * convas_w2)
            bx22 = int(bboxb2_[2] * convas_w2) + convas_w2
            by22 = int(bboxb2_[3] * convas_w2)
            cv2.rectangle(img=frame, pt1=(bx12, by12), pt2=(bx22, by22), color=c_, thickness=2)

    if "f" in cam_name:
        for det in detections["f"]:
            fbbox_ = det["box"]
            c_2 = det["color"]
            fx12 = int(fbbox_[0] * convas_w2)
            fy12 = int(fbbox_[1] * convas_w2)
            fx22 = int(fbbox_[2] * convas_w2)
            fy22 = int(fbbox_[3] * convas_w2)
            cv2.rectangle(img=frame, pt1=(fx12, fy12), pt2=(fx22, fy22), color=c_2, thickness=2)

def redraw_current_canvas():
    global canvas_np, frame_f, frame_b, all_detections
    imgf = cv2.resize(frame_f, dsize=[convas_dim, convas_dim])
    imgb = cv2.resize(frame_b, dsize=[convas_dim, convas_dim])
    canvas_np[:convas_dim, :convas_dim, :] = copy.deepcopy(imgf)
    canvas_np[:convas_dim, convas_dim:, :] = copy.deepcopy(imgb)
    draw_boxes(canvas_np, all_detections, cam_name="fb")

def undo_last_annotation():
    global saved_annotations, all_detections, last_point_selected

    if len(saved_annotations) == 0:
        print("[INFO] no annotation to undo on this frame")
        return

    last_pair = saved_annotations.pop()

    # restore B detection
    for i in range(len(all_detections["b"])):
        det = all_detections["b"][i]
        if det["head"] == last_pair["b"]["head"] and det["human"] == last_pair["b"]["human"]:
            all_detections["b"][i]["assigned"] = False
            all_detections["b"][i]["color"] = last_pair["b"]["color"]
            break

    # restore F detection
    for i in range(len(all_detections["f"])):
        det = all_detections["f"][i]
        if det["box"] == last_pair["f"]["box"]:
            all_detections["f"][i]["assigned"] = False
            all_detections["f"][i]["color"] = last_pair["f"]["color"]
            break

    last_point_selected = None
    redraw_current_canvas()
    print("[INFO] last annotation undone")

def save_json():
    '''
    frame f and frame b is same size
    '''
    global saved_annotations, FRAME_NUM_F, FRAME_NUM_B, canvas_np, all_detections
    img_path = f"/d_{FRAME_NUM_B}_{FRAME_NUM_F}.png"
    json_path = f"/d_{FRAME_NUM_B}_{FRAME_NUM_F}.json"

    if os.path.exists(OUTPUT_DIR + json_path):
        os.remove(OUTPUT_DIR + json_path)

    if os.path.exists(OUTPUT_DIR + img_path):
        os.remove(OUTPUT_DIR + img_path)

    new_dict = {"assigned": saved_annotations, "unassigned": []}
    for d in range(len(all_detections["b"])):
        det = all_detections["b"][d]
        if det["assigned"]:
            continue
        new_dict["unassigned"].append(det)

    with open(OUTPUT_DIR + json_path, "w") as f:
        json.dump(new_dict, f, indent=4)

    saved_annotations = []
    cv2.imwrite(OUTPUT_DIR + img_path, canvas_np)

# =============================
# MOUSE CALLBACK
# =============================

def mouse_callback(event, x, y, flags, param):
    global saved_annotations, canvas_np, last_point_selected, all_detections

    if event == cv2.EVENT_LBUTTONDOWN:
        convas_w2 = canvas_np.shape[1] // 2
        if x > convas_w2:  ## on camera B frame
            last_point_selected = (x, y)
            return
        else:  ## on camera F frame
            if last_point_selected is None:
                print("[ERROR] Selecet source from Frame B on Head (Left side) ....")
                return
            xb, yb = last_point_selected
            x_normb = (xb - convas_w2) / convas_w2
            y_normb = yb / convas_w2

            paire_value = {"b": None, "f": None}
            b_number_det = -1

            for d in range(len(all_detections["b"])):
                b_number_det += 1
                det = all_detections["b"][d]
                if det["assigned"]:
                    continue
                x1b, y1b, x2b, y2b = det["head"]
                if x1b <= x_normb <= x2b and y1b <= y_normb <= y2b:
                    paire_value["b"] = copy.deepcopy(det)
                    break
            else:
                print("[ERROR] can not found and Head Box that mached to selected point on CAM B")
                return

            x_normf = x / convas_w2
            y_normf = y / convas_w2
            f_number_det = -1
            for d in range(len(all_detections["f"])):
                f_number_det += 1
                det = all_detections["f"][d]
                if det["assigned"]:
                    continue
                x1f, y1f, x2f, y2f = det["box"]
                if x1f <= x_normf <= x2f and y1f <= y_normf <= y2f:
                    paire_value["f"] = copy.deepcopy(det)
                    break
            else:
                print("[ERROR] can not found and Head Box that mached to selected point on CAM B")
                return

            all_detections["b"][b_number_det]["assigned"] = True
            all_detections["b"][b_number_det]["color"] = (0, 0, 0)
            all_detections["f"][f_number_det]["assigned"] = True
            all_detections["f"][f_number_det]["color"] = (0, 0, 0)

            paire_value["b"]["assigned"] = True
            paire_value["f"]["assigned"] = True

            saved_annotations.append(paire_value)

        draw_boxes(canvas_np, all_detections, cam_name="fb")

# =============================
# START FROM SPECIFIC FRAME
# =============================

frame_f = None
frame_b = None

if START_FROM_SAVED_FRAME is not None and FRAME_NUM_F > 0 and FRAME_NUM_B > 0:
    print(f"[INFO] starting from saved frame: B={FRAME_NUM_B}, F={FRAME_NUM_F}")

    frame_f = seek_to_frame(container_f, FRAME_NUM_F)
    frame_b = seek_to_frame(container_b, FRAME_NUM_B)

    if frame_f is not None:
        all_detections["f"] = detect_persons_f(frame_f)
    if frame_b is not None:
        all_detections["b"] = detect_persons_b(frame_b)

    if frame_f is not None and frame_b is not None:
        redraw_current_canvas()
        paused_f = True
        paused_b = True

# =============================
# MAIN LOOP
# =============================

cv2.namedWindow("video")
forward_frames = 0
try_get_frame = 0
STOP_FLAG = False

while True:
    while forward_frames > 1:
        forward_frames -= 1
        frame_f_f = next(container_f.decode(video=0))
        frame_b_f = next(container_b.decode(video=0))

        frame_f = frame_f_f.to_ndarray(format='bgr24')
        if frame_f is None or not any(frame_f.shape):
            if try_get_frame > 60:
                STOP_FLAG = True
                break
            try_get_frame += 1
            continue

        frame_b = frame_b_f.to_ndarray(format='bgr24')
        if frame_b is None or not any(frame_b.shape):
            if try_get_frame > 60:
                STOP_FLAG = True
                break
            try_get_frame += 1
            continue

        imgf = cv2.resize(frame_f, dsize=[convas_dim, convas_dim])
        imgb = cv2.resize(frame_b, dsize=[convas_dim, convas_dim])
        canvas_np[:convas_dim, :convas_dim, :] = copy.deepcopy(imgf)
        canvas_np[:convas_dim, convas_dim:, :] = copy.deepcopy(imgb)
        cv2.imshow("video", canvas_np)
        key = cv2.waitKey(5)

    if STOP_FLAG:
        break

    if not paused_f:
        frame_f_f = next(container_f.decode(video=0))
        frame_f = frame_f_f.to_ndarray(format='bgr24')
        if frame_f is None or not any(frame_f.shape):
            if try_get_frame > 60:
                STOP_FLAG = True
                break
            try_get_frame += 1
            continue

        FRAME_NUM_F += 1
        detections_f = detect_persons_f(frame_f)
        all_detections["f"] = detections_f
        imgf = cv2.resize(frame_f, dsize=[convas_dim, convas_dim])
        canvas_np[:convas_dim, :convas_dim, :] = copy.deepcopy(imgf)
        draw_boxes(canvas_np, all_detections, cam_name="f")
        if len(detections_f) > 0:
            if paused_b:
                paused_f = True

    if not paused_b:
        frame_b_f = next(container_b.decode(video=0))
        frame_b = frame_b_f.to_ndarray(format='bgr24')
        if frame_b is None or not any(frame_b.shape):
            if try_get_frame > 60:
                STOP_FLAG = True
                break
            try_get_frame += 1
            continue

        FRAME_NUM_B += 1
        if len(saved_annotations) > 1:
            save_json()

        detections_b = detect_persons_b(frame_b)
        if len(detections_b) > 0:
            paused_f = True
            paused_b = True

        all_detections["b"] = detections_b
        imgb = cv2.resize(frame_b, dsize=[convas_dim, convas_dim])
        canvas_np[:convas_dim, convas_dim:, :] = copy.deepcopy(imgb)
        draw_boxes(canvas_np, all_detections, cam_name="b")

    cv2.imshow("video", canvas_np)
    cv2.setMouseCallback("video", mouse_callback)

    key = cv2.waitKey(50) & 0xFF

    # =============================
    # KEY CONTROLS
    # =============================
    if key == ord('q'):
        break
    elif key == ord('n'):
        paused_f = True
        paused_b = True
    elif key == ord('b'):
        paused_f = True
        paused_b = False
    elif key == ord('f'):
        paused_b = True
        paused_f = False
    elif key == ord(' '):
        paused_f = False
        paused_b = False
    elif key == ord('u'):
        imgf = cv2.resize(frame_f, dsize=[convas_dim, convas_dim])
        canvas_np[:convas_dim, :convas_dim, :] = copy.deepcopy(imgf)

        imgb = cv2.resize(frame_b, dsize=[convas_dim, convas_dim])
        canvas_np[:convas_dim, convas_dim:, :] = copy.deepcopy(imgb)

        draw_boxes(canvas_np, all_detections, cam_name="fb")
        saved_annotations = []
        last_point_selected = None
    elif key == ord('z'):
        undo_last_annotation()

# =============================
# CLEANUP
# =============================

container_b.close()
container_f.close()
cv2.destroyAllWindows()