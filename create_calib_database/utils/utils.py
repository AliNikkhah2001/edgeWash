from typing import List
import lap
import numpy as np
from utils.detectbox import DetectState

def _point_in_rect(cx: float, cy: float, rect: List[float]) -> bool:
    """Check if point center (cx,cy) is inside rect [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = rect
    return x1 <= cx <= x2 and y1 <= cy <= y2

def find_point_in_zone(box_,zone_layout=None):
    # zones_=zone_layout.get("zones",None)
    if zone_layout is None:
        return 'room'
    
    zone_now='room'
    x1, y1, x2, y2 = box_
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    for z in zone_layout:
        zb = z.get("bbox")
        if not zb and "points" in z:
            xs = [p[0] for p in z["points"]]
            ys = [p[1] for p in z["points"]]
            zb = [min(xs), min(ys), max(xs), max(ys)]
        if _point_in_rect(cx, cy, zb):
            zone_now = z["id"]
            break
    return zone_now


def compute_iou(a_boxes, b_boxes):
    """
    Compute cost based on IoU
    :type a_boxes: list[tlbr] | np.ndarray
    :type b_boxes: list[tlbr] | np.ndarray

    :rtype iou | np.ndarray
    """
    iou = np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float32)
    if iou.size == 0:
        return iou
    a_boxes = np.ascontiguousarray(a_boxes, dtype=np.float32)
    b_boxes = np.ascontiguousarray(b_boxes, dtype=np.float32)
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = a_boxes.T
    b2_x1, b2_y1, b2_x2, b2_y2 = b_boxes.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # box2 area
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (box2_area + box1_area[:, None] - inter_area + 1E-7)

def assignment_dets2boxes(
        dets:List[DetectState],
        anotated_boxes,
        ppe_list,
        scores,
        cam_id):
    human_score_cam=scores
    human_ppes_camb=ppe_list
    # NOTE :some time lenght "anotated_boxes" is bigger or smaller than "ppe_list" and "scores"
    x=np.empty(shape=[0,1])
    if len(dets) and len(anotated_boxes):
        dets_boxes=[z.bbox for z in dets]
        ## TODO : using nmx to find duble boxes
        cost_matrix=1-compute_iou(dets_boxes,anotated_boxes)
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True,cost_limit=0.985)

        for row_, col_ in enumerate(x):
            fisheye_det=dets[row_]
            if col_<0: ## con not assing box from cam_b on this detect
                continue
            ## if found -> update fisheye detect
            det_box=anotated_boxes[col_]
            if col_<len(human_ppes_camb):
                dets[row_].score=max(human_score_cam[col_],dets[row_].score)
                dets[row_].ppe=human_ppes_camb[col_]
            
        un_assgined_ = np.where(y < 0)[0]
    else:
        un_assgined_=[]
        for it in range(len(anotated_boxes)):
            un_assgined_.append(it)

    
    for it in un_assgined_:
        if it<len(human_ppes_camb):
            dets.append(DetectState(zone_id='',bbox=anotated_boxes[it],
                                        score=human_score_cam[it],
                                        cam_id=cam_id,
                                        ppe=human_ppes_camb[it]))

    return dets,x