
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
from typing import Callable, Optional, Tuple, List,Dict
import numpy as np
import lap
import time
import copy
import cv2

from utils.utils import compute_iou
from ai_edge_litert.interpreter import Interpreter
from utils.dev_configs import CAM_B,CAM_D,CAM_F
from utils.timeutil import ts_to_iso
from utils.ppedetector import PPEDetector
# interpreter = Interpreter(model_path=args.model_file)


class DetectionFisheye:
    """
    Unified detector wrapper:
      - PyTorch (Ultralytics YOLO .pt)
      - TFLite (int8)
    API:
      detect(frame_bgr) -> (boxes_norm, scores)
      boxes_norm: List[List[x1,y1,x2,y2]] normalized to 0..1
    """
    def __init__(self, model_path: str="./storages/camf_detector.tflite", conf: float = 0.50):
        self.conf = float(conf)
        try:
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            input_shape = self.input_details[0]['shape']  
            self.input_dtype = self.input_details[0]['dtype']
            self.input_h, self.input_w = input_shape[1], input_shape[2]
        except Exception as e:
            raise ValueError("ERROR, loading model got exception : "+str(e))

        # Dynamically map output tensors
        self.idx_num, self.idx_boxes, self.idx_scores, self.idx_classes = None, None, None, None
        for det in output_details:
            shape = tuple(det['shape'])
            name = det['name']
            if shape == (1,): 
                self.idx_num = det['index']
            elif len(shape) == 3 and shape[2] == 4: 
                self.idx_boxes = det['index']
            elif len(shape) == 2 and shape[1] > 1:
                if name.endswith(':1'): 
                    self.idx_scores = det['index']
                elif name.endswith(':2'): 
                    self.idx_classes = det['index']
                else:
                    if self.idx_scores is None: 
                        self.idx_scores = det['index']
                    else: 
                        self.idx_classes = det['index']

        # print(f"Tensor map: boxes={idx_boxes}, scores={idx_scores}, classes={idx_classes}, num={idx_num}\n")

    def _crop_roi(self,frame):
        # ROI_X, ROI_Y, ROI_W, ROI_H = 631, 980, 1836, 1138
        hf,wf,cf=frame.shape
        roi_x1=int(wf*0.21)  ## ROI_X/2992
        roi_y1=int(hf*0.327) ## ROI_Y/2992
        roi_x2=int(wf*0.824) ## ROI_X+ROI_W/2992
        roi_y2=int(wf*0.709) ## ROI_Y+ROI_H/2992
        roi_img=frame[roi_y1:roi_y2,roi_x1:roi_x2,:]
        return roi_img

    def return_roi_to_org(self,bbox):
        ROI_X, ROI_Y, ROI_W, ROI_H = 631, 980, 1836, 1138
        y_min_norm, x_min_norm, y_max_norm, x_max_norm=bbox
        # x1_norm=float((roi_x1*0.61363)+0.21)  ## (ROI_W/2992)+ ROI_X/2992
        # y1_norm=float((roi_y1*0.38034)+0.327) ## (ROI_H/2992)+ ROI_Y/2992
        # x2_norm=float((roi_x2*0.61363)+0.21)  ## (ROI_W/2992)+ ROI_X/2992
        # y2_norm=float((roi_y2*0.38034)+0.327) ## (ROI_H/2992)+ ROI_Y/2992
        # Calculate absolute coordinates inside the ROI

        final_xmin = int((x_min_norm * ROI_W)+ROI_X)
        final_ymin = int((y_min_norm * ROI_H)+ROI_Y)
        final_xmax = int((x_max_norm * ROI_W)+ROI_X)
        final_ymax = int((y_max_norm * ROI_H)+ROI_Y)
        
        # Shift coordinates out to the original 2992x2992 image space
        # final_xmin = int(roi_abs_xmin + ROI_X)
        # final_ymin = int(roi_abs_ymin + ROI_Y)
        # final_xmax = int(roi_abs_xmax + ROI_X)
        # final_ymax = int(roi_abs_ymax + ROI_Y)
        x1_norm=final_xmin/2992
        y1_norm=final_ymin/2992
        x2_norm=final_xmax/2992
        y2_norm=final_ymax/2992
        
        return [x1_norm,y1_norm,x2_norm,y2_norm]

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        ### convert bgr to rgb
        frame_rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        roi_img = self._crop_roi(frame_rgb)

        # 2. Resize to model requirements
        resized_img = cv2.resize(roi_img,(self.input_w, self.input_h))
        
        # 3. Format dtype based on dynamic model inspection
        if self.input_dtype == np.uint8:
            inp_data=resized_img.astype(np.uint8)
        elif self.input_dtype == np.float32:
            inp_data = (resized_img.astype(np.float32) / 127.5) - 1.0
        else:
            inp_data=resized_img.astype(self.input_dtype)
            
        inp_tensor = np.expand_dims(inp_data, axis=0)
        
        #  INFERENCE 
        self.interpreter.set_tensor(self.input_details[0]['index'], inp_tensor)
        self.interpreter.invoke()
        
        # Extract results
        num_det = min(int(self.interpreter.get_tensor(self.idx_num)[0]) if self.idx_num else 50, 50)
        boxes = self.interpreter.get_tensor(self.idx_boxes)[0][:num_det]
        scores = self.interpreter.get_tensor(self.idx_scores)[0][:num_det]
        if self.idx_classes is not None:
            classes = self.interpreter.get_tensor(self.idx_classes)[0][:num_det]
        else:
            classes = [1] * num_det 

        
        boxes_out: List[List[float]] = []
        scores_out: List[float] = []
        extra_data=[]
        for i in range(num_det):
            score = float(scores[i])
            if score < self.conf:
                continue
            
            # TFLite outputs normalized boxes [ymin, xmin, ymax, xmax] relative to the ROI
            # y_min_norm, x_min_norm, y_max_norm, x_max_norm = boxes[i]
            # Calculate absolute coordinates inside the ROI
            det_box=self.return_roi_to_org(boxes[i])
            boxes_out.append(det_box)
            # boxes_out.append(boxes[i])
            scores_out.append(score)
        return boxes_out,scores_out,extra_data

    def detect_json(self, frame_bgr: np.ndarray) -> Dict[str,List]:
        t0_=time.time()
        sequence_id=int(t0_*10)
        time_=ts_to_iso(t0_)
        ### ###################################################
        frame_rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        roi_img = self._crop_roi(frame_rgb)

        # 2. Resize to model requirements
        resized_img = cv2.resize(roi_img,(self.input_w, self.input_h))
        
        # 3. Format dtype based on dynamic model inspection
        if self.input_dtype == np.uint8:
            inp_data=resized_img.astype(np.uint8)
        elif self.input_dtype == np.float32:
            inp_data = (resized_img.astype(np.float32) / 127.5) - 1.0
        else:
            inp_data=resized_img.astype(self.input_dtype)
            
        inp_tensor = np.expand_dims(inp_data, axis=0)
        
        #  INFERENCE 
        self.interpreter.set_tensor(self.input_details[0]['index'], inp_tensor)
        self.interpreter.invoke()
        
        # Extract results
        num_det = min(int(self.interpreter.get_tensor(self.idx_num)[0]) if self.idx_num else 50, 50)
        boxes = self.interpreter.get_tensor(self.idx_boxes)[0][:num_det]
        scores = self.interpreter.get_tensor(self.idx_scores)[0][:num_det]
        if self.idx_classes is not None:
            classes = self.interpreter.get_tensor(self.idx_classes)[0][:num_det]
        else:
            classes = [1] * num_det 

        

        payload_f={
            "schema_version": "v1",
            "cam_id": str(CAM_F),
            "ts": str(time_),
            "sequence_id":sequence_id,
            "fps":8.0,
            "boxes":[],
            "scores":[],
            "extra":{"extra":[]}
            }
        for i in range(num_det):
            score = float(scores[i])
            if score < self.conf:
                continue
            
            # TFLite outputs normalized boxes [ymin, xmin, ymax, xmax] relative to the ROI
            # y_min_norm, x_min_norm, y_max_norm, x_max_norm = boxes[i]
            # Calculate absolute coordinates inside the ROI
            y_min_norm, x_min_norm, y_max_norm, x_max_norm=boxes[i]
            b3=[
                float(x_min_norm),
                float(y_min_norm),
                float(x_max_norm),
                float(y_max_norm)]
            payload_f["boxes"].append(b3)
            payload_f["scores"].append(score)
        
        ############################################################
        return payload_f



     
CLASS_MAP = {1: "Head", 2: "Human"} 
class DetectorCamerasB:
    def __init__(
            self,
            detect_model_path:str="./storages/camb_detector.tflite",
            ppe_model_path:str="",
            conf_detect:float=0.30,
            conf_ppe:float=0.3,
            cost_lime_ann:float=0.67):
        # MODEL_A_PATH = "storages/Human_Head_Best_V1.pt"              # Human + Head
        # MODEL_B_PATH = "storages/Camera_B_Final_Version_Head_PPE_Polygon.pt"   # PPE
        self.conf = conf_detect   # Human/Head
        self.conf_thr_modelB=conf_ppe   # PPE
        self.cost_lime_ann=cost_lime_ann

        # self.interpreter = tf.lite.Interpreter(model_path=detect_model_path)
        self.interpreter = Interpreter(model_path=detect_model_path)
        self.interpreter.allocate_tensors()


        self.input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Dynamically extract input requirements
        input_shape = self.input_details[0]['shape']  
        self.input_dtype = self.input_details[0]['dtype']
        self.input_h, self.input_w = input_shape[1], input_shape[2]
        #################  Create PPE detector for CamB #############
        self.ppe_detector=PPEDetector()

        print(f"Model Input Requirement: Shape {input_shape}, Dtype {self.input_dtype}")

        # Dynamically map output tensors based on shape heuristics
        self.idx_num, self.idx_boxes, self.idx_scores, self.idx_classes = None, None, None, None
        for det in output_details:
            shape = tuple(det['shape'])
            name = det['name']
            if shape == (1,):
                self.idx_num = det['index']
            elif len(shape) == 3 and shape[2] == 4:
                self.idx_boxes = det['index']
            elif len(shape) == 2 and shape[1] > 1:
                if name.endswith(':1'): self.idx_scores = det['index']
                elif name.endswith(':2'): self.idx_classes = det['index']
                else:
                    if self.idx_scores is None: self.idx_scores = det['index']
                    else: self.idx_classes = det['index']

    def _assign_baseed_on_headbox(self,humanboxes_norm,headboxes_norm):
        #### create Left Box for Head
        human_heads_boxes_=[]
        for det_ in humanboxes_norm:
            x1, y1, x2, y2 = det_
            #### create Left Box for Head
            human_heads_boxes_.append([x1,y1,x1+((x2-x1)/2),y1+((y2-y1)/5)])
            #### -----  Create Mid Box for Head
            h_c_box = self._create_head_box(det_)
            human_heads_boxes_.append(h_c_box)
            ####  ----  Create Right Box for Head
            human_heads_boxes_.append([x2-((x2-x1)/2),y1,x2,y1+((y2-y1)/5)])

        
        
        
        human_heads_boxes_=np.array(human_heads_boxes_)
        head_boxes_np=np.array(headboxes_norm)
        ## cost_matrix shape (number of head)x(number human)
        ########----------  Find Match boxes from Left side Box
        cost_matrix=1-compute_iou(head_boxes_np, human_heads_boxes_) 
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True,cost_limit=self.cost_lime_ann)
        x_out=np.ones(shape=[len(headboxes_norm),],dtype=int)*(-1)
        y_out=np.ones(shape=[len(humanboxes_norm),],dtype=int)*(-1)
        for idx,h_id in enumerate(x):
            if h_id<0: ## con not assing human box on this head box
                x_out[idx]=-1
                continue
            h_id_2=int(h_id/3)
            x_out[idx]=h_id_2
            y_out[h_id_2]=idx
            
        ########----------  Find Match boxes from Mid Box
        # un_assgined_y = np.where(y < 0)[0]
        # un_assgined_x = np.where(x < 0)[0]
        # if len(un_assgined_y) and len(un_assgined_x):
        #     un_assigned_heads=head_boxes_np[un_assgined_x]
        #     un_assigned_humans=human_heads_boxes_m[un_assgined_y]
        #     cost_matrix2=1-_compute_iou(un_assigned_heads, un_assigned_humans) 
        #     _, x2, y2 = lap.lapjv(cost_matrix2, extend_cost=True,cost_limit=self.cost_lime_ann)
        #     for idx,h_id in enumerate(x2):
        #         if h_id<0:
        #             continue
        #         x[un_assgined_x[idx]]=un_assgined_y[h_id]
        #         y[un_assgined_y[h_id]]=un_assgined_x[idx]
        #         cost_matrix[un_assgined_x[idx],un_assgined_y[h_id]]=cost_matrix2[idx,h_id]
        return x_out,y_out
    def _create_head_box(self,humanbox_norm):
        x1,y1,x2,y2=humanbox_norm
        x1_h=max(0,x1+((x2-x1)*0.2))
        x2_h=min(0.999,x2-((x2-x1)*0.2))
        y2_h=min(0.999,y1+((y2-y1)/4))
        return [x1_h,y1,x2_h,y2_h]
    
    def _create_human_box(self,headboxe_norm): 
        x1,y1,x2,y2=headboxe_norm
        x1_h=max(0,x1-((x2-x1)/3))
        x2_h=min(0.999,x2+((x2-x1)/3))
        y2_h=min(0.999,y2+(3.5*(y2-y1)))
        return [x1_h,y1,x2_h,y2_h]

    def detect_human(self, frame_bgr: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        # Convert BGR (OpenCV default) to RGB
        rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (self.input_w, self.input_h))

        # Format according to model input requirements
        if self.input_dtype == np.uint8:
            inp_data=resized_img.astype(np.uint8)
        elif self.input_dtype == np.float32:
            inp_data = (resized_img.astype(np.float32) / 127.5) - 1.0
        else:
            inp_data=resized_img.astype(self.input_dtype)
            
        input_data = np.expand_dims(inp_data, axis=0)

        # --- INFERENCE ---
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Extract results
        num_det = int(self.interpreter.get_tensor(self.idx_num)[0])
        boxes = self.interpreter.get_tensor(self.idx_boxes)[0]
        scores = self.interpreter.get_tensor(self.idx_scores)[0]
        classes = self.interpreter.get_tensor(self.idx_classes)[0]
        
        # --- POST-PROCESSING & SAVING ---
        humanboxes_norm=[]
        headboxes_norm=[]
        human_socres=[]
        head_scores=[]
        for i in range(num_det):
            score = float(scores[i])
            
            # Filter by threshold
            if score < self.conf:
                continue
                
            # Box coordinates are normalized [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = boxes[i]
            if xmin>0.83:
                continue
            # ASSUMPTION 4: Output class logic (+1 shift depending on export)
            cls_id = int(round(classes[i])) + 1
            cls_name:str = CLASS_MAP.get(cls_id, f"Class_{cls_id}")

            if cls_name.lower() == "human":
                humanboxes_norm.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                human_socres.append(float(score))
            elif cls_name.lower() == "head":
                headboxes_norm.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                head_scores.append(float(score))
        

        ###############   ---- prepare output
        result_out=[]
        if len(humanboxes_norm)<1:
            for h in range(len(headboxes_norm)):
                
                h_box_=self._create_human_box(headboxes_norm[h])
                result_out.append({
                    "head":headboxes_norm[h],
                    "human":h_box_,
                    "conf_head":head_scores[h],
                    "conf_human":None})
            return result_out
        if len(headboxes_norm)<1:
            for h in range(len(humanboxes_norm)):
                h_box_=self._create_head_box(humanboxes_norm[h])
                result_out.append({
                    "head":h_box_,
                    "human":humanboxes_norm[h],
                    "conf_head":None,
                    "conf_human":human_socres[h]})
            return result_out
        #############  -----  assign human to head
        # dist_matrix,x,y=self._assign_baseed_on_center(humanboxes_norm,headboxes_norm)
        x,y=self._assign_baseed_on_headbox(humanboxes_norm,headboxes_norm)
        for idx,h_id in enumerate(x):
            if h_id<0: ## con not assing human box on this head box
                h_box_=self._create_human_box(headboxes_norm[idx])
                result_out.append({
                    "head":headboxes_norm[idx],
                    "human":h_box_,
                    "conf_head":head_scores[idx],
                    "conf_human":None})
                continue
            
            result_out.append({
                "head":headboxes_norm[idx],
                "human":humanboxes_norm[h_id],
                "conf_head":head_scores[idx],
                "conf_human":human_socres[h_id]})
        
        ## TODO : remove to create head for unassigned body
        # ## human box that has not any head box
        # un_assgined_ = np.where(y < 0)[0]
        # for it in un_assgined_:
        #     h_box_=self._create_head_box(humanboxes_norm[it])
        #     result_out.append({
        #         "head":h_box_,
        #         "human":humanboxes_norm[it],
        #         "conf_head":None,
        #         "conf_human":human_socres[it]})

        return result_out
    

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        # Convert BGR (OpenCV default) to RGB
        rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (self.input_w, self.input_h))

        # Format according to model input requirements
        if self.input_dtype == np.uint8:
            inp_data=resized_img.astype(np.uint8)
        elif self.input_dtype == np.float32:
            inp_data = (resized_img.astype(np.float32) / 127.5) - 1.0
        else:
            inp_data=resized_img.astype(self.input_dtype)
            
        input_data = np.expand_dims(inp_data, axis=0)

        # --- INFERENCE ---
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Extract results
        num_det = int(self.interpreter.get_tensor(self.idx_num)[0])
        boxes = self.interpreter.get_tensor(self.idx_boxes)[0]
        scores = self.interpreter.get_tensor(self.idx_scores)[0]
        classes = self.interpreter.get_tensor(self.idx_classes)[0]
        
        # --- POST-PROCESSING & SAVING ---
        humanboxes_norm=[]
        headboxes_norm=[]
        human_socres=[]
        head_scores=[]
        for i in range(num_det):
            score = float(scores[i])
            
            # Filter by threshold
            if score < self.conf:
                continue
                
            # Box coordinates are normalized [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = boxes[i]
            # ymin, xmin, ymax, xmax = y1/1944, x1/2592, y2/1944, x2/2592
            # ASSUMPTION 4: Output class logic (+1 shift depending on export)
            cls_id = int(round(classes[i])) + 1
            cls_name = CLASS_MAP.get(cls_id, f"Class_{cls_id}")

            if cls_name.lower() == "human":
                humanboxes_norm.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                human_socres.append(float(score))
            elif cls_name.lower() == "head":
                headboxes_norm.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                head_scores.append(float(score))
        

        return humanboxes_norm,headboxes_norm,human_socres,head_scores


    
    def detect_json(self, frame_bgr: np.ndarray) -> Dict[str,List]:
        t0_=time.time()
        sequence_id=int(t0_*10)
        time_=ts_to_iso(t0_)
        hf,wf,cf=frame_bgr.shape
        ### TODO : add ppe after created model
        humanboxes,headboxes,human_confs,head_confs=self.detect(frame_bgr=frame_bgr)

        payload_b={
            "schema_version": "v1",
            "cam_id": str(CAM_B),
            "ts": str(time_),
            "sequence_id":sequence_id,
            "fps":8.0,
            "boxes":[],
            "scores":[],
            "labels":[],
            "extra":{"ppe":[]}
            }
        for d in range(len(humanboxes)):
            payload_b["boxes"].append(humanboxes[d])
            payload_b["scores"].append(human_confs[d])
            payload_b["labels"].append("human")
        
        for d in range(len(headboxes)):
            payload_b["boxes"].append(headboxes[d])
            payload_b["scores"].append(head_confs[d])
            payload_b["labels"].append("head")
            x1n,y1n,x2n,y2n =headboxes[d]
            x1_=int(x1n*wf)
            y1_=int(y1n*hf)
            x2_=int(x2n*wf)
            y2_=int(y2n*hf)
            img_np=frame_bgr[y1_:y2_,x1_:x2_,:]
            ppe_res=self.ppe_detector.detect(img_np)
            payload_b["extra"]["ppe"].append(ppe_res)
        
        return payload_b




     
CLASS_MAP_D = {1:"Handwash",2: "Head", 3: "Human"} 
class DetectorCamerasD:
    def __init__(
            self,
            detect_model_path:str="./storages/camb_detector.tflite",
            ppe_model_path:str="",
            conf_detect:float=0.30,
            conf_ppe:float=0.3,
            cost_lime_ann:float=0.67):
        # MODEL_A_PATH = "storages/Human_Head_Best_V1.pt"              # Human + Head
        # MODEL_B_PATH = "storages/Camera_B_Final_Version_Head_PPE_Polygon.pt"   # PPE
        self.conf = conf_detect   # Human/Head
        self.conf_thr_modelB=conf_ppe   # PPE
        self.cost_lime_ann=cost_lime_ann

        # self.interpreter = tf.lite.Interpreter(model_path=detect_model_path)
        self.interpreter = Interpreter(model_path=detect_model_path)
        self.interpreter.allocate_tensors()


        self.input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Dynamically extract input requirements
        input_shape = self.input_details[0]['shape']  
        self.input_dtype = self.input_details[0]['dtype']
        self.input_h, self.input_w = input_shape[1], input_shape[2]
        #################  Create PPE detector for CamB #############
        self.ppe_detector=PPEDetector()

        print(f"Model Input Requirement: Shape {input_shape}, Dtype {self.input_dtype}")

        # Dynamically map output tensors based on shape heuristics
        self.idx_num, self.idx_boxes, self.idx_scores, self.idx_classes = None, None, None, None
        for det in output_details:
            shape = tuple(det['shape'])
            name = det['name']
            if shape == (1,):
                self.idx_num = det['index']
            elif len(shape) == 3 and shape[2] == 4:
                self.idx_boxes = det['index']
            elif len(shape) == 2 and shape[1] > 1:
                if name.endswith(':1'): self.idx_scores = det['index']
                elif name.endswith(':2'): self.idx_classes = det['index']
                else:
                    if self.idx_scores is None: self.idx_scores = det['index']
                    else: self.idx_classes = det['index']

    def _assign_baseed_on_headbox(self,humanboxes_norm,headboxes_norm):
        #### create Left Box for Head
        human_heads_boxes_=[]
        for det_ in humanboxes_norm:
            x1, y1, x2, y2 = det_
            #### create Left Box for Head
            human_heads_boxes_.append([x1,y1,x1+((x2-x1)/2),y1+((y2-y1)/5)])
            #### -----  Create Mid Box for Head
            h_c_box = self._create_head_box(det_)
            human_heads_boxes_.append(h_c_box)
            ####  ----  Create Right Box for Head
            human_heads_boxes_.append([x2-((x2-x1)/2),y1,x2,y1+((y2-y1)/5)])

        
        
        
        human_heads_boxes_=np.array(human_heads_boxes_)
        head_boxes_np=np.array(headboxes_norm)
        ## cost_matrix shape (number of head)x(number human)
        ########----------  Find Match boxes from Left side Box
        cost_matrix=1-compute_iou(head_boxes_np, human_heads_boxes_) 
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True,cost_limit=self.cost_lime_ann)
        x_out=np.ones(shape=[len(headboxes_norm),],dtype=int)*(-1)
        y_out=np.ones(shape=[len(humanboxes_norm),],dtype=int)*(-1)
        for idx,h_id in enumerate(x):
            if h_id<0: ## con not assing human box on this head box
                x_out[idx]=-1
                continue
            h_id_2=int(h_id/3)
            x_out[idx]=h_id_2
            y_out[h_id_2]=idx
            
        ########----------  Find Match boxes from Mid Box
        # un_assgined_y = np.where(y < 0)[0]
        # un_assgined_x = np.where(x < 0)[0]
        # if len(un_assgined_y) and len(un_assgined_x):
        #     un_assigned_heads=head_boxes_np[un_assgined_x]
        #     un_assigned_humans=human_heads_boxes_m[un_assgined_y]
        #     cost_matrix2=1-_compute_iou(un_assigned_heads, un_assigned_humans) 
        #     _, x2, y2 = lap.lapjv(cost_matrix2, extend_cost=True,cost_limit=self.cost_lime_ann)
        #     for idx,h_id in enumerate(x2):
        #         if h_id<0:
        #             continue
        #         x[un_assgined_x[idx]]=un_assgined_y[h_id]
        #         y[un_assgined_y[h_id]]=un_assgined_x[idx]
        #         cost_matrix[un_assgined_x[idx],un_assgined_y[h_id]]=cost_matrix2[idx,h_id]
        return x_out,y_out
    def _create_head_box(self,humanbox_norm):
        x1,y1,x2,y2=humanbox_norm
        x1_h=max(0,x1+((x2-x1)*0.2))
        x2_h=min(0.999,x2-((x2-x1)*0.2))
        y2_h=min(0.999,y1+((y2-y1)/4))
        return [x1_h,y1,x2_h,y2_h]
    
    def _create_human_box(self,headboxe_norm): 
        x1,y1,x2,y2=headboxe_norm
        x1_h=max(0,x1-((x2-x1)/3))
        x2_h=min(0.999,x2+((x2-x1)/3))
        y2_h=min(0.999,y2+(3.5*(y2-y1)))
        return [x1_h,y1,x2_h,y2_h]

    def detect_human(self, frame_bgr: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        # Convert BGR (OpenCV default) to RGB
        rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (self.input_w, self.input_h))

        # Format according to model input requirements
        if self.input_dtype == np.uint8:
            inp_data=resized_img.astype(np.uint8)
        elif self.input_dtype == np.float32:
            inp_data = (resized_img.astype(np.float32) / 127.5) - 1.0
        else:
            inp_data=resized_img.astype(self.input_dtype)
            
        input_data = np.expand_dims(inp_data, axis=0)

        # --- INFERENCE ---
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Extract results
        num_det = int(self.interpreter.get_tensor(self.idx_num)[0])
        boxes = self.interpreter.get_tensor(self.idx_boxes)[0]
        scores = self.interpreter.get_tensor(self.idx_scores)[0]
        classes = self.interpreter.get_tensor(self.idx_classes)[0]
        
        # --- POST-PROCESSING & SAVING ---
        humanboxes_norm=[]
        headboxes_norm=[]
        human_socres=[]
        head_scores=[]
        for i in range(num_det):
            score = float(scores[i])
            
            # Filter by threshold
            if score < self.conf:
                continue
            
            # Box coordinates are normalized [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax= boxes[i]
            # box2= [int(xmin*680), int(ymin*680), int(xmax*680), int(ymax*680)]
            # ASSUMPTION 4: Output class logic (+1 shift depending on export)
            cls_id = int(round(classes[i])) + 1
            cls_name:str = CLASS_MAP_D.get(cls_id, f"Class_{cls_id}")
            # print(f"Detection {i}: Score {score}, Box {box2}, Class {cls_name}")

            if cls_name.lower() == "human":
                humanboxes_norm.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                human_socres.append(float(score))
            elif cls_name.lower() == "head":
                headboxes_norm.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                head_scores.append(float(score))
        

        ###############   ---- prepare output
        result_out=[]
        if len(humanboxes_norm)<1:
            for h in range(len(headboxes_norm)):
                
                h_box_=self._create_human_box(headboxes_norm[h])
                result_out.append({
                    "head":headboxes_norm[h],
                    "human":h_box_,
                    "conf_head":head_scores[h],
                    "conf_human":None})
            return result_out
        if len(headboxes_norm)<1:
            for h in range(len(humanboxes_norm)):
                h_box_=self._create_head_box(humanboxes_norm[h])
                result_out.append({
                    "head":h_box_,
                    "human":humanboxes_norm[h],
                    "conf_head":None,
                    "conf_human":human_socres[h]})
            return result_out
        #############  -----  assign human to head
        # dist_matrix,x,y=self._assign_baseed_on_center(humanboxes_norm,headboxes_norm)
        x,y=self._assign_baseed_on_headbox(humanboxes_norm,headboxes_norm)
        for idx,h_id in enumerate(x):
            if h_id<0: ## con not assing human box on this head box
                h_box_=self._create_human_box(headboxes_norm[idx])
                result_out.append({
                    "head":headboxes_norm[idx],
                    "human":h_box_,
                    "conf_head":head_scores[idx],
                    "conf_human":None})
                continue
            
            result_out.append({
                "head":headboxes_norm[idx],
                "human":humanboxes_norm[h_id],
                "conf_head":head_scores[idx],
                "conf_human":human_socres[h_id]})
        
        ## TODO : remove to create head for unassigned body
        # ## human box that has not any head box
        # un_assgined_ = np.where(y < 0)[0]
        # for it in un_assgined_:
        #     h_box_=self._create_head_box(humanboxes_norm[it])
        #     result_out.append({
        #         "head":h_box_,
        #         "human":humanboxes_norm[it],
        #         "conf_head":None,
        #         "conf_human":human_socres[it]})

        return result_out
    

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        # Convert BGR (OpenCV default) to RGB
        rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (self.input_w, self.input_h))

        # Format according to model input requirements
        if self.input_dtype == np.uint8:
            inp_data=resized_img.astype(np.uint8)
        elif self.input_dtype == np.float32:
            inp_data = (resized_img.astype(np.float32) / 127.5) - 1.0
        else:
            inp_data=resized_img.astype(self.input_dtype)
            
        input_data = np.expand_dims(inp_data, axis=0)

        # --- INFERENCE ---
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Extract results
        num_det = int(self.interpreter.get_tensor(self.idx_num)[0])
        boxes = self.interpreter.get_tensor(self.idx_boxes)[0]
        scores = self.interpreter.get_tensor(self.idx_scores)[0]
        classes = self.interpreter.get_tensor(self.idx_classes)[0]
        
        # --- POST-PROCESSING & SAVING ---
        humanboxes_norm=[]
        headboxes_norm=[]
        human_socres=[]
        head_scores=[]
        for i in range(num_det):
            score = float(scores[i])
            
            # Filter by threshold
            if score < self.conf:
                continue
                
            # Box coordinates are normalized [ymin, xmin, ymax, xmax]
            y1, x1, y2, x2 = boxes[i]
            ymin, xmin, ymax, xmax = y1/1944, x1/2592, y2/1944, x2/2592
            # ASSUMPTION 4: Output class logic (+1 shift depending on export)
            cls_id = int(round(classes[i])) + 1
            cls_name = CLASS_MAP_D.get(cls_id, f"Class_{cls_id}")

            if cls_name.lower() == "human":
                humanboxes_norm.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                human_socres.append(float(score))
            elif cls_name.lower() == "head":
                headboxes_norm.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                head_scores.append(float(score))
        

        return humanboxes_norm,headboxes_norm,human_socres,head_scores


    
    def detect_json(self, frame_bgr: np.ndarray) -> Dict[str,List]:
        t0_=time.time()
        sequence_id=int(t0_*10)
        time_=ts_to_iso(t0_)
        hf,wf,cf=frame_bgr.shape
        ### TODO : add ppe after created model
        humanboxes,headboxes,human_confs,head_confs=self.detect(frame_bgr=frame_bgr)

        payload_b={
            "schema_version": "v1",
            "cam_id": str(CAM_B),
            "ts": str(time_),
            "sequence_id":sequence_id,
            "fps":8.0,
            "boxes":[],
            "scores":[],
            "labels":[],
            "extra":{"ppe":[]}
            }
        for d in range(len(humanboxes)):
            payload_b["boxes"].append(humanboxes[d])
            payload_b["scores"].append(human_confs[d])
            payload_b["labels"].append("human")
        
        for d in range(len(headboxes)):
            payload_b["boxes"].append(headboxes[d])
            payload_b["scores"].append(head_confs[d])
            payload_b["labels"].append("head")
            x1n,y1n,x2n,y2n =headboxes[d]
            x1_=int(x1n*wf)
            y1_=int(y1n*hf)
            x2_=int(x2n*wf)
            y2_=int(y2n*hf)
            img_np=frame_bgr[y1_:y2_,x1_:x2_,:]
            ppe_res=self.ppe_detector.detect(img_np)
            payload_b["extra"]["ppe"].append(ppe_res)
        
        return payload_b