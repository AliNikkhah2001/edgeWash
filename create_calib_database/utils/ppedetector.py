###  PPE detector on CAM_B

import numpy as np
import cv2
import base64

class PPEDetector():
    def __init__(
            self,
            ppe_model_path="./storages/Camera_B_Final_Version_Head_PPE_Polygon.pt",
            conf_ppe:float=0.3):
        self.conf_ppe=conf_ppe
        try:
            from ultralytics import YOLO  # pip install ultralytics
            self._model_ppe   = YOLO(ppe_model_path)
            self._model_ppe.fuse() if hasattr(self._model_ppe, "fuse") else None
            return
        except Exception as e:
            raise ValueError("ERROR, loading model got exception : "+str(e))
    
    def convertimg2str(self,imgnp):
        retval, buffer = cv2.imencode('.jpg', imgnp)
        jpg_as_text = base64.b64encode(buffer)
        return jpg_as_text.decode("utf-8")
    
    def convert_str2img(self,jpg_as_text:str):
        jpg_original = base64.b64decode(jpg_as_text)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        image_buffer = cv2.imdecode(jpg_as_np, flags=1)
        return image_buffer
    
    def detect(self,head_np:np.ndarray):
        results_b = self._model_ppe.predict(head_np, conf=self.conf_ppe, verbose=False)
        if len(results_b) == 0 or results_b[0].boxes is None:
            return []
        
        ppe_result=[]
        # for d in results_b:
        #     p1=[]
        d=results_b[0]
        dets_b = d.boxes.data.cpu().numpy()
        for bx1, by1, bx2, by2, conf_b, cls_b in dets_b:
            cls_b = int(cls_b)
            label_b = self._model_ppe.names[cls_b]   # {'beardnet','earmuff','hairnet','hardhat'}
            # p1.append(label_b)
            ppe_result.append(label_b)
        return ppe_result

