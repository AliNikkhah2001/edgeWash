
# app/tracking/core/track.py
from dataclasses import dataclass
from typing import Optional, Tuple,List

# @dataclass
# class DetectState:
#     zone_id:str
#     bbox: Tuple[float,float,float,float]
#     score: float
#     cam_id: Optional[str] = None
#     ppe:List[str]=[]


@dataclass
class DetectState:
    zone_id: Optional[str]          # str or None
    bbox: List[float]               # list of 4 floats [x1, y1, x2, y2]
    score: float                    # confidence score
    cam_id: Optional[str]            # str or None
    ppe: List[str]                  # list of strings, length not fixed