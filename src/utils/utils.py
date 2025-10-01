# src/utils.py
import os
import json
from typing import List, Dict




def xyxy_to_list(xyxy):
# convert [x1,y1,x2,y2] tensor/array to Python list of ints
return [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]




def write_detection_submission(results: Dict[str, List[List[int]]], out_path: str):
# results: {image_id: [[x1,y1,x2,y2],[...]]}
submission = []
for image_id, boxes in results.items():
submission.append({"image_id": image_id, "qrs": [{"bbox": box} for box in boxes]})
with open(out_path, "w") as f:
json.dump(submission, f, indent=2)




def write_decoding_submission(results: Dict[str, List[Dict]], out_path: str):
# results: {image_id: [{"bbox": [..], "value": "...", "type": "..."}, ...]}
submission = []
for image_id, items in results.items():
submission.append({"image_id": image_id, "qrs": items})
with open(out_path, "w") as f:
json.dump(submission, f, indent=2)
