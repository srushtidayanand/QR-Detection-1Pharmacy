import json, glob, os
from pathlib import Path

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = boxAArea + boxBArea - interArea
    if union==0: return 0
    return interArea/union

def load_gt_labels(labels_folder):
    # expects YOLO labels (normalized); convert to pixel coords using the images (skipped here)
    # Simplest: assume you also have a JSON GT or other format. Implement as needed.
    return {}

if __name__ == '__main__':
    print("This is a placeholder. Use your GT to compare submissions using IoU thresholds (e.g. 0.5).")
