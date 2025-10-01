import numpy as np

def iou(boxA, boxB):
    xx1 = max(boxA[0], boxB[0])
    yy1 = max(boxA[1], boxB[1])
    xx2 = min(boxA[2], boxB[2])
    yy2 = min(boxA[3], boxB[3])
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    A = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    B = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    if A + B - inter == 0:
        return 0.0
    return inter / (A + B - inter + 1e-8)

def nms(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep=[]
    while order.size>0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds+1]
    return boxes[keep].astype(int).tolist(), scores[keep].tolist()

def make_square_bbox(xmin, ymin, xmax, ymax, img_w=None, img_h=None, pad=6):
    w = xmax - xmin
    h = ymax - ymin
    side = max(w, h)
    cx = xmin + w//2
    cy = ymin + h//2
    x_min_new = int(max(0, cx - side//2 - pad))
    y_min_new = int(max(0, cy - side//2 - pad))
    x_max_new = int(cx + side/2 + pad)
    y_max_new = int(cy + side/2 + pad)
    if img_w is not None:
        x_max_new = min(x_max_new, img_w-1)
    if img_h is not None:
        y_max_new = min(y_max_new, img_h-1)
    return [x_min_new, y_min_new, x_max_new, y_max_new]
