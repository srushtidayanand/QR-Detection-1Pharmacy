# evaluate.py




def iou(boxA, boxB):
# box format [x1,y1,x2,y2]
xA = max(boxA[0], boxB[0])
yA = max(boxA[1], boxB[1])
xB = min(boxA[2], boxB[2])
yB = min(boxA[3], boxB[3])
interW = max(0, xB - xA)
interH = max(0, yB - yA)
interArea = interW * interH
boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
if boxAArea + boxBArea - interArea == 0:
return 0
return interArea / float(boxAArea + boxBArea - interArea)




def main(gt_path, pred_path, iou_thresh=0.5):
gt = load_json(gt_path)
pred = load_json(pred_path)
gt_map = {item['image_id']: item['qrs'] for item in gt}
pred_map = {item['image_id']: item['qrs'] for item in pred}
total = 0
tp = 0
for img_id, gt_qrs in gt_map.items():
total += len(gt_qrs)
pred_qs = pred_map.get(img_id, [])
matched = set()
for g in gt_qrs:
gb = g['bbox']
found = False
for i, p in enumerate(pred_qs):
pb = p['bbox']
if i in matched:
continue
if iou(gb, pb) >= iou_thresh:
tp += 1
matched.add(i)
found = True
break
recall = tp/total if total>0 else 0
precision = tp/len(pred) if len(pred)>0 else 0
print(f"TP: {tp}, Total GT: {total}, Recall: {recall:.4f}")


if __name__ == '__main__':
import sys
if len(sys.argv) < 3:
print('Usage: python evaluate.py <gt.json> <pred.json>')
else:
main(sys.argv[1], sys.argv[2])
