# infer.py
def run_detection(weights, input_folder, conf_thres=0.25, iou_thres=0.45):
model = YOLO(weights)
results = {}
files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for fn in sorted(files):
path = os.path.join(input_folder, fn)
res = model.predict(source=path, conf=conf_thres, iou=iou_thres, save=False)
# res is a list; take first (single image)
boxes = []
if len(res) > 0 and hasattr(res[0], 'boxes'):
for box in res[0].boxes.xyxy.tolist():
boxes.append(xyxy_to_list(box))
image_id = os.path.splitext(fn)[0]
results[image_id] = boxes
return results




def run_decoding(detection_results, input_folder):
qr_detector = cv2.QRCodeDetector()
decode_results = {}
for image_id, boxes in detection_results.items():
img_path = None
# try common extensions
for ext in ['.jpg', '.jpeg', '.png']:
candidate = os.path.join(input_folder, image_id + ext)
if os.path.exists(candidate):
img_path = candidate
break
if img_path is None:
continue
img = cv2.imread(img_path)
items = []
for bbox in boxes:
x1, y1, x2, y2 = bbox
crop = img[y1:y2, x1:x2]
# attempt decode with OpenCV
try:
data, points, straight_qrcode = qr_detector.detectAndDecode(crop)
except Exception:
data = ''
if data is None:
data = ''
# classification heuristic: based on value patterns (placeholder)
qtype = 'unknown'
if isinstance(data, str) and len(data) > 0:
if data.isalnum():
qtype = 'alphanumeric'
else:
qtype = 'other'
items.append({"bbox": bbox, "value": data if data else "", "type": qtype})
decode_results[image_id] = items
return decode_results




def main():
args = parse_args()
os.makedirs(os.path.dirname(args.output), exist_ok=True)


det_results = run_detection(args.weights, args.input, args.conf, args.iou)
write_detection_submission(det_results, args.output)
print(f"Wrote detection submission to {args.output}")


if args.decode:
os.makedirs(os.path.dirname(args.dec_output), exist_ok=True)
dec_results = run_decoding(det_results, args.input)
write_decoding_submission(dec_results, args.dec_output)
print(f"Wrote decoding submission to {args.dec_output}")


if __name__ == '__main__':
main()
