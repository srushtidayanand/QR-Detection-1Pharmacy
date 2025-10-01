import os
import json
import cv2
import argparse
from ultralytics import YOLO

# -----------------------
# 0️⃣ Arguments
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="data/test_images", help="Folder of test images")
parser.add_argument("--output", type=str, default="outputs/submission_detection_1.json", help="Output JSON file")
parser.add_argument("--decode", action="store_true", help="Also decode QR codes")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

# -----------------------
# 1️⃣ Load model
# -----------------------
model_path = "runs/detect/qr_detection/weights/best.pt"  # trained weights
model = YOLO(model_path)

# -----------------------
# 2️⃣ Run inference
# -----------------------
results_list = []

for img_name in os.listdir(args.input):
    if img_name.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(args.input, img_name)
        result = model.predict(img_path, save=False)[0]

        qrs = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box)
                qrs.append({"bbox": [x_min, y_min, x_max, y_max]})

        results_list.append({
            "image_id": img_name,
            "qrs": qrs
        })

# -----------------------
# 3️⃣ Optional: decode QR codes
# -----------------------
if args.decode:
    for item in results_list:
        img_path = os.path.join(args.input, item["image_id"])
        img = cv2.imread(img_path)
        qr_decoder = cv2.QRCodeDetector()

        for q in item["qrs"]:
            x_min, y_min, x_max, y_max = q["bbox"]
            roi = img[y_min:y_max, x_min:x_max]
            val, _, _ = qr_decoder.detectAndDecode(roi)
            q["value"] = val

# -----------------------
# 4️⃣ Save JSON
# -----------------------
with open(args.output, "w") as f:
    json.dump(results_list, f, indent=4)

print(f"✅ Detection JSON saved at {args.output}")
