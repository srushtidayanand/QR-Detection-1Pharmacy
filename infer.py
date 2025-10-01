import os
import json
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("outputs/yolov8_qr.pt")  # trained weights

# Input folder & output JSON
input_folder = "data/test_images"
output_json = "outputs/submission_detection_1.json"

results = []

for img_name in os.listdir(input_folder):
    if img_name.lower().endswith((".jpg",".png",".jpeg")):
        img_path = os.path.join(input_folder, img_name)
        detections = model.predict(img_path, imgsz=640, conf=0.3, verbose=False)

        qrs = []
        for det in detections[0].boxes.xyxy:
            x_min, y_min, x_max, y_max = det.cpu().numpy().astype(int)
            qrs.append({"bbox": [x_min, y_min, x_max, y_max]})

        results.append({"image_id": img_name, "qrs": qrs})

# Save JSON
with open(output_json, "w") as f:
    json.dump(results, f, indent=4)

print(f"Detection JSON saved to {output_json}")
