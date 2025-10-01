from ultralytics import YOLO
import cv2
import os
import json

model = YOLO("outputs/qr_detection/weights/best.pt")

image_folder = "data/test_images"
output_json = "outputs/submission_detection_1.json"

results_list = []

for img_name in os.listdir(image_folder):
    if img_name.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(image_folder, img_name)
        results = model.predict(img_path, save=False)
        qrs = []
        for box in results[0].boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, box)
            qrs.append({"bbox": [x_min, y_min, x_max, y_max]})
        results_list.append({"image_id": img_name, "qrs": qrs})

with open(output_json, "w") as f:
    json.dump(results_list, f, indent=4)

print(f"Detection results saved to {output_json}")
