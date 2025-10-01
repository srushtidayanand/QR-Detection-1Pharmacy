import os
import random
import shutil
from ultralytics import YOLO

# -----------------------
# 0️⃣ Set directories
# -----------------------
WORK_DIR = "/content/drive/MyDrive/QR-Detection-1Pharmacy"
DATA_DIR = os.path.join(WORK_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
VAL_DIR = os.path.join(DATA_DIR, "val_images")

os.makedirs(VAL_DIR, exist_ok=True)

# -----------------------
# 1️⃣ Split 10% of training for validation
# -----------------------
all_images = [f for f in os.listdir(TRAIN_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(all_images)
val_split = 0.1
num_val = int(len(all_images) * val_split)

for img in all_images[:num_val]:
    shutil.move(os.path.join(TRAIN_DIR, img), os.path.join(VAL_DIR, img))

print(f"Training images: {len(os.listdir(TRAIN_DIR))}")
print(f"Validation images: {len(os.listdir(VAL_DIR))}")

# -----------------------
# 2️⃣ Create data.yaml
# -----------------------
data_yaml_path = os.path.join(WORK_DIR, "data.yaml")
data_yaml = f"""
train: {TRAIN_DIR}
val: {VAL_DIR}
nc: 1
names: ['qr']
"""

with open(data_yaml_path, "w") as f:
    f.write(data_yaml)

# -----------------------
# 3️⃣ Train YOLOv8
# -----------------------
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n

model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    augment=True,
    name="qr_detection",
    save_dir=os.path.join(WORK_DIR, "runs/detect")
)

print("✅ Training completed. Weights saved in runs/detect/qr_detection/weights/")
