# QR-Detection-1Pharmacy
This project detects multiple QR codes on medicine packs, handling challenges like tilt, blur, and occlusion. Advanced modules decode and classify QR contents, ensuring robust and efficient performance.



---

## ⚙️ Setup
✅ Cell 1 — Clone the repo
!git clone https://github.com/srushtidayanand/QR-Detection-1Pharmacy.git
%cd QR-Detection-1Pharmacy

✅ Cell 2 — Install dependencies
!pip install -r requirements.txt

# Extra libs for QR decoding
!apt-get update -y
!apt-get install -y libzbar0

# Additional Python packages
!pip install ultralytics opencv-python-headless pillow pyzbar tqdm numpy pandas

✅ Cell 3 — Run Inference with trained weights
!python src/infer.py \
    --weights weights/best.pt \
    --input demo_images/ \
    --output outputs/submission_detection_1.json

✅ Cell 4 — Train from scratch (if dataset is available)
!yolo detect train \
    data=your_dataset.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=640



<img width="1345" height="615" alt="Screenshot 2025-10-03 130507" src="https://github.com/user-attachments/assets/a64bd27b-2c2e-4a11-87ba-9cfa9eb45ec8" />
output for test dataset : annotation value(type)
<img width="717" height="714" alt="image" src="https://github.com/user-attachments/assets/a53a4b5c-f68c-4124-946b-eaddb236687b" />
