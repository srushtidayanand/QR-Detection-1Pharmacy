# QR-Detection-1Pharmacy
This project detects multiple QR codes on medicine packs, handling challenges like tilt, blur, and occlusion. Advanced modules decode and classify QR contents, ensuring robust and efficient performance.



---

## ⚙️ Setup


## ⚙️ Setup Instructions

 
✅ Step 1 — Clone the Repository
```bash
git clone https://github.com/srushtidayanand/QR-Detection-1Pharmacy.git
cd QR-Detection-1Pharmacy


✅ Step 2 — Install Dependencies

pip install -r requirements.txt
**Extra libraries for QR decoding:
**
apt-get update -y

apt-get install -y libzbar0
**Additional Python packages:
**
pip install ultralytics opencv-python-headless pillow pyzbar tqdm numpy pandas

**Run Inference (one line):
**
python src/infer.py --weights weights/best.pt --input demo_images/ --output outputs/submission_detection_1.json


**Train from Scratch (one line):
**
yolo detect train data=your_dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
