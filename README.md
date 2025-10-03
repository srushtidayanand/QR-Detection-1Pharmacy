# QR-Detection-1Pharmacy
This project detects multiple QR codes on medicine packs, handling challenges like tilt, blur, and occlusion. Advanced modules decode and classify QR contents, ensuring robust and efficient performance.



---

## ⚙️ Setup



 
✅ Step 1 — Clone the Repository

!git clone https://github.com/srushtidayanand/QR-Detection-1Pharmacy.git


%cd QR-Detection-1Pharmacy


✅ Step 2 — Install Dependencies

pip install -r requirements.txt

✅ Step 3 Extra libraries for QR decoding:

!apt-get update -y


!apt-get install -y libzbar0


!pip install ultralytics opencv-python-headless pillow pyzbar tqdm numpy pandas

✅ Step 4 Run Inference (one line):

uplaod your images file in Data/QRDataset/ folder under name of demo
example : <img width="367" height="232" alt="image" src="https://github.com/user-attachments/assets/3a01de88-2947-442d-a26e-6d974f620f4b" />


!python infer.py --weights weights/best.pt --source Data/QRDataset/demo --output outputs/submission_detection_1demo.json --decode --save_annotated


## TRAINING : 


<img width="1345" height="615" alt="Screenshot 2025-10-03 130507" src="https://github.com/user-attachments/assets/4afb766f-f5eb-40eb-905c-eed3af814338" />

## OUTPUT : 


<img width="717" height="714" alt="Screenshot 2025-10-03 204341" src="https://github.com/user-attachments/assets/62a71a9b-fcf9-412f-bea7-d7bfbb89ab2f" />


