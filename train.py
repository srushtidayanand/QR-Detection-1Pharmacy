from ultralytics import YOLO

# Initialize YOLOv8 model (lightweight)
model = YOLO("yolov8n.pt")  

# Train
model.train(
    data="data/data.yaml",   # YAML path
    epochs=50,
    imgsz=640,
    batch=16,
    augment=True
)

# Save trained weights
model.save("outputs/yolov8_qr.pt")
