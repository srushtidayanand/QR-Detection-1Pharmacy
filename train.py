from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # lightweight pretrained model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    augment=True,
    project="outputs",
    name="qr_detection",
)
