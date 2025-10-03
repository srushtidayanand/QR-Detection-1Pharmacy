from ultralytics import YOLO
import argparse, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data/QRDataset/data.yaml')
    p.add_argument('--model', type=str, default='yolov8n.pt')  # use 'yolov8n.pt' or local weights
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--name', type=str, default='qr_yolov8n')
    args = p.parse_args()

    model = YOLO(args.model)
    print("Starting training with model:", args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name)
    print("Training finished. Check runs/train/ directory for weights.")

if __name__ == '__main__':
    main()
