# train.py


from ultralytics import YOLO
import argparse
import os


# Training script for YOLOv8 model (yolov8m) with augmentation good for small/tilted/blurred QRs


def parse_args():
p = argparse.ArgumentParser()
p.add_argument('--data', type=str, default='dataset.yaml', help='path to dataset yaml')
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--imgsz', type=int, default=960)
p.add_argument('--batch', type=int, default=8)
p.add_argument('--weights', type=str, default='yolov8m.pt', help='pretrained weights or path')
p.add_argument('--project', type=str, default='runs/train')
p.add_argument('--name', type=str, default='yolov8m_multiqr')
return p.parse_args()




def main():
args = parse_args()


# Create project dir
os.makedirs(args.project, exist_ok=True)


# Load model
model = YOLO(args.weights)


# Train with augmentations (Ultralytics integrates mosaic, mixup, hsv by default when 'augment=True')
model.train(
data=args.data,
epochs=args.epochs,
imgsz=args.imgsz,
batch=args.batch,
project=args.project,
name=args.name,
exist_ok=True,
augment=True, # basic augmentations
# You can fine tune hyperparameters in 'hyperparameters' dict if needed
)


if __name__ == '__main__':
main()
