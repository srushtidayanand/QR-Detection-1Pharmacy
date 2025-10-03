import os, argparse, json, cv2
from ultralytics import YOLO
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image

def decode_crop(img):
    # img: numpy image (BGR)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dec = decode(pil)
    if dec:
        # return first decoded string
        return dec[0].data.decode('utf-8'), dec[0].type
    return None, None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='runs/train/qr_yolov8n/weights/best.pt')
    p.add_argument('--source', type=str, default='data/QRDataset/images/test') # folder of images
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--output', type=str, default='outputs/submission_detection_1.json')
    p.add_argument('--decode', action='store_true', help='also try to decode QR contents (pyzbar)')
    p.add_argument('--save_annotated', action='store_true', help='save annotated images')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    ann_dir = os.path.join("outputs", "annotated")
    if args.save_annotated:
        os.makedirs(ann_dir, exist_ok=True)

    model = YOLO(args.weights)

    images = sorted([os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    results_list = []

    for img_path in images:
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        res = model.predict(source=img_path, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
        
        detections = res[0].boxes if len(res) > 0 else None
        qrs = []
        if detections is not None and len(detections) > 0:
            boxes = detections.xyxy.cpu().numpy()
            for bbox in boxes:
                x1,y1,x2,y2 = [float(x) for x in bbox[:4]]
                entry = {"bbox":[x1,y1,x2,y2]}
                if args.decode:
                    h,w = img.shape[:2]
                    xa = int(max(0, round(x1)))
                    ya = int(max(0, round(y1)))
                    xb = int(min(w-1, round(x2)))
                    yb = int(min(h-1, round(y2)))
                    if xb>xa and yb>ya:
                        crop = img[ya:yb, xa:xb]
                        val, typ = decode_crop(crop)
                        entry["value"] = val if val else None
                        entry["type"] = typ if typ else "unknown"
                qrs.append(entry)

                # Draw annotation if enabled
                if args.save_annotated:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    if args.decode and entry.get("value"):
                        cv2.putText(img, entry["value"], (int(x1), int(y1)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        results_list.append({"image_id": image_id, "qrs": qrs})

        # Save annotated image
        if args.save_annotated:
            save_path = os.path.join(ann_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, img)

    # save detection JSON
    with open(args.output, 'w') as f:
        json.dump(results_list, f, indent=2)
    print("Saved detection JSON to", args.output)

    if args.decode:
        dec_out = args.output.replace('.json','_decoding.json')
        with open(dec_out, 'w') as f:
            json.dump(results_list, f, indent=2)
        print("Saved decoding JSON to", dec_out)

if __name__ == '__main__':
    main()

