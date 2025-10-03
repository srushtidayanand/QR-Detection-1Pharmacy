#!/usr/bin/env python3
import os, shutil, glob, re, argparse, random, json
from PIL import Image

def parse_numbers(s):
    return [float(x) for x in re.split('[,\\s]+', s.strip()) if x!='']

def convert_label_to_yolo(parts, img_w, img_h):
    # parts: list of floats (various formats handled)
    # return x_center_rel,y_center_rel,w_rel,h_rel
    if len(parts) == 5:
        # Maybe: cls x_center y_center w h  (normalized) OR cls x_min y_min x_max y_max (pixels)
        cls = int(parts[0])
        rest = parts[1:]
        if max(rest) <= 1.0:
            x_center,y_center,w_rel,h_rel = rest
            return x_center,y_center,w_rel,h_rel
        else:
            # assume pixel bbox: x_min y_min x_max y_max and ignore cls (if provided)
            x_min,y_min,x_max,y_max = rest
    elif len(parts) == 4:
        x_min,y_min,x_max,y_max = parts
    elif len(parts) > 5:
        # polygon: take bounding rectangle from pairs
        xs = parts[0::2]
        ys = parts[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
    else:
        # unknown format
        return None
    # convert pixels->relative center/wh
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    w_rel = (x_max - x_min) / img_w
    h_rel = (y_max - y_min) / img_h
    return x_center, y_center, w_rel, h_rel

def prepare(input_dir, output_dir, val_ratio=0.1, seed=42):
    random.seed(seed)
    train_images_dir = os.path.join(input_dir, 'train_images')
    train_labels_dir = os.path.join(input_dir, 'train_labels')
    test_images_dir = os.path.join(input_dir, 'test_images')

    assert os.path.isdir(train_images_dir), f"Missing {train_images_dir}"
    assert os.path.isdir(train_labels_dir), f"Missing {train_labels_dir}"

    out_images = os.path.join(output_dir, 'images')
    out_labels = os.path.join(output_dir, 'labels')
    for sub in ['train','val','test']:
        os.makedirs(os.path.join(out_images, sub), exist_ok=True)
        os.makedirs(os.path.join(out_labels, sub), exist_ok=True)

    # gather train images
    train_imgs = sorted(glob.glob(os.path.join(train_images_dir, '*')))
    # split val
    n_val = max(1, int(len(train_imgs) * val_ratio))
    random.shuffle(train_imgs)
    val_imgs = set(train_imgs[:n_val])
    train_imgs_set = set(train_imgs[n_val:])

    def process_image_list(img_list, split):
        for img_path in img_list:
            basename = os.path.basename(img_path)
            name, ext = os.path.splitext(basename)
            dest_img = os.path.join(out_images, split, basename)
            shutil.copyfile(img_path, dest_img)
            # read label if exists
            src_label = os.path.join(train_labels_dir, f'{name}.txt')
            dest_label = os.path.join(out_labels, split, f'{name}.txt')
            labels_out = []
            if os.path.exists(src_label):
                with open(src_label, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                im = Image.open(img_path)
                img_w, img_h = im.size
                for l in lines:
                    parts = parse_numbers(l)
                    if not parts:
                        continue
                    converted = convert_label_to_yolo(parts, img_w, img_h)
                    if converted is None:
                        continue
                    x_center,y_center,w_rel,h_rel = converted
                    # clamp
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    w_rel = max(0.0, min(1.0, w_rel))
                    h_rel = max(0.0, min(1.0, h_rel))
                    # class 0 (single class 'qr')
                    labels_out.append(f"0 {x_center:.6f} {y_center:.6f} {w_rel:.6f} {h_rel:.6f}")
            # write (possibly empty) label file
            with open(dest_label, 'w') as fo:
                if labels_out:
                    fo.write('\n'.join(labels_out))
                else:
                    fo.write('')  # empty file for no-object images

    process_image_list([p for p in train_imgs if p in train_imgs_set], 'train')
    process_image_list([p for p in train_imgs if p in val_imgs], 'val')

    # if test_images exist, copy them to images/test (labels left blank)
    if os.path.isdir(test_images_dir):
        test_imgs = sorted(glob.glob(os.path.join(test_images_dir, '*')))
        for img_path in test_imgs:
            basename = os.path.basename(img_path)
            name, ext = os.path.splitext(basename)
            shutil.copyfile(img_path, os.path.join(out_images, 'test', basename))
            open(os.path.join(out_labels, 'test', f'{name}.txt'), 'w').close()

    # write data.yaml
    data_yaml = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test' if os.path.isdir(test_images_dir) else None,
        'nc': 1,
        'names': ['qr']
    }
    # Save yaml as plain text
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        if data_yaml['test']:
            f.write(f"test: {data_yaml['test']}\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write("names: ['qr']\n")
    print("Prepared dataset at", output_dir)
    print("Data yaml saved at", yaml_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/content/drive/MyDrive/QR_Dataset', help='path to folder containing train_images & train_labels')
    parser.add_argument('--output_dir', type=str, default='data/QRDataset', help='output folder for YOLO formatted dataset')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    args = parser.parse_args()
    prepare(args.input_dir, args.output_dir, val_ratio=args.val_ratio)
