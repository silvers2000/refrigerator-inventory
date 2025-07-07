# src/data_prep.py
import argparse, os, cv2
from albumentations import Compose, Resize, Normalize, RandomRotate90

def get_transforms():
    return Compose([
        Resize(256, 256),
        RandomRotate90(),
    ])

def preprocess_folder(src_dir, dst_dir):
    transforms = get_transforms()
    for root, dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        out_dir = os.path.join(dst_dir, rel) if rel!='.' else dst_dir
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            if f.lower().endswith(('.jpg','.png')):
                img = cv2.imread(os.path.join(root, f))
                out = transforms(image=img)['image']
                cv2.imwrite(os.path.join(out_dir, f), out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    preprocess_folder(args.input, args.output)

if __name__=='__main__':
    main()
