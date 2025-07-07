# src/generate_detection_labels.py
import os
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed_root',
        default='data/processed/fruits360',
        help='Root of processed splits (Training, Validation, Test)')
    args = p.parse_args()

    # get sorted class names from the Training folder
    train_dir = os.path.join(args.processed_root, 'Training')
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    class_to_id = {c:i for i,c in enumerate(classes)}

    for split in ['Training','Validation','Test']:
        split_root = os.path.join(args.processed_root, split)
        for cls in classes:
            img_dir = os.path.join(split_root, cls)
            if not os.path.isdir(img_dir): continue
            for fname in os.listdir(img_dir):
                if not fname.lower().endswith(('.jpg','.png')): continue
                base, _ = os.path.splitext(fname)
                label_dir = img_dir  # labels next to images
                os.makedirs(label_dir, exist_ok=True)
                label_path = os.path.join(label_dir, f'{base}.txt')
                # full-frame box: x_center=0.5, y_center=0.5, w=1.0, h=1.0
                with open(label_path, 'w') as f:
                    f.write(f"{class_to_id[cls]} 0.5 0.5 1.0 1.0\n")

    print("Done generating detection labels for all splits.")

if __name__ == '__main__':
    main()
