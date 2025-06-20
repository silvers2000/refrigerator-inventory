# src/train_classification.py
import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',   type=str, default='data/fruits360-cls.yaml')
    p.add_argument('--model',  type=str, default='yolov8n-cls.pt')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--imgsz',  type=int, default=256)
    p.add_argument('--batch',  type=int, default=32)
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)  # classification model
    model.train(
        data   = args.data,
        task   = 'classify',
        epochs = args.epochs,
        imgsz  = args.imgsz,
        batch  = args.batch,
        name   = 'fruits360-cls',
        project= 'runs/classify'
    )

if __name__ == "__main__":
    main()
