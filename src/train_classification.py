# src/train_classification.py
import os
import argparse
import torch
from ultralytics import YOLO

# Allow duplicate OpenMP (only if you still see that warning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Enable cuDNN autotuner for best performance on fixed-size inputs
torch.backends.cudnn.benchmark = True

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',   type=str, default='data/fruits360-cls',
                   help='Path to root of classification dataset (train/ & val/)')
    p.add_argument('--model',  type=str,   default='yolov8n-cls.pt')
    p.add_argument('--epochs', type=int,   default=30)
    p.add_argument('--imgsz',  type=int,   default=256,
                   help='Input image size (pixels)')
    p.add_argument('--batch',  type=int,   default=64,
                   help='Batch size (increase until you hit VRAM limit)')
    p.add_argument('--device', type=str,   default='0',
                   help='CUDA device to use (e.g. "0" or "0,1")')
    p.add_argument('--half',   action='store_true',
                   help='Use mixed-precision training')
    return p.parse_args()

def main():
    args = parse_args()

    # Print out GPU info
    if torch.cuda.is_available():
        print(f"Using CUDA device(s): {args.device}")
        print("Total VRAM:", torch.cuda.get_device_properties(int(args.device.split(',')[0])).total_memory/1e9, "GB")

    # Load a YOLOv8 classification model
    model = YOLO(args.model)

    # Train with maximum GPU utilization
    model.train(
        data    = args.data,
        task    = 'classify',
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        half    = args.half,
        name    = 'fruits360-cls-gpu',
        project = 'runs/classify'
    )

if __name__ == "__main__":
    main()
