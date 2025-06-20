import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on Fruits360")
    p.add_argument('--data',   type=str, default="data/fruits360.yaml", help="Path to dataset YAML")
    p.add_argument('--model',  type=str, default="yolov8n.pt",         help="Pretrained YOLOv8 model")
    p.add_argument('--epochs', type=int, default=50,                   help="Number of epochs")
    p.add_argument('--imgsz',  type=int, default=256,                  help="Input image size (pixels)")
    p.add_argument('--batch',  type=int, default=16,                   help="Batch size")
    p.add_argument('--project',type=str, default="runs/train",         help="Save runs/project name")
    p.add_argument('--name',   type=str, default="fruits360",          help="Name of this run")
    return p.parse_args()

def main():
    args = parse_args()
    # Initialize a YOLOv8 model (nano by default)
    model = YOLO(args.model)
    # Train!
    model.train(
        data    = args.data,
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        project = args.project,
        name    = args.name
    )

if __name__ == "__main__":
    main()
