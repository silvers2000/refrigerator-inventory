# Smart Refrigerator Fruit Detection & Spoilage Classification

## Overview
This project builds a two-stage computer vision pipeline to  
1. **Detect** multiple fruit types in a refrigerator scene  
2. **Classify** each detected fruit as fresh or spoiled  

We leverage Ultralytics YOLOv8 for both detection and classification tasks, using a combination of public datasets.

---

## 1. Datasets & Training

### 1.1 Fruits360 (Classification pre-training)
- **What**: 131,000+ images of 120 fruit classes, single-fruit centered  
- **Why**: Warm-start our classification backbone on clean examples  
- **How**:  
  ```bash
  yolo classify train     data=fruits360.yaml     model=yolov8n-cls.pt     epochs=50     imgsz=224     batch=64     name=fruits360-cls
  ```

* **Result**:

  * Top-1 accuracy: ≈ 98% on held-out split  
  * Saved weights: `runs/classify/fruits360-cls/weights/best.pt`

### 1.2 Open Images V7 – Fruit Subset (Detection & Fine-tuning)

* **What**: “Open Images V7” dataset filtered to 17 fruit classes  
* **Why**: Teach YOLO to locate & distinguish multiple fruits in realistic fridge scenes  
* **How**:

  1. **Export to YOLOv5 format** via FiftyOne  
     ```python
     import fiftyone.zoo as foz
     from fiftyone.types import YOLOv5Dataset

     FRUIT_CLASSES = [
       "Apple","Banana","Cantaloupe","Coconut","Grape","Grapefruit",
       "Lemon","Mango","Orange","Peach","Pear","Pineapple",
       "Pomegranate","Strawberry","Tomato","Watermelon","Winter melon"
     ]

     ds = foz.load_zoo_dataset(
       "open-images-v7",
       splits="train,validation",
       label_types=["detections"],
       classes=FRUIT_CLASSES,
     )
     ds.export(
       export_dir="data/open_images_fruits",
       dataset_type=YOLOv5Dataset,
       label_field="ground_truth",
       classes=FRUIT_CLASSES,
       overwrite=True,
     )
     ```
  2. **Train YOLOv8 detection**  
     ```bash
     yolo detect train        data=open_images_fruits.yaml        model=yolov8n.pt        epochs=30        imgsz=640        batch=16        device=0        half=True        name=stage2-openimages-fruits
     ```

* **Validation Performance** (520 held-out images):

  | Metric    | mAP₅₀ | mAP₅₀₋₉₅ |
  | --------- | ----- | -------- |
  | Detection | 0.475 | 0.386    |

* **Saved weights**:

  * `runs/detect/stage2-openimages-fruits11/weights/best.pt`  
  * `runs/detect/stage2-openimages-fruits11/weights/last.pt`

---

# 2. Results

## Validation Batches

### Batch 0  
![val_batch0_labels](runs/detect/stage2-openimages-fruits11/val_batch0_labels.png)  
![val_batch0_pred](runs/detect/stage2-openimages-fruits11/val_batch0_pred.png)

### Batch 1  
![val_batch1_labels](runs/detect/stage2-openimages-fruits11/val_batch1_labels.png)  
![val_batch1_pred](runs/detect/stage2-openimages-fruits11/val_batch1_pred.png)

### Batch 2  
![val_batch2_labels](runs/detect/stage2-openimages-fruits11/val_batch2_labels.png)  
![val_batch2_pred](runs/detect/stage2-openimages-fruits11/val_batch2_pred.png)


---

# 3. Next Steps & Roadmap

1. **Spoilage Classification**

   * Collect & annotate a fresh vs. spoiled fruit dataset  
   * Fine-tune the Fruits360-pretrained classifier on this binary task

2. **End-to-End Integration**

   * Run detection → crop each box → run spoilage classifier  
   * Overlay “fresh” / “spoiled” labels on fridge images

3. **User Interface & Deployment**

   * Prototype a web/mobile dashboard for fridge inventory & spoilage alerts  
   * Package model for edge devices (e.g. Raspberry Pi / Jetson Nano)

4. **Dataset Expansion & Robustness**

   * Augment with more fridge scenes & varied lighting  
   * Add “mixed” classes (e.g. fruit salads, cut fruit)

---
