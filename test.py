import fiftyone.zoo as foz
from fiftyone.types import YOLOv5Dataset

FRUIT_CLASSES = [
    "Apple", "Banana", "Cantaloupe", "Coconut", "Grape",
    "Grapefruit", "Lemon", "Mango", "Orange", "Peach",
    "Pear", "Pineapple", "Pomegranate", "Strawberry",
    "Tomato", "Watermelon", "Winter melon",
]

# ——— TRAIN ———
train_ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=FRUIT_CLASSES,
)

train_ds.export(
    export_dir="data/open_images_fruits/train",
    dataset_type=YOLOv5Dataset,
    label_field="ground_truth",
    classes=FRUIT_CLASSES,
    overwrite=True,
)

# ——— VALIDATION ———
val_ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=FRUIT_CLASSES,
)

val_ds.export(
    export_dir="data/open_images_fruits/val",
    dataset_type=YOLOv5Dataset,
    label_field="ground_truth",
    classes=FRUIT_CLASSES,
    overwrite=True,
)
