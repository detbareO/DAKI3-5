from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_TRANSFORM_DESCRIPTION = (
    "bbox crop (clamped to the image, full image if the box is degenerate) -> "
    "Resize((224, 224)) -> ToTensor -> Normalize(ImageNet mean and std)"
)
VIEWPOINT_NAMES = {-1: "unknown", 1: "front", 2: "rear", 3: "side", 4: "front-side", 5: "rear-side"}


def load_index(path):
    return pd.read_csv(path)


def load_split(path):
    return pd.read_csv(path)


def crop_box(image, x1, y1, x2, y2):
    width, height = image.size
    # hold boksen inden for billedet
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    # nogle få bokse har ingen areal, så tag hele billedet
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


def load_crop(data_root, path, box):
    with Image.open(Path(data_root) / "image" / path) as image:
        image = image.convert("RGB")
    return crop_box(image, *box)


def eval_resize():
    from torchvision import transforms

    # ingen center crop, det skærer front og bagende af bilen
    return transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))


def eval_transform():
    from torchvision import transforms

    return transforms.Compose([
        eval_resize(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CarDataset:
    def __init__(self, index, rows, data_root, transform):
        self.rows = np.asarray(rows)
        self.paths = index["path"].to_numpy()
        self.boxes = index[["x1", "y1", "x2", "y2"]].to_numpy()
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = int(self.rows[i])
        image = load_crop(self.data_root, self.paths[row], self.boxes[row])
        return self.transform(image), row
