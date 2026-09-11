import albumentations as A

from src.common import *
from src.instance import Instance

OUTPUT_DIR = Path("./processed")
BAND_OUT_DIR = OUTPUT_DIR / "bands"
MASK_OUT_DIR = OUTPUT_DIR / "masks"
PREVIEW_DIR = Path("./preview")

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
BAND_OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)

transform = A.Compose([
    A.RandomCrop(height=512, width=512, p=1.0),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.05, 0.05),
        rotate=(-10, 10),
        p=0.3
    ),
])

