import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import albumentations as A

from src.instance import Instance
from src.common import *


# =========================================================
# CONFIGURATION
# =========================================================

OUTPUT_DIR = Path("./processed")

BAND0_OUT_DIR = OUTPUT_DIR / "band0"
BAND1_OUT_DIR = OUTPUT_DIR / "band1"
MASK_OUT_DIR = OUTPUT_DIR / "masks"

PREVIEW_DIR = Path("./preview")

PATCHES_PER_IMAGE = 10
MAX_ATTEMPTS = 20
BLANK_ACCEPT_PROBABILITY = 0.3

# Number of worker processes.
# Usually start with CPU count - 1 or CPU count.
NUM_WORKERS = max(1, os.cpu_count() - 4)


# =========================================================
# AUGMENTATION
# =========================================================

def create_transform():

    return A.Compose([

        A.RandomCrop(
            height=512,
            width=512,
            p=1.0
        ),

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


# =========================================================
# PROCESS ONE IMAGE
# =========================================================

def process_image(args):

    image_index, image_name, mask_name = args

    # Create directories inside the worker if necessary.
    # This makes the worker self-contained.
    BAND0_OUT_DIR.mkdir(parents=True, exist_ok=True)
    BAND1_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Give every worker its own RNG
    # -----------------------------------------------------

    seed = (
        os.getpid()
        + image_index * 100003
        + random.randint(0, 1_000_000)
    )

    random.seed(seed)
    np.random.seed(seed)

    transform = create_transform()

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    instance = Instance()

    instance.read_from_paths([
        BAND_DIR / image_name,
        MASK_DIR / mask_name
    ])

    # -----------------------------------------------------
    # Create H,W,2 image
    # -----------------------------------------------------

    original_image = np.stack(
        (
            instance.band0,
            instance.band1
        ),
        axis=-1
    ).astype(np.float32)

    original_mask = instance.mask0.astype(np.uint8)

    # -----------------------------------------------------
    # Generate patches
    # -----------------------------------------------------

    accepted = 0
    attempts = 0

    results = []

    while accepted < PATCHES_PER_IMAGE:

        attempts += 1

        if attempts > PATCHES_PER_IMAGE * MAX_ATTEMPTS:

            break

        # -------------------------------------------------
        # Augmentation
        # -------------------------------------------------

        augmented = transform(
            image=original_image,
            mask=original_mask
        )

        image_aug = augmented["image"]
        mask_aug = augmented["mask"]

        # -------------------------------------------------
        # Check mask
        # -------------------------------------------------

        has_oil = np.any(mask_aug > 0)

        if has_oil:

            accept = True

        else:

            accept = (
                random.random()
                < BLANK_ACCEPT_PROBABILITY
            )

        # -------------------------------------------------
        # Reject
        # -------------------------------------------------

        if not accept:
            continue

        # -------------------------------------------------
        # Accept
        # -------------------------------------------------

        accepted += 1

        patch_name = (
            f"{Path(image_name).stem}"
            f"_patch_{accepted:02d}"
        )

        band0_patch = image_aug[:, :, 0]
        band1_patch = image_aug[:, :, 1]

        # -------------------------------------------------
        # Save arrays
        # -------------------------------------------------

        np.save(
            BAND0_OUT_DIR / f"{patch_name}.npy",
            band0_patch
        )

        np.save(
            BAND1_OUT_DIR / f"{patch_name}.npy",
            band1_patch
        )

        np.save(
            MASK_OUT_DIR / f"{patch_name}.npy",
            mask_aug
        )

        results.append({
            "patch": patch_name,
            "oil_pixels": int(np.sum(mask_aug > 0)),
            "attempts": attempts,
        })

    return {
        "image_index": image_index,
        "image": image_name,
        "accepted": accepted,
        "attempts": attempts,
        "results": results,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    BAND0_OUT_DIR.mkdir(parents=True, exist_ok=True)
    BAND1_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Get files
    # -----------------------------------------------------

    instances = sorted(os.listdir(BAND_DIR))
    masks = sorted(os.listdir(MASK_DIR))

    if len(instances) != len(masks):

        raise RuntimeError(
            f"Number of images ({len(instances)}) "
            f"does not match number of masks ({len(masks)})"
        )

    print(f"Images : {len(instances)}")
    print(f"Masks  : {len(masks)}")
    print(f"Workers: {NUM_WORKERS}")

    # -----------------------------------------------------
    # Create jobs
    # -----------------------------------------------------

    jobs = [
        (i, instances[i], masks[i])
        for i in range(len(instances))
    ]

    # -----------------------------------------------------
    # Multiprocessing
    # -----------------------------------------------------

    completed = 0

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:

        futures = [
            executor.submit(process_image, job)
            for job in jobs
        ]

        for future in as_completed(futures):

            result = future.result()

            completed += 1

            print(
                f"[{completed}/{len(jobs)}] "
                f"{result['image']} → "
                f"{result['accepted']}/{PATCHES_PER_IMAGE} "
                f"patches "
                f"({result['attempts']} attempts)"
            )


    print("\nFinished generating patches.")


# =========================================================
# IMPORTANT FOR MULTIPROCESSING
# =========================================================

if __name__ == "__main__":
    main()