# evaluate.py

import random
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader, Subset

from src.dataset import OilSpillDataset


# =========================================================
# CONFIG
# =========================================================

DATASET_DIR = "./processed"

CHECKPOINT_PATH = "./checkpoints/best_model.pth"

NUM_IMAGES = 100

BATCH_SIZE = 8

THRESHOLD = 0.5

SEED = 42


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)

print(device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================================================
# MODEL
# =========================================================

def create_model():

    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        activation=None,
    )

    return model


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading model...")

model = create_model()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint.get('epoch', 'unknown')}"
)


# =========================================================
# DATASET
# =========================================================

print("\nLoading dataset...")

dataset = OilSpillDataset(
    root_dir=DATASET_DIR,
    transform=None
)

total_samples = len(dataset)

print(
    f"Total dataset samples: "
    f"{total_samples}"
)


# =========================================================
# SELECT 100 RANDOM IMAGES
# =========================================================

random.seed(SEED)

num_samples = min(
    NUM_IMAGES,
    total_samples
)

indices = random.sample(
    range(total_samples),
    num_samples
)

evaluation_dataset = Subset(
    dataset,
    indices
)

evaluation_loader = DataLoader(
    evaluation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


print(
    f"Evaluating {num_samples} images"
)


# =========================================================
# LOSS
# =========================================================

bce_loss = torch.nn.BCEWithLogitsLoss()


def dice_loss(logits, targets):

    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        +
        targets.sum(dim=(1, 2, 3))
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    return 1.0 - dice.mean()


def combined_loss(logits, targets):

    return (
        bce_loss(logits, targets)
        +
        dice_loss(logits, targets)
    )


# =========================================================
# METRIC ACCUMULATORS
# =========================================================

total_tp = 0
total_tn = 0
total_fp = 0
total_fn = 0

total_loss = 0.0

num_batches = 0


# =========================================================
# EVALUATION
# =========================================================

print("\nEvaluating...")

with torch.no_grad():

    for batch_idx, (images, masks) in enumerate(
        evaluation_loader
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        # -------------------------------------------------
        # Model prediction
        # -------------------------------------------------

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        loss = combined_loss(
            logits,
            masks
        )

        total_loss += loss.item()

        num_batches += 1

        # -------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------

        tp = (
            (predictions == 1)
            & (masks == 1)
        ).sum().item()

        tn = (
            (predictions == 0)
            & (masks == 0)
        ).sum().item()

        fp = (
            (predictions == 1)
            & (masks == 0)
        ).sum().item()

        fn = (
            (predictions == 0)
            & (masks == 1)
        ).sum().item()

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        print(
            f"Batch "
            f"{batch_idx + 1}/"
            f"{len(evaluation_loader)} "
            f"| loss={loss.item():.4f}"
        )


# =========================================================
# CALCULATE METRICS
# =========================================================

tp = total_tp
tn = total_tn
fp = total_fp
fn = total_fn

eps = 1e-8


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

accuracy = (
    (tp + tn)
    /
    (tp + tn + fp + fn + eps)
)


# ---------------------------------------------------------
# Precision
# ---------------------------------------------------------

precision = (
    tp
    /
    (tp + fp + eps)
)


# ---------------------------------------------------------
# Recall / Sensitivity
# ---------------------------------------------------------

recall = (
    tp
    /
    (tp + fn + eps)
)


# ---------------------------------------------------------
# F1 / Dice
# ---------------------------------------------------------

f1 = (
    2 * precision * recall
    /
    (precision + recall + eps)
)


# ---------------------------------------------------------
# IoU
# ---------------------------------------------------------

iou = (
    tp
    /
    (tp + fp + fn + eps)
)


# ---------------------------------------------------------
# Specificity
# ---------------------------------------------------------

specificity = (
    tn
    /
    (tn + fp + eps)
)


# ---------------------------------------------------------
# False Positive Rate
# ---------------------------------------------------------

fpr = (
    fp
    /
    (fp + tn + eps)
)


# ---------------------------------------------------------
# False Negative Rate
# ---------------------------------------------------------

fnr = (
    fn
    /
    (fn + tp + eps)
)


# ---------------------------------------------------------
# Average loss
# ---------------------------------------------------------

average_loss = (
    total_loss
    /
    num_batches
)


# =========================================================
# PIXEL STATISTICS
# =========================================================

total_pixels = (
    tp + tn + fp + fn
)

actual_positive = tp + fn

predicted_positive = tp + fp

actual_negative = tn + fp

predicted_negative = tn + fn


actual_positive_ratio = (
    actual_positive
    /
    (total_pixels + eps)
)

predicted_positive_ratio = (
    predicted_positive
    /
    (total_pixels + eps)
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(
    f"Images evaluated      : "
    f"{num_samples}"
)

print(
    f"Threshold             : "
    f"{THRESHOLD}"
)

print(
    f"Average Loss          : "
    f"{average_loss:.6f}"
)

print("\n")
print("METRICS")
print("-" * 60)

print(
    f"Accuracy              : "
    f"{accuracy:.6f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision             : "
    f"{precision:.6f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall                : "
    f"{recall:.6f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1 Score / Dice       : "
    f"{f1:.6f} "
    f"({f1 * 100:.2f}%)"
)

print(
    f"IoU                   : "
    f"{iou:.6f} "
    f"({iou * 100:.2f}%)"
)

print(
    f"Specificity           : "
    f"{specificity:.6f} "
    f"({specificity * 100:.2f}%)"
)

print(
    f"False Positive Rate   : "
    f"{fpr:.6f} "
    f"({fpr * 100:.2f}%)"
)

print(
    f"False Negative Rate   : "
    f"{fnr:.6f} "
    f"({fnr * 100:.2f}%)"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n")
print("CONFUSION MATRIX")
print("-" * 60)

print(
    f"True Positive (TP)   : "
    f"{tp:,}"
)

print(
    f"True Negative (TN)   : "
    f"{tn:,}"
)

print(
    f"False Positive (FP)  : "
    f"{fp:,}"
)

print(
    f"False Negative (FN)  : "
    f"{fn:,}"
)


# =========================================================
# PIXEL DISTRIBUTION
# =========================================================

print("\n")
print("PIXEL DISTRIBUTION")
print("-" * 60)

print(
    f"Total pixels         : "
    f"{total_pixels:,}"
)

print(
    f"Actual oil pixels    : "
    f"{actual_positive:,}"
)

print(
    f"Actual background    : "
    f"{actual_negative:,}"
)

print(
    f"Predicted oil pixels : "
    f"{predicted_positive:,}"
)

print(
    f"Predicted background : "
    f"{predicted_negative:,}"
)

print(
    f"Actual oil ratio     : "
    f"{actual_positive_ratio:.6%}"
)

print(
    f"Predicted oil ratio  : "
    f"{predicted_positive_ratio:.6%}"
)


print("\n")
print("=" * 60)
print("DONE")
print("=" * 60)