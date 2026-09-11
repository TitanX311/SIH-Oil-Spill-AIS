import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from segmentation_models_pytorch import Unet

from src.dataset import OilSpillDataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "./processed"
CHECKPOINT_PATH = "./checkpoints/best_model.pth"

NUM_SAMPLES = 5
THRESHOLD = 0.5
SEED = 42

OUTPUT_DIR = Path("./prediction_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# DATASET
# ============================================================

dataset = OilSpillDataset(
    root_dir=DATASET_DIR,
    normalize=False,
    transform=None
)

print("Dataset size:", len(dataset))


# ============================================================
# SELECT SAMPLES
# ============================================================

random.seed(SEED)

num_samples = min(NUM_SAMPLES, len(dataset))

indices = random.sample(
    range(len(dataset)),
    num_samples
)

print("\nSelected samples:")

for index in indices:
    print(index, dataset.files[index].name)


# ============================================================
# CREATE MODEL
# ============================================================

model = Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=2,
    classes=1,
    activation=None,
)

model = model.to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint:")
print(CHECKPOINT_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)


# Handle different checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    print("Checkpoint type: full training checkpoint")

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if "epoch" in checkpoint:
        print("Epoch:", checkpoint["epoch"])

    if "train_loss" in checkpoint:
        print("Train loss:", checkpoint["train_loss"])

    if "val_loss" in checkpoint:
        print("Validation loss:", checkpoint["val_loss"])

elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

    print("Checkpoint type: state_dict checkpoint")

    model.load_state_dict(
        checkpoint["state_dict"]
    )

else:

    print("Checkpoint type: raw model state_dict")

    model.load_state_dict(checkpoint)


model.eval()

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    for plot_number, index in enumerate(indices, start=1):

        image, mask = dataset[index]

        # ----------------------------------------------------
        # image:
        # [2, H, W]
        #
        # mask:
        # [1, H, W]
        # ----------------------------------------------------

        print("\n" + "=" * 60)

        print("Sample:", plot_number)
        print("Index:", index)
        print("Filename:", dataset.files[index].name)

        print("Image shape:", tuple(image.shape))
        print("Mask shape:", tuple(mask.shape))

        # Add batch dimension
        input_tensor = image.unsqueeze(0).to(device)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        logits = model(input_tensor)

        probability = torch.sigmoid(logits)

        prediction = (
            probability >= THRESHOLD
        ).float()

        # Remove batch/channel dimensions
        probability = probability[0, 0].cpu().numpy()
        prediction = prediction[0, 0].cpu().numpy()

        ground_truth = mask[0].cpu().numpy()

        band0 = image[0].cpu().numpy()
        band1 = image[1].cpu().numpy()

        # ----------------------------------------------------
        # BINARY MASK
        # ----------------------------------------------------

        ground_truth = (ground_truth > 0.5).astype(np.uint8)
        prediction = prediction.astype(np.uint8)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        gt = ground_truth.astype(bool)
        pred = prediction.astype(bool)

        intersection = np.logical_and(gt, pred).sum()

        gt_pixels = gt.sum()
        pred_pixels = pred.sum()

        dice = (
            2.0 * intersection /
            (gt_pixels + pred_pixels)
            if (gt_pixels + pred_pixels) > 0
            else 1.0
        )

        union = np.logical_or(gt, pred).sum()

        iou = (
            intersection / union
            if union > 0
            else 1.0
        )

        true_positive = np.logical_and(gt, pred).sum()
        false_positive = np.logical_and(~gt, pred).sum()
        false_negative = np.logical_and(gt, ~pred).sum()

        precision = (
            true_positive /
            (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )

        recall = (
            true_positive /
            (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        # ----------------------------------------------------
        # PRINT STATISTICS
        # ----------------------------------------------------

        print("Band 0 min/max:",
              band0.min(),
              band0.max())

        print("Band 1 min/max:",
              band1.min(),
              band1.max())

        print("Ground truth positive pixels:",
              gt_pixels)

        print("Predicted positive pixels:",
              pred_pixels)

        print("Prediction probability min/max:",
              probability.min(),
              probability.max())

        print("Dice:",
              round(dice, 4))

        print("IoU:",
              round(iou, 4))

        print("Precision:",
              round(precision, 4))

        print("Recall:",
              round(recall, 4))

        # ====================================================
        # VISUALIZATION
        # ====================================================

        fig, axes = plt.subplots(
            1,
            5,
            figsize=(20, 4)
        )

        # ----------------------------------------------------
        # BAND 0
        # ----------------------------------------------------

        axes[0].imshow(
            band0,
            cmap="gray"
        )

        axes[0].set_title("Band 0")

        axes[0].axis("off")

        # ----------------------------------------------------
        # BAND 1
        # ----------------------------------------------------

        axes[1].imshow(
            band1,
            cmap="gray"
        )

        axes[1].set_title("Band 1")

        axes[1].axis("off")

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        axes[2].imshow(
            ground_truth,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[2].set_title(
            f"Ground Truth\npositive={gt_pixels}"
        )

        axes[2].axis("off")

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        axes[3].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[3].set_title(
            f"Prediction\npositive={pred_pixels}"
        )

        axes[3].axis("off")

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        axes[4].imshow(
            probability,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[4].set_title(
            "Probability"
        )

        axes[4].axis("off")

        # ----------------------------------------------------
        # FIGURE TITLE
        # ----------------------------------------------------

        fig.suptitle(
            dataset.files[index].name,
            fontsize=12
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR /
            f"prediction_{plot_number:02d}.png"
        )

        plt.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )

        print("Saved:", output_file)

        plt.show()

        plt.close()


print("\nDone.")