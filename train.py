# train.py

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from segmentation_models_pytorch import Unet

from src.dataset import OilSpillDataset


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_DIR = "./processed"

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

EPOCHS = 10

BATCH_SIZE = 2

LEARNING_RATE = 1e-4

# L2 regularisation
WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------
# Validation
#
# 0.05% = 0.0005
# ---------------------------------------------------------

VAL_PERCENT = 0.0005

# ---------------------------------------------------------
# DataLoader
# ---------------------------------------------------------

NUM_WORKERS = min(8, os.cpu_count() or 1)

PIN_MEMORY = True

# ---------------------------------------------------------
# Early stopping
# ---------------------------------------------------------

EARLY_STOPPING_PATIENCE = 10

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

SEED = 42


# =========================================================
# SEED
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# DICE LOSS
# =========================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.contiguous()
        targets = targets.contiguous()

        intersection = (
            probabilities * targets
        ).sum(dim=(1, 2, 3))

        denominator = (
            probabilities.sum(dim=(1, 2, 3))
            +
            targets.sum(dim=(1, 2, 3))
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


# =========================================================
# COMBINED LOSS
# =========================================================

class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        return bce_loss + dice_loss


# =========================================================
# DICE METRIC
# =========================================================

def dice_score(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > 0.5
    ).float()

    intersection = (
        predictions * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        predictions.sum(dim=(1, 2, 3))
        +
        targets.sum(dim=(1, 2, 3))
    )

    dice = (
        2.0 * intersection + 1e-7
    ) / (
        denominator + 1e-7
    )

    return dice.mean().item()


# =========================================================
# IOU METRIC
# =========================================================

def iou_score(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > 0.5
    ).float()

    intersection = (
        predictions * targets
    ).sum(dim=(1, 2, 3))

    union = (
        predictions
        +
        targets
        -
        predictions * targets
    ).sum(dim=(1, 2, 3))

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return iou.mean().item()


# =========================================================
# CREATE MODEL
# =========================================================

def create_model():

    model = Unet(
        encoder_name="mobilenet_v2",

        # ImageNet pretrained weights
        encoder_weights="imagenet",

        # Two SAR bands
        in_channels=2,

        # Binary segmentation
        classes=1,

        activation=None,
    )

    return model


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):

    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    num_batches = len(loader)

    for batch_idx, (images, masks) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # -------------------------------------------------
        # Mixed precision
        # -------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):

            logits = model(images)

            loss = criterion(
                logits,
                masks
            )

        # -------------------------------------------------
        # Backpropagation
        # -------------------------------------------------

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        with torch.no_grad():

            dice = dice_score(
                logits,
                masks
            )

            iou = iou_score(
                logits,
                masks
            )

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

        if (
            batch_idx + 1
        ) % 50 == 0:

            print(
                f"    Batch "
                f"{batch_idx + 1}/{num_batches} "
                f"| loss={loss.item():.4f}"
            )

    return (
        total_loss / num_batches,
        total_dice / num_batches,
        total_iou / num_batches,
    )


# =========================================================
# VALIDATION
# =========================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    num_batches = len(loader)

    for images, masks in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):

            logits = model(images)

            loss = criterion(
                logits,
                masks
            )

        dice = dice_score(
            logits,
            masks
        )

        iou = iou_score(
            logits,
            masks
        )

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    return (
        total_loss / num_batches,
        total_dice / num_batches,
        total_iou / num_batches,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(SEED)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print("\nLoading dataset...")

    dataset = OilSpillDataset(
        root_dir=DATASET_DIR,
        transform=None,
    )

    total_size = len(dataset)

    # -----------------------------------------------------
    # Validation size
    # -----------------------------------------------------

    val_size = max(
        1,
        int(total_size * VAL_PERCENT)
    )

    train_size = total_size - val_size

    print(
        f"\nTotal samples : {total_size}"
    )

    print(
        f"Train samples : {train_size}"
    )

    print(
        f"Val samples   : {val_size}"
    )

    print(
        f"Validation    : "
        f"{VAL_PERCENT * 100:.4f}%"
    )

    # -----------------------------------------------------
    # Train / validation split
    # -----------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print("\nCreating MobileNetV2 U-Net...")

    model = create_model()

    model = model.to(device)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = BCEDiceLoss()

    # -----------------------------------------------------
    # Optimizer
    #
    # weight_decay = L2 regularisation
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Learning-rate scheduler
    # -----------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,

        mode="min",

        factor=0.5,

        patience=3,

        min_lr=1e-7,
    )

    # -----------------------------------------------------
    # Mixed precision scaler
    # -----------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    # -----------------------------------------------------
    # Training state
    # -----------------------------------------------------

    best_val_loss = float("inf")

    best_val_dice = 0.0

    epochs_without_improvement = 0

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for epoch in range(1, EPOCHS + 1):

        print("\n")
        print("=" * 60)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print("=" * 60)

        # -------------------------------------------------
        # Current LR
        # -------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning rate: "
            f"{current_lr:.2e}"
        )

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        train_loss, train_dice, train_iou = train_one_epoch(
            model=model,

            loader=train_loader,

            optimizer=optimizer,

            criterion=criterion,

            scaler=scaler,

            device=device,
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        val_loss, val_dice, val_iou = validate(
            model=model,

            loader=val_loader,

            criterion=criterion,

            device=device,
        )

        # -------------------------------------------------
        # Scheduler
        # -------------------------------------------------

        scheduler.step(val_loss)

        # -------------------------------------------------
        # Print results
        # -------------------------------------------------

        print("\nResults:")

        print(
            f"Train Loss : {train_loss:.6f}"
        )

        print(
            f"Train Dice : {train_dice:.6f}"
        )

        print(
            f"Train IoU  : {train_iou:.6f}"
        )

        print(
            f"Val Loss   : {val_loss:.6f}"
        )

        print(
            f"Val Dice   : {val_dice:.6f}"
        )

        print(
            f"Val IoU    : {val_iou:.6f}"
        )

        # -------------------------------------------------
        # Save best model
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_val_dice = val_dice

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "val_loss":
                        val_loss,

                    "val_dice":
                        val_dice,

                    "val_iou":
                        val_iou,

                    "train_loss":
                        train_loss,

                    "train_dice":
                        train_dice,

                    "train_iou":
                        train_iou,

                    "learning_rate":
                        optimizer.param_groups[0]["lr"],
                },

                BEST_MODEL_PATH,
            )

            print(
                f"\n✓ Best model saved:"
                f" {BEST_MODEL_PATH}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break

    # =====================================================
    # FINISHED
    # =====================================================

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation loss : "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best validation Dice : "
        f"{best_val_dice:.6f}"
    )

    print(
        f"Best model           : "
        f"{BEST_MODEL_PATH}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()