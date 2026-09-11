from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):

    def __init__(
        self,
        root_dir,
        normalize=False,
        transform=None,
    ):
        """
        Dataset for:

        root_dir/
            band0/
            band1/
            masks/

        Each directory contains matching .npy files.

        Example:
            band0/image001_patch_01.npy
            band1/image001_patch_01.npy
            masks/image001_patch_01.npy
        """

        self.root_dir = Path(root_dir)

        self.band0_dir = self.root_dir / "band0"
        self.band1_dir = self.root_dir / "band1"
        self.mask_dir = self.root_dir / "masks"

        self.normalize = normalize
        self.transform = transform

        # -------------------------------------------------
        # Get all band0 files
        # -------------------------------------------------

        self.files = sorted(
            self.band0_dir.glob("*.npy")
        )

        if not self.files:
            raise RuntimeError(
                f"No .npy files found in {self.band0_dir}"
            )

        # -------------------------------------------------
        # Verify that corresponding files exist
        # -------------------------------------------------

        for band0_path in self.files:

            filename = band0_path.name

            band1_path = self.band1_dir / filename
            mask_path = self.mask_dir / filename

            if not band1_path.exists():
                raise FileNotFoundError(
                    f"Missing band1 file: {band1_path}"
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask file: {mask_path}"
                )

        print(
            f"Loaded dataset with "
            f"{len(self.files)} samples"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        # -------------------------------------------------
        # Get paths
        # -------------------------------------------------

        band0_path = self.files[index]

        filename = band0_path.name

        band1_path = self.band1_dir / filename
        mask_path = self.mask_dir / filename

        # -------------------------------------------------
        # Load NumPy arrays
        # -------------------------------------------------

        band0 = np.load(band0_path)
        band1 = np.load(band1_path)
        mask = np.load(mask_path)

        # -------------------------------------------------
        # Convert dtype
        # -------------------------------------------------

        band0 = band0.astype(np.float32)
        band1 = band1.astype(np.float32)

        mask = mask.astype(np.float32)

        # -------------------------------------------------
        # Stack VV + VH
        #
        # (H,W) + (H,W)
        #       ↓
        # (H,W,2)
        # -------------------------------------------------

        image = np.stack(
            [band0, band1],
            axis=-1
        )

        # -------------------------------------------------
        # Optional transform
        #
        # Transform receives:
        #
        # image: H,W,2
        # mask : H,W
        # -------------------------------------------------

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

        # -------------------------------------------------
        # H,W,2 -> 2,H,W
        # -------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # -------------------------------------------------
        # NumPy -> PyTorch
        # -------------------------------------------------

        image = torch.from_numpy(
            image.copy()
        ).float()

        mask = torch.from_numpy(
            mask.copy()
        ).float()

        # -------------------------------------------------
        # H,W -> 1,H,W
        # -------------------------------------------------

        mask = mask.unsqueeze(0)

        return image, mask


if __name__ == "__main__":
    dataset = OilSpillDataset(
        root_dir="./processed"
    )

    print("Dataset size:", len(dataset))

    image, mask = dataset[0]

    print("Image:", image.shape, image.dtype)
    print("Mask :", mask.shape, mask.dtype)