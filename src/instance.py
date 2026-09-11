import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import rasterio

class Instance:
    def __init__(self):
        self.band0: np.ndarray | None = None
        self.band1: np.ndarray | None = None
        self.mask0: np.ndarray | None = None

    def read_from_paths(self, paths: list[Path]) -> None:

        assert len(paths) == 2

        for i, path in enumerate(paths):
            with rasterio.open(path) as f:
                if i == 0:
                    self.band0 = f.read(1)
                    self.band1 = f.read(2)
                else:
                    self.mask0 = f.read(1)
            
    def plot(self, out: Path) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15,15))

        axes[0].imshow(self.band0, cmap="viridis")
        axes[0].set_title("band0")

        axes[1].imshow(self.band1, cmap="viridis")
        axes[1].set_title("band1")
        
        axes[2].imshow(self.mask0, cmap="viridis")
        axes[2].set_title("mask0")

        plt.tight_layout()
        plt.savefig(out)
        