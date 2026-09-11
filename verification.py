import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# =========================================================
# CONFIG
# =========================================================

MASK_DIR = Path("./processed/masks")

NUM_WORKERS = max(1, (os.cpu_count() or 2) - 1)


# =========================================================
# PROCESS ONE MASK
# =========================================================

def analyze_mask(mask_path):

    mask = np.load(mask_path)

    total_pixels = mask.size
    oil_pixels = np.count_nonzero(mask)

    oil_ratio = oil_pixels / total_pixels

    is_blank = oil_pixels == 0
    is_full = oil_pixels == total_pixels

    return {
        "name": mask_path.name,
        "total_pixels": total_pixels,
        "oil_pixels": int(oil_pixels),
        "oil_ratio": oil_ratio,
        "blank": is_blank,
        "full": is_full,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    mask_files = sorted(MASK_DIR.glob("*.npy"))

    if not mask_files:
        print(f"No .npy masks found in {MASK_DIR}")
        return

    print(f"Masks found : {len(mask_files)}")
    print(f"Workers     : {NUM_WORKERS}")
    print()

    blank_count = 0
    non_blank_count = 0
    full_count = 0

    oil_ratios = []

    # -----------------------------------------------------
    # Multiprocessing
    # -----------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:

        futures = [
            executor.submit(analyze_mask, path)
            for path in mask_files
        ]

        for i, future in enumerate(as_completed(futures), 1):

            result = future.result()

            if result["blank"]:
                blank_count += 1
            else:
                non_blank_count += 1

            if result["full"]:
                full_count += 1

            oil_ratios.append(result["oil_ratio"])

            # Progress
            if i % 100 == 0 or i == len(mask_files):
                print(
                    f"Processed "
                    f"{i}/{len(mask_files)}"
                )

    # =====================================================
    # STATISTICS
    # =====================================================

    total = len(mask_files)

    blank_ratio = blank_count / total
    non_blank_ratio = non_blank_count / total
    full_ratio = full_count / total

    oil_ratios = np.array(oil_ratios)

    print("\n" + "=" * 50)
    print("MASK STATISTICS")
    print("=" * 50)

    print(f"Total masks       : {total}")

    print(
        f"Blank masks       : "
        f"{blank_count} "
        f"({blank_ratio:.2%})"
    )

    print(
        f"Non-blank masks   : "
        f"{non_blank_count} "
        f"({non_blank_ratio:.2%})"
    )

    print(
        f"Completely full   : "
        f"{full_count} "
        f"({full_ratio:.2%})"
    )

    print("\n" + "=" * 50)
    print("OIL PIXEL RATIO")
    print("=" * 50)

    print(f"Minimum : {oil_ratios.min():.6%}")
    print(f"Maximum : {oil_ratios.max():.6%}")
    print(f"Mean    : {oil_ratios.mean():.6%}")
    print(f"Median  : {np.median(oil_ratios):.6%}")


if __name__ == "__main__":
    main()