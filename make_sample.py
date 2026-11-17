"Copy a small random sample of the 140k Real and Fake Faces"

import random
import shutil
import sys
from pathlib import Path

SOURCE = Path(r"C:\datasets\140k\archive\real_vs_fake\real-vs-fake")
DEST = Path(__file__).resolve().parent / "dataset"
SEED = 42

SAMPLE_SIZES = {"train": 500, "valid": 100, "test": 100}
OVERFIT_SIZE = 16
CLASSES = ["real", "fake"]


def copy_images(files, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, out_dir / f.name)


def main():
    if not SOURCE.exists():
        sys.exit(f"Source folder not found: {SOURCE}")
    if DEST.exists() and any(DEST.iterdir()):
        sys.exit(f"{DEST} is not empty. Empty it first so samples don't get mixed.")

    rng = random.Random(SEED)

    for split, n in SAMPLE_SIZES.items():
        for cls in CLASSES:
            files = sorted((SOURCE / split / cls).glob("*.jpg"))
            chosen = rng.sample(files, n)
            copy_images(chosen, DEST / split / cls)
            print(f"{split}/{cls}: {n} images")

            if split == "train":
                copy_images(chosen[:OVERFIT_SIZE], DEST / "overfit" / cls)
                print(f"overfit/{cls}: {OVERFIT_SIZE} images")

    print(f"Done. Sample saved to {DEST}")


if __name__ == "__main__":
    main()
