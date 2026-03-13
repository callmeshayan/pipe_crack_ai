"""
Script to prepare the Walls dataset for YOLO crack detection training.
This dataset has cracked and non-cracked images without annotations yet.

For this dataset, we'll:
1. Copy cracked images and create placeholder label files (you'll need to annotate them)
2. Copy non-cracked images with empty label files (no objects)
3. Split everything into train/val/test
"""

from pathlib import Path
import shutil
import random

# =========================
# CONFIG
# =========================
DATASET_SOURCE = Path("/Users/shayannaghashpour/Desktop/--/archive/archive-2/Walls")
PROJECT_ROOT = Path("/Users/shayannaghashpour/Desktop/--/pipe_crack_ai")

CRACKED_DIR = DATASET_SOURCE / "Cracked"
UNCRACKED_DIR = DATASET_SOURCE / "Non-cracked"
DATASET_DIR = PROJECT_ROOT / "dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42
CLASS_NAME = "crack"

# =========================
# HELPERS
# =========================
def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def clear_and_make_dataset_dirs():
    if DATASET_DIR.exists():
        print(f"Removing existing dataset at {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val", "test"]:
        safe_mkdir(DATASET_DIR / "images" / split)
        safe_mkdir(DATASET_DIR / "labels" / split)

def split_items(items):
    random.shuffle(items)
    n = len(items)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    return {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }

def copy_image_and_create_label(img_path: Path, split: str, has_crack: bool):
    """
    Copy image to dataset and create corresponding label file.
    If has_crack=True, create a placeholder label (needs manual annotation later).
    If has_crack=False, create empty label file.
    """
    dst_img = DATASET_DIR / "images" / split / img_path.name
    dst_lbl = DATASET_DIR / "labels" / split / f"{img_path.stem}.txt"
    
    shutil.copy2(img_path, dst_img)
    
    if has_crack:
        # Create placeholder - YOU MUST ANNOTATE THESE IMAGES LATER
        # Format: class x_center y_center width height (normalized 0-1)
        # For now, we create an empty file as a reminder to annotate
        dst_lbl.write_text("", encoding="utf-8")
    else:
        # No cracks = empty label file
        dst_lbl.write_text("", encoding="utf-8")

def build_dataset():
    print("1) Clearing and creating dataset directories...")
    clear_and_make_dataset_dirs()

    print("2) Loading cracked images...")
    cracked_images = list_images(CRACKED_DIR)
    print(f"   Found {len(cracked_images)} cracked images")

    print("3) Loading non-cracked images...")
    uncracked_images = list_images(UNCRACKED_DIR)
    print(f"   Found {len(uncracked_images)} non-cracked images")

    print("4) Splitting datasets...")
    cracked_splits = split_items(cracked_images)
    uncracked_splits = split_items(uncracked_images)

    print("5) Copying images and creating label files...")
    stats = {"train": 0, "val": 0, "test": 0}
    
    for split in ["train", "val", "test"]:
        print(f"   Processing {split} split...")
        for img_path in cracked_splits[split]:
            copy_image_and_create_label(img_path, split, has_crack=True)
            stats[split] += 1
        
        for img_path in uncracked_splits[split]:
            copy_image_and_create_label(img_path, split, has_crack=False)
            stats[split] += 1

    print("\n6) Creating dataset.yaml...")
    yaml_content = f"""# YOLO dataset configuration for crack detection
path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

# Class names
names:
  0: {CLASS_NAME}
"""
    (PROJECT_ROOT / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    for split, count in stats.items():
        cracked_count = len(cracked_splits[split])
        uncracked_count = len(uncracked_splits[split])
        print(f"{split.upper():5s}: {count:4d} images ({cracked_count} cracked, {uncracked_count} non-cracked)")
    
    print("\n" + "="*60)
    print("IMPORTANT: NEXT STEPS")
    print("="*60)
    print("⚠️  Your cracked images need to be ANNOTATED with bounding boxes!")
    print("    Currently, all label files are empty placeholders.")
    print("\nOptions to annotate:")
    print("  1. Use tools like:")
    print("     - Label Studio: https://labelstud.io/")
    print("     - Roboflow: https://roboflow.com/")
    print("     - CVAT: https://www.cvat.ai/")
    print("     - LabelImg: https://github.com/tzutalin/labelImg")
    print("\n  2. Export labels in YOLO format (class x y w h, normalized)")
    print("\n  3. Place exported labels in:")
    print(f"     {DATASET_DIR / 'labels' / 'train'}")
    print(f"     {DATASET_DIR / 'labels' / 'val'}")
    print(f"     {DATASET_DIR / 'labels' / 'test'}")
    print("\nOnce annotated, you can start training with:")
    print("  python scripts/train.py")
    print("="*60)

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    build_dataset()
