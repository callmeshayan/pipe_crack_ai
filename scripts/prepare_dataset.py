from pathlib import Path
import shutil
import random

# =========================
# CONFIG
# =========================
BASE_DIR = Path(".").resolve()

CRACKED_DIR = BASE_DIR / "cracked"
UNCRACKED_DIR = BASE_DIR / "uncracked"
CRACKED_LABELS_DIR = BASE_DIR / "cracked_labels"   # manual labels go here

DATASET_DIR = BASE_DIR / "dataset"
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
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val", "test"]:
        safe_mkdir(DATASET_DIR / "images" / split)
        safe_mkdir(DATASET_DIR / "labels" / split)

def rename_images(folder: Path, prefix: str):
    images = list_images(folder)
    mapping = []

    for idx, img_path in enumerate(images, start=1):
        new_name = f"{prefix}_{idx:04d}{img_path.suffix.lower()}"
        new_path = folder / new_name

        # avoid renaming to same file unnecessarily
        if img_path.name != new_name:
            img_path.rename(new_path)
        else:
            new_path = img_path

        mapping.append(new_path)

    return mapping

def create_empty_label_files_for_uncracked(uncracked_images):
    for img_path in uncracked_images:
        txt_path = UNCRACKED_DIR / f"{img_path.stem}.txt"
        txt_path.write_text("", encoding="utf-8")

def check_cracked_labels_exist(cracked_images):
    missing = []
    empty = []

    for img_path in cracked_images:
        label_path = CRACKED_LABELS_DIR / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing.append(img_path.name)
        else:
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                empty.append(img_path.name)

    return missing, empty

def check_uncracked_labels_exist(uncracked_images):
    missing = []
    non_empty = []

    for img_path in uncracked_images:
        label_path = UNCRACKED_DIR / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing.append(img_path.name)
        else:
            content = label_path.read_text(encoding="utf-8").strip()
            if content:
                non_empty.append(img_path.name)

    return missing, non_empty

def split_items(items):
    random.shuffle(items)
    n = len(items)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    return {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }

def copy_pair(img_path: Path, label_path: Path, split: str):
    dst_img = DATASET_DIR / "images" / split / img_path.name
    dst_lbl = DATASET_DIR / "labels" / split / label_path.name
    shutil.copy2(img_path, dst_img)
    shutil.copy2(label_path, dst_lbl)

def build_dataset(cracked_images, uncracked_images):
    all_pairs = []

    for img_path in cracked_images:
        lbl_path = CRACKED_LABELS_DIR / f"{img_path.stem}.txt"
        all_pairs.append((img_path, lbl_path))

    for img_path in uncracked_images:
        lbl_path = UNCRACKED_DIR / f"{img_path.stem}.txt"
        all_pairs.append((img_path, lbl_path))

    splits = split_items(all_pairs)

    for split, pairs in splits.items():
        for img_path, lbl_path in pairs:
            copy_pair(img_path, lbl_path, split)

    return splits

def create_dataset_yaml():
    yaml_text = f"""path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
"""
    (BASE_DIR / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

def validate_final_dataset():
    issues = []

    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split

        images = list_images(img_dir)
        labels = sorted(lbl_dir.glob("*.txt"))

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        missing_labels = img_stems - lbl_stems
        orphan_labels = lbl_stems - img_stems

        if missing_labels:
            issues.append(f"{split}: missing labels for {sorted(missing_labels)}")
        if orphan_labels:
            issues.append(f"{split}: orphan labels {sorted(orphan_labels)}")

    return issues

def main():
    random.seed(RANDOM_SEED)

    if not CRACKED_DIR.exists() or not UNCRACKED_DIR.exists():
        raise FileNotFoundError("Both 'cracked' and 'uncracked' folders must exist.")

    safe_mkdir(CRACKED_LABELS_DIR)

    print("1) Renaming cracked images...")
    cracked_images = rename_images(CRACKED_DIR, "crack")

    print("2) Renaming uncracked images...")
    uncracked_images = rename_images(UNCRACKED_DIR, "clean")

    print("3) Creating empty label files for uncracked images...")
    create_empty_label_files_for_uncracked(uncracked_images)

    print("4) Checking labels...")
    missing_cracked, empty_cracked = check_cracked_labels_exist(cracked_images)
    missing_uncracked, non_empty_uncracked = check_uncracked_labels_exist(uncracked_images)

    if missing_cracked:
        print("\nMissing cracked labels:")
        for x in missing_cracked:
            print("  ", x)

    if empty_cracked:
        print("\nEmpty cracked labels:")
        for x in empty_cracked:
            print("  ", x)

    if missing_uncracked:
        print("\nMissing uncracked labels:")
        for x in missing_uncracked:
            print("  ", x)

    if non_empty_uncracked:
        print("\nNon-empty labels found in uncracked:")
        for x in non_empty_uncracked:
            print("  ", x)

    if missing_cracked or empty_cracked or missing_uncracked or non_empty_uncracked:
        print("\nFix the label issues first, then rerun.")
        return

    print("5) Building dataset structure...")
    clear_and_make_dataset_dirs()
    splits = build_dataset(cracked_images, uncracked_images)

    print("6) Creating dataset.yaml...")
    create_dataset_yaml()

    print("7) Validating final dataset...")
    issues = validate_final_dataset()
    if issues:
        print("\nValidation issues found:")
        for issue in issues:
            print(" ", issue)
        return

    print("\nDone.")
    for split, pairs in splits.items():
        print(f"{split}: {len(pairs)} files")

    print("\nDataset ready:")
    print(DATASET_DIR)
    print(BASE_DIR / "dataset.yaml")

if __name__ == "__main__":
    main()
