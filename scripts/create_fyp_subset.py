"""
Create a manageable FYP-sized subset from the full dataset.
Target: ~1000 images (350 cracked, 650 non-cracked)
"""

from pathlib import Path
import random
import shutil

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path("/Users/shayannaghashpour/Desktop/--/pipe_crack_ai")
DATASET_DIR = PROJECT_ROOT / "dataset"
BACKUP_DIR = PROJECT_ROOT / "dataset_backup_full"

# Target counts per split (total ~1000 images)
TARGET = {
    "train": {"cracked": 245, "non_cracked": 455},  # 700 total
    "val": {"cracked": 70, "non_cracked": 130},     # 200 total
    "test": {"cracked": 35, "non_cracked": 65},     # 100 total
}

RANDOM_SEED = 42

# Cracked images have prefixes 7069-7091, non-cracked are 7101+
CRACKED_PREFIXES = [f"7{i:03d}" for i in range(69, 92)]

# =========================
# HELPERS
# =========================
def is_cracked(image_path):
    """Check if image is from cracked folder based on filename."""
    stem = image_path.stem
    prefix = stem.split("-")[0] if "-" in stem else stem[:4]
    return prefix in CRACKED_PREFIXES

def get_images_by_type(split_dir):
    """Separate images into cracked and non-cracked."""
    images_dir = DATASET_DIR / "images" / split_dir
    all_images = list(images_dir.glob("*.jpg"))
    
    cracked = [img for img in all_images if is_cracked(img)]
    non_cracked = [img for img in all_images if not is_cracked(img)]
    
    return cracked, non_cracked

def backup_full_dataset():
    """Backup the full dataset before modification."""
    if BACKUP_DIR.exists():
        print(f"Backup already exists at {BACKUP_DIR}")
        response = input("Overwrite backup? (y/n): ").lower()
        if response != 'y':
            print("Using existing backup...")
            return
        shutil.rmtree(BACKUP_DIR)
    
    print(f"Creating backup of full dataset to {BACKUP_DIR}...")
    shutil.copytree(DATASET_DIR, BACKUP_DIR)
    print("✓ Backup complete!")

def delete_unselected_images(split_dir, selected_images):
    """Delete images and labels that weren't selected."""
    images_dir = DATASET_DIR / "images" / split_dir
    labels_dir = DATASET_DIR / "labels" / split_dir
    
    # Get all current images
    all_images = set(images_dir.glob("*.jpg"))
    selected_set = set(selected_images)
    to_delete = all_images - selected_set
    
    deleted_count = 0
    for img_path in to_delete:
        # Delete image
        img_path.unlink()
        
        # Delete corresponding label
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            label_path.unlink()
        
        deleted_count += 1
    
    return deleted_count

def create_subset():
    """Create FYP-sized subset by randomly sampling and deleting extras."""
    random.seed(RANDOM_SEED)
    
    print("="*60)
    print("Creating FYP-Sized Dataset Subset")
    print("="*60)
    print(f"Target: ~1000 images (350 cracked, 650 non-cracked)")
    print(f"Split: 70% train, 20% val, 10% test\n")
    
    # Backup first
    backup_full_dataset()
    
    total_kept = 0
    total_deleted = 0
    
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split.upper()} split...")
        
        # Get current images
        cracked, non_cracked = get_images_by_type(split)
        
        print(f"  Current: {len(cracked)} cracked, {len(non_cracked)} non-cracked")
        
        # Randomly sample target amounts
        target_cracked = TARGET[split]["cracked"]
        target_non_cracked = TARGET[split]["non_cracked"]
        
        selected_cracked = random.sample(cracked, min(target_cracked, len(cracked)))
        selected_non_cracked = random.sample(non_cracked, min(target_non_cracked, len(non_cracked)))
        
        selected_images = selected_cracked + selected_non_cracked
        
        # Delete unselected images
        deleted = delete_unselected_images(split, selected_images)
        kept = len(selected_images)
        
        total_kept += kept
        total_deleted += deleted
        
        print(f"  Kept: {len(selected_cracked)} cracked, {len(selected_non_cracked)} non-cracked")
        print(f"  Deleted: {deleted} images")
    
    print("\n" + "="*60)
    print("Dataset Subset Created Successfully!")
    print("="*60)
    print(f"Total images kept: {total_kept}")
    print(f"Total images deleted: {total_deleted}")
    print(f"\nFull backup saved to: {BACKUP_DIR}")
    print("\nYour dataset is now FYP-sized and ready for annotation!")
    print("="*60)

if __name__ == "__main__":
    print("\n⚠️  WARNING: This will DELETE excess images from your dataset!")
    print("A full backup will be created first.\n")
    
    response = input("Proceed? (yes/no): ").lower()
    if response == "yes":
        create_subset()
    else:
        print("Cancelled.")
