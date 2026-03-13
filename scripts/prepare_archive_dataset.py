"""
Prepare the archive/dataset for YOLO training with FYP-appropriate size
Dataset location: desktop/--/archive/dataset/
Structure: Positive/ (cracked) and Negative/ (non-cracked)
Total: 40,000 images (20,000 each)
Output: ~1,000 images for FYP sweet spot
"""

import os
import random
import shutil
from pathlib import Path

# Dataset paths
SOURCE_DIR = Path("/Users/shayannaghashpour/Desktop/--/archive/dataset")
POSITIVE_DIR = SOURCE_DIR / "Positive"
NEGATIVE_DIR = SOURCE_DIR / "Negative"

# Target YOLO dataset structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# FYP-appropriate dataset size
TARGET_TOTAL = 1000
TARGET_POSITIVE = 350  # 35% cracked images
TARGET_NEGATIVE = 650  # 65% non-cracked images

# Train/val/test split ratios
TRAIN_RATIO = 0.70  # 70%
VAL_RATIO = 0.20    # 20%
TEST_RATIO = 0.10   # 10%

# Image extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')

random.seed(42)  # For reproducibility


def get_image_files(directory):
    """Get all image files from directory"""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(list(directory.glob(f"*{ext}")))
    return files


def create_empty_label(label_path):
    """Create empty label file for YOLO format"""
    label_path.write_text("")


def process_dataset():
    """Process and split the dataset"""
    print("=" * 70)
    print("PREPARING ARCHIVE DATASET FOR YOLO TRAINING")
    print("=" * 70)
    print(f"Source: {SOURCE_DIR}")
    print(f"Target size: {TARGET_TOTAL} images ({TARGET_POSITIVE} positive, {TARGET_NEGATIVE} negative)")
    print(f"Split: {TRAIN_RATIO:.0%} train, {VAL_RATIO:.0%} val, {TEST_RATIO:.0%} test")
    print("=" * 70)
    
    # Get all images
    print("\n[1/5] Counting images...")
    positive_images = get_image_files(POSITIVE_DIR)
    negative_images = get_image_files(NEGATIVE_DIR)
    
    print(f"  Found {len(positive_images)} positive (cracked) images")
    print(f"  Found {len(negative_images)} negative (non-cracked) images")
    print(f"  Total available: {len(positive_images) + len(negative_images)} images")
    
    # Randomly select subset
    print(f"\n[2/5] Selecting random subset of {TARGET_TOTAL} images...")
    selected_positive = random.sample(positive_images, min(TARGET_POSITIVE, len(positive_images)))
    selected_negative = random.sample(negative_images, min(TARGET_NEGATIVE, len(negative_images)))
    
    print(f"  Selected {len(selected_positive)} positive images")
    print(f"  Selected {len(selected_negative)} negative images")
    
    # Shuffle and split
    print("\n[3/5] Splitting into train/val/test...")
    random.shuffle(selected_positive)
    random.shuffle(selected_negative)
    
    # Calculate split points
    pos_train_end = int(len(selected_positive) * TRAIN_RATIO)
    pos_val_end = int(len(selected_positive) * (TRAIN_RATIO + VAL_RATIO))
    
    neg_train_end = int(len(selected_negative) * TRAIN_RATIO)
    neg_val_end = int(len(selected_negative) * (TRAIN_RATIO + VAL_RATIO))
    
    splits = {
        'train': {
            'positive': selected_positive[:pos_train_end],
            'negative': selected_negative[:neg_train_end]
        },
        'val': {
            'positive': selected_positive[pos_train_end:pos_val_end],
            'negative': selected_negative[neg_train_end:neg_val_end]
        },
        'test': {
            'positive': selected_positive[pos_val_end:],
            'negative': selected_negative[neg_val_end:]
        }
    }
    
    # Clean and create directories
    print("\n[4/5] Creating YOLO directory structure...")
    if DATASET_DIR.exists():
        print(f"  Removing existing dataset directory...")
        shutil.rmtree(DATASET_DIR)
    
    for split in ['train', 'val', 'test']:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    
    # Copy images and create labels
    print("\n[5/5] Copying images and creating label files...")
    stats = {
        'train': {'positive': 0, 'negative': 0},
        'val': {'positive': 0, 'negative': 0},
        'test': {'positive': 0, 'negative': 0}
    }
    
    # Track positive images for annotation
    positive_images_list = []
    
    for split_name, split_data in splits.items():
        print(f"\n  Processing {split_name} split...")
        
        for category, images in split_data.items():
            for img_path in images:
                # Copy image
                dest_img = IMAGES_DIR / split_name / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Track positive images
                if category == 'positive':
                    positive_images_list.append({
                        'split': split_name,
                        'filename': img_path.name,
                        'path': str(dest_img.relative_to(PROJECT_ROOT))
                    })
                
                # Create empty label file (will be populated after annotation)
                label_name = img_path.stem + ".txt"
                label_path = LABELS_DIR / split_name / label_name
                create_empty_label(label_path)
                
                stats[split_name][category] += 1
        
        total_split = stats[split_name]['positive'] + stats[split_name]['negative']
        print(f"    ✓ {total_split} images ({stats[split_name]['positive']} positive, {stats[split_name]['negative']} negative)")
    
    # Save positive images list for annotation workflow
    positive_list_file = PROJECT_ROOT / "positive_images_list.json"
    import json
    positive_list_file.write_text(json.dumps(positive_images_list, indent=2))
    print(f"\n  ✓ Saved positive images list to: {positive_list_file.name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\nDataset location: {DATASET_DIR}")
    print(f"\nSplit summary:")
    print(f"  Train: {stats['train']['positive'] + stats['train']['negative']} images "
          f"({stats['train']['positive']} positive, {stats['train']['negative']} negative)")
    print(f"  Val:   {stats['val']['positive'] + stats['val']['negative']} images "
          f"({stats['val']['positive']} positive, {stats['val']['negative']} negative)")
    print(f"  Test:  {stats['test']['positive'] + stats['test']['negative']} images "
          f"({stats['test']['positive']} positive, {stats['test']['negative']} negative)")
    
    total_images = sum(stats[s]['positive'] + stats[s]['negative'] for s in ['train', 'val', 'test'])
    total_positive = sum(stats[s]['positive'] for s in ['train', 'val', 'test'])
    total_negative = sum(stats[s]['negative'] for s in ['train', 'val', 'test'])
    
    print(f"\n  TOTAL: {total_images} images ({total_positive} positive, {total_negative} negative)")
    print(f"\n⚠️  NOTE: All label files are empty and need annotation!")
    print(f"  Use Roboflow or auto-labeling script to annotate the {total_positive} positive images.")
    print("=" * 70)


if __name__ == "__main__":
    process_dataset()
