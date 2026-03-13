"""
Gather all positive (cracked) images into one folder for easy Roboflow upload.
Uses the positive_images_list.json generated during dataset preparation.
"""

from pathlib import Path
import shutil
import json

PROJECT_ROOT = Path("/Users/shayannaghashpour/Desktop/--/pipe_crack_ai")
DATASET_DIR = PROJECT_ROOT / "dataset"
UPLOAD_DIR = PROJECT_ROOT / "roboflow_upload"
POSITIVE_LIST_FILE = PROJECT_ROOT / "positive_images_list.json"

def gather_cracked_images():
    """Copy all positive (cracked) images to upload folder."""
    
    # Load positive images list
    if not POSITIVE_LIST_FILE.exists():
        print(f"❌ Error: {POSITIVE_LIST_FILE.name} not found!")
        print("   Run prepare_archive_dataset.py first.")
        return
    
    positive_images = json.loads(POSITIVE_LIST_FILE.read_text())
    
    # Clean and create upload directory
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir()
    
    print("Gathering positive (cracked) images for Roboflow upload...")
    print("="*60)
    
    total_copied = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for img_info in positive_images:
        split = img_info['split']
        filename = img_info['filename']
        img_path = PROJECT_ROOT / img_info['path']
        
        if img_path.exists():
            # Copy to upload folder with split prefix
            dest = UPLOAD_DIR / f"{split}_{filename}"
            shutil.copy2(img_path, dest)
            total_copied += 1
            split_counts[split] += 1
    
    for split in ['train', 'val', 'test']:
        print(f"✓ {split:5s}: {split_counts[split]:3d} positive images")
    
    print("="*60)
    print(f"Total: {total_copied} positive (cracked) images ready for upload")
    print(f"\nLocation: {UPLOAD_DIR}")
    print("\n📤 Upload these to Roboflow for annotation!")
    print("="*60)

if __name__ == "__main__":
    gather_cracked_images()
