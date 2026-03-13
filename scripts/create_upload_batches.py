"""
Organize cracked images into batches of 10 for Roboflow free tier uploads.
"""

from pathlib import Path
import shutil

PROJECT_ROOT = Path("/Users/shayannaghashpour/Desktop/--/pipe_crack_ai")
UPLOAD_DIR = PROJECT_ROOT / "roboflow_upload"
BATCHED_DIR = PROJECT_ROOT / "roboflow_batched"

BATCH_SIZE = 10

def create_batches():
    """Organize images into batches of 10."""
    
    # Clean and create batched directory
    if BATCHED_DIR.exists():
        shutil.rmtree(BATCHED_DIR)
    BATCHED_DIR.mkdir()
    
    # Get all images
    all_images = sorted(UPLOAD_DIR.glob("*.jpg"))
    total_images = len(all_images)
    num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE
    
    print("="*60)
    print("Creating upload batches (10 images each)")
    print("="*60)
    print(f"Total images: {total_images}")
    print(f"Number of batches: {num_batches}\n")
    
    for batch_num in range(num_batches):
        # Create batch folder
        batch_dir = BATCHED_DIR / f"batch_{batch_num + 1:02d}"
        batch_dir.mkdir()
        
        # Get images for this batch
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_images)
        batch_images = all_images[start_idx:end_idx]
        
        # Copy images to batch folder
        for img in batch_images:
            shutil.copy2(img, batch_dir / img.name)
        
        print(f"✓ Batch {batch_num + 1:2d}: {len(batch_images)} images → {batch_dir.name}/")
    
    print("\n" + "="*60)
    print("Batches created!")
    print("="*60)
    print(f"Location: {BATCHED_DIR}")
    print("\n📤 Upload Instructions:")
    print("  1. Upload batch_01/ to Roboflow")
    print("  2. Wait for processing")
    print("  3. Upload batch_02/")
    print("  4. Repeat for all batches")
    print("\n💡 Tip: You can upload multiple batches in quick succession")
    print("="*60)

if __name__ == "__main__":
    create_batches()
