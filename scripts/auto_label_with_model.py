"""
Use existing crack detection model to auto-label images for training.
This generates YOLO format labels from your Roboflow workflow predictions.
"""

import os
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any

import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from tqdm import tqdm

# Load environment variables
ENV_PATH = Path("/Users/shayannaghashpour/Desktop/--/Final Year Project/Code/crack_app/.env")
load_dotenv(ENV_PATH, override=True)

API_KEY = os.getenv("RF_API_KEY", "").strip()
WORKSPACE = os.getenv("RF_WORKSPACE", "").strip()
WORKFLOW_ID = os.getenv("RF_WORKFLOW_ID", "").strip()

if not API_KEY or not WORKSPACE or not WORKFLOW_ID:
    raise ValueError("Missing .env vars - check your crack_app/.env file")

# Setup client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)

# Directories
PROJECT_ROOT = Path("/Users/shayannaghashpour/Desktop/--/pipe_crack_ai")
DATASET_DIR = PROJECT_ROOT / "dataset"
AUTO_LABELS_DIR = PROJECT_ROOT / "auto_labels_review"
AUTO_LABELS_DIR.mkdir(exist_ok=True)

# Settings
CONF_THRESH = 0.3  # Lower threshold to catch more cracks for review
MIN_CRACK_AREA = 50  # Smaller minimum area

print("="*60)
print("AUTO-LABELING WITH EXISTING MODEL")
print("="*60)
print(f"Workspace: {WORKSPACE}")
print(f"Workflow:  {WORKFLOW_ID}")
print(f"Confidence threshold: {CONF_THRESH}")
print("="*60)


def extract_predictions(result: Any) -> List[Dict[str, Any]]:
    """Extract predictions from Roboflow workflow result"""
    if isinstance(result, list):
        for item in result:
            preds = extract_predictions(item)
            if preds:
                return preds
        return []
    if isinstance(result, dict):
        preds = result.get("predictions")
        if isinstance(preds, list):
            return preds
        for v in result.values():
            preds = extract_predictions(v)
            if preds:
                return preds
    return []


def convert_to_yolo_format(pred: Dict[str, Any], img_width: int, img_height: int) -> str:
    """
    Convert prediction to YOLO format: class x_center y_center width height
    All values normalized to [0, 1]
    """
    x_center = float(pred.get("x", 0))
    y_center = float(pred.get("y", 0))
    width = float(pred.get("width", 0))
    height = float(pred.get("height", 0))
    
    # Normalize to 0-1
    x_norm = x_center / img_width
    y_norm = y_center / img_height
    w_norm = width / img_width
    h_norm = height / img_height
    
    # Clamp to valid range
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    
    # Class 0 for crack
    return f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"


def process_image(img_path: Path, split: str) -> Dict[str, Any]:
    """Run inference on single image and generate YOLO label"""
    img = cv2.imread(str(img_path))
    if img is None:
        return {"error": "Could not read image", "predictions": 0}
    
    img_height, img_width = img.shape[:2]
    
    # Run inference
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as f:
            cv2.imwrite(f.name, img)
            
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": f.name},
                use_cache=False,
            )
        
        # Extract predictions
        preds = extract_predictions(result)
        
        # Filter by confidence and area
        filtered_preds = []
        for p in preds:
            conf = float(p.get("confidence", p.get("score", 0.0)) or 0.0)
            if conf < CONF_THRESH:
                continue
            
            area = float(p.get("width", 0)) * float(p.get("height", 0))
            if area < MIN_CRACK_AREA:
                continue
            
            filtered_preds.append(p)
        
        # Convert to YOLO format
        yolo_lines = []
        for pred in filtered_preds:
            yolo_line = convert_to_yolo_format(pred, img_width, img_height)
            yolo_lines.append(yolo_line)
        
        # Save label file
        label_filename = img_path.stem + ".txt"
        
        # Save to dataset labels
        label_path = DATASET_DIR / "labels" / split / label_filename
        label_path.write_text("\n".join(yolo_lines) if yolo_lines else "")
        
        # Save annotated image for review
        annotated_img = img.copy()
        for pred in filtered_preds:
            x = int(pred.get("x", 0))
            y = int(pred.get("y", 0))
            w = int(pred.get("width", 0))
            h = int(pred.get("height", 0))
            conf = float(pred.get("confidence", 0.0))
            
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            color = (0, 255, 0) if conf >= 0.7 else (0, 255, 255)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, f"{conf:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        review_path = AUTO_LABELS_DIR / f"{split}_{img_path.name}"
        cv2.imwrite(str(review_path), annotated_img)
        
        return {
            "success": True,
            "predictions": len(filtered_preds),
            "label_path": str(label_path),
            "review_path": str(review_path),
        }
        
    except Exception as e:
        return {"error": str(e), "predictions": 0}


def main():
    stats = {
        "total": 0,
        "success": 0,
        "errors": 0,
        "with_cracks": 0,
        "no_cracks": 0,
    }
    
    print("\n🤖 Starting auto-labeling...\n")
    
    # Process each split
    for split in ["train", "val", "test"]:
        images_dir = DATASET_DIR / "images" / split
        
        # Get only cracked images (based on filename pattern)
        cracked_prefixes = [f"7{i:03d}" for i in range(69, 92)]
        all_images = list(images_dir.glob("*.jpg"))
        cracked_images = [img for img in all_images 
                         if img.stem.split("-")[0] in cracked_prefixes]
        
        if not cracked_images:
            print(f"⚠️  No cracked images found in {split}/")
            continue
        
        print(f"\n📁 Processing {split}/ ({len(cracked_images)} cracked images)")
        
        for img_path in tqdm(cracked_images, desc=f"  {split}"):
            stats["total"] += 1
            
            result = process_image(img_path, split)
            
            if "error" in result:
                stats["errors"] += 1
                print(f"  ❌ Error on {img_path.name}: {result['error']}")
            else:
                stats["success"] += 1
                if result["predictions"] > 0:
                    stats["with_cracks"] += 1
                else:
                    stats["no_cracks"] += 1
            
            time.sleep(0.1)  # Rate limiting
    
    print("\n" + "="*60)
    print("AUTO-LABELING COMPLETE!")
    print("="*60)
    print(f"Total images:     {stats['total']}")
    print(f"Successfully labeled: {stats['success']}")
    print(f"  - With cracks:  {stats['with_cracks']}")
    print(f"  - No cracks:    {stats['no_cracks']}")
    print(f"Errors:           {stats['errors']}")
    print("\n📁 Review annotated images at:")
    print(f"   {AUTO_LABELS_DIR}")
    print("\n✅ Labels saved to dataset/labels/")
    print("\n⚠️  IMPORTANT: Review the annotated images and correct any mistakes!")
    print("   You can edit the .txt files manually if needed.")
    print("\nYOLO format: class x_center y_center width height (normalized)")
    print("="*60)


if __name__ == "__main__":
    main()
