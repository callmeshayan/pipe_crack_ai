"""
Auto-label positive (cracked) images using the existing Roboflow crack detection model.
Uses the same Roboflow API setup as realtime_pi5_dual_web.py
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = Path("/Users/shayannaghashpour/Desktop/--/Final Year Project/Code/crack_app/.env")

print(f"Loading .env from: {ENV_PATH}")
if not ENV_PATH.exists():
    print(f"❌ ERROR: .env file not found at {ENV_PATH}")
    print("Please ensure your .env file exists with:")
    print("  RF_API_KEY=your_api_key")
    print("  RF_WORKSPACE=your_workspace")
    print("  RF_WORKFLOW_ID=your_workflow_id")
    exit(1)

load_dotenv(ENV_PATH, override=True)

API_KEY = os.getenv("RF_API_KEY", "").strip()
WORKSPACE = os.getenv("RF_WORKSPACE", "").strip()
WORKFLOW_ID = os.getenv("RF_WORKFLOW_ID", "").strip()

if not API_KEY or not WORKSPACE or not WORKFLOW_ID:
    print("❌ ERROR: Missing .env variables!")
    print(f"  RF_API_KEY: {'✓' if API_KEY else '✗'}")
    print(f"  RF_WORKSPACE: {'✓' if WORKSPACE else '✗'}")
    print(f"  RF_WORKFLOW_ID: {'✓' if WORKFLOW_ID else '✗'}")
    exit(1)

print(f"✓ API configured: {WORKSPACE}/{WORKFLOW_ID}")

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)

# Settings (matching realtime_pi5_dual_web.py)
CONF_THRESH = 0.5
MIN_CRACK_AREA = 100
CLASS_NAME = "crack"  # YOLO class name
CLASS_ID = 0  # YOLO class ID (single class)

# Paths
POSITIVE_LIST_FILE = PROJECT_ROOT / "positive_images_list.json"
DATASET_DIR = PROJECT_ROOT / "dataset"


def extract_predictions(result: Any) -> List[Dict[str, Any]]:
    """Extract predictions from nested Roboflow workflow result (from realtime_pi5_dual_web.py)"""
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


def pred_conf(p: Dict[str, Any]) -> float:
    """Get prediction confidence"""
    return float(p.get("confidence", p.get("score", 0.0)) or 0.0)


def calculate_crack_area(pred: Dict[str, Any]) -> float:
    """Calculate crack area in pixels"""
    w = float(pred.get("width", 0))
    h = float(pred.get("height", 0))
    return w * h


def filter_preds(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter predictions by confidence and area"""
    out = []
    for p in preds:
        if pred_conf(p) < CONF_THRESH:
            continue
        if calculate_crack_area(p) < MIN_CRACK_AREA:
            continue
        out.append(p)
    return out


def convert_to_yolo_format(pred: Dict[str, Any], img_width: int, img_height: int) -> str:
    """
    Convert Roboflow prediction to YOLO format.
    
    Roboflow format: x, y (center), width, height in pixels
    YOLO format: class_id x_center y_center width height (normalized 0-1)
    
    Args:
        pred: Prediction dict with x, y, width, height in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        YOLO format string: "0 x_center y_center width height"
    """
    # Get values in pixels (center-based)
    x_center_px = float(pred.get("x", 0))
    y_center_px = float(pred.get("y", 0))
    width_px = float(pred.get("width", 0))
    height_px = float(pred.get("height", 0))
    
    # Normalize to 0-1 range
    x_center_norm = x_center_px / img_width
    y_center_norm = y_center_px / img_height
    width_norm = width_px / img_width
    height_norm = height_px / img_height
    
    # Clamp to valid range [0, 1]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    # YOLO format: class_id x_center y_center width height
    return f"{CLASS_ID} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"


def auto_label_image(image_path: Path) -> tuple[int, str]:
    """
    Auto-label a single image using Roboflow workflow.
    
    Returns:
        (num_detections, label_content) tuple
    """
    # Read image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ⚠️  Could not read image: {image_path.name}")
        return 0, ""
    
    img_height, img_width = img.shape[:2]
    
    # Run inference
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as f:
            # Encode image to jpg
            ok, jpg_data = cv2.imencode(".jpg", img)
            if not ok:
                print(f"  ⚠️  Could not encode image: {image_path.name}")
                return 0, ""
            
            f.write(jpg_data.tobytes())
            f.flush()
            
            # Call Roboflow workflow
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": f.name},
                use_cache=True,
            )
        
        # Extract and filter predictions
        preds = extract_predictions(result)
        preds = filter_preds(preds)
        
        if not preds:
            return 0, ""
        
        # Convert to YOLO format
        yolo_lines = []
        for pred in preds:
            yolo_line = convert_to_yolo_format(pred, img_width, img_height)
            yolo_lines.append(yolo_line)
        
        label_content = "\n".join(yolo_lines)
        return len(preds), label_content
        
    except Exception as e:
        print(f"  ❌ Error processing {image_path.name}: {type(e).__name__}: {e}")
        return 0, ""


def main():
    """Main auto-labeling process"""
    print("=" * 70)
    print("AUTO-LABELING WITH ROBOFLOW CRACK DETECTION MODEL")
    print("=" * 70)
    
    # Load positive images list
    if not POSITIVE_LIST_FILE.exists():
        print(f"❌ ERROR: {POSITIVE_LIST_FILE.name} not found!")
        print("   Run prepare_archive_dataset.py first.")
        return
    
    positive_images = json.loads(POSITIVE_LIST_FILE.read_text())
    print(f"\n✓ Loaded {len(positive_images)} positive images to label")
    print(f"  Confidence threshold: {CONF_THRESH}")
    print(f"  Min crack area: {MIN_CRACK_AREA}px")
    print(f"  Class: {CLASS_NAME} (ID: {CLASS_ID})")
    
    # Statistics
    stats = {
        'total': len(positive_images),
        'labeled': 0,
        'no_detections': 0,
        'errors': 0,
        'total_bboxes': 0,
        'by_split': {'train': 0, 'val': 0, 'test': 0}
    }
    
    print("\n" + "=" * 70)
    print("Starting auto-labeling...")
    print("=" * 70)
    
    # Process each positive image
    for idx, img_info in enumerate(positive_images, 1):
        split = img_info['split']
        filename = img_info['filename']
        img_path = PROJECT_ROOT / img_info['path']
        
        # Get label path
        label_filename = Path(filename).stem + ".txt"
        label_path = DATASET_DIR / "labels" / split / label_filename
        
        # Progress indicator
        progress = (idx / len(positive_images)) * 100
        print(f"\n[{idx}/{len(positive_images)}] ({progress:.1f}%) {split}/{filename}")
        
        if not img_path.exists():
            print(f"  ⚠️  Image not found: {img_path}")
            stats['errors'] += 1
            continue
        
        # Auto-label the image
        num_detections, label_content = auto_label_image(img_path)
        
        if num_detections > 0:
            # Write label file
            label_path.write_text(label_content)
            stats['labeled'] += 1
            stats['total_bboxes'] += num_detections
            stats['by_split'][split] += 1
            print(f"  ✓ Detected {num_detections} crack(s) → {label_filename}")
        else:
            # No detections - image might not have visible cracks
            # Keep empty label file (already exists)
            stats['no_detections'] += 1
            print(f"  ⚠️  No cracks detected (conf<{CONF_THRESH} or area<{MIN_CRACK_AREA}px)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("AUTO-LABELING COMPLETE")
    print("=" * 70)
    print(f"\nTotal images processed:  {stats['total']}")
    print(f"Successfully labeled:     {stats['labeled']} ({(stats['labeled']/stats['total']*100):.1f}%)")
    print(f"No detections:            {stats['no_detections']} ({(stats['no_detections']/stats['total']*100):.1f}%)")
    print(f"Errors:                   {stats['errors']}")
    print(f"\nTotal bounding boxes:     {stats['total_bboxes']}")
    print(f"Average per image:        {stats['total_bboxes']/max(stats['labeled'], 1):.1f}")
    
    print(f"\nBy split:")
    print(f"  Train: {stats['by_split']['train']} images")
    print(f"  Val:   {stats['by_split']['val']} images")
    print(f"  Test:  {stats['by_split']['test']} images")
    
    if stats['no_detections'] > 0:
        print(f"\n⚠️  Note: {stats['no_detections']} images had no detections.")
        print("   These may be false positives in your dataset, or the")
        print("   detection threshold may be too strict. Review manually.")
    
    print("\n✓ Dataset ready for training!")
    print(f"  Location: {DATASET_DIR}")
    print("  Next step: Run scripts/train.py on your RTX 3070 Ti system")
    print("=" * 70)


if __name__ == "__main__":
    main()
