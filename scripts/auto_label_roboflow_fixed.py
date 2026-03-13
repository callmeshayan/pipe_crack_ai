"""
Auto-label positive (cracked) images using the existing Roboflow crack detection workflow.
This version replicates the InferenceHTTPClient behavior without requiring inference-sdk.
Compatible with Python 3.14+
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import cv2
import requests
from dotenv import load_dotenv

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

if not all([API_KEY, WORKSPACE, WORKFLOW_ID]):
    print("❌ ERROR: Missing required environment variables")
    print(f"RF_API_KEY: {'✓' if API_KEY else '✗'}")
    print(f"RF_WORKSPACE: {'✓' if WORKSPACE else '✗'}")
    print(f"RF_WORKFLOW_ID: {'✓' if WORKFLOW_ID else '✗'}")
    exit(1)

print(f"✓ Loaded environment variables")
print(f"  Workspace: {WORKSPACE}")
print(f"  Workflow ID: {WORKFLOW_ID}")
print(f"  API URL: https://detect.roboflow.com")

# Configuration
CONFIDENCE_THRESHOLD = 0.5
MIN_CRACK_AREA = 100  # minimum area in pixels to filter out noise

# Paths
DATASET_ROOT = PROJECT_ROOT / "dataset"
POSITIVE_IMAGES_JSON = PROJECT_ROOT / "positive_images_list.json"


def run_workflow_inference(image_path: str) -> Dict[str, Any]:
    """
    Run Roboflow workflow inference on an image.
    Uses the hosted inference API with workflow execution.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Encode image to JPEG bytes
    ok, jpg_buffer = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError("Failed to encode image")
    
    image_bytes = jpg_buffer.tobytes()
    
    # Construct the API URL for workflow execution
    # Format: POST to /<workspace_id>/<workflow_id>
    api_url = f"https://detect.roboflow.com/{WORKSPACE}/{WORKFLOW_ID}"
    
    params = {
        'api_key': API_KEY
    }
    
    # Send image as multipart/form-data
    files = {
        'file': ('image.jpg', image_bytes, 'image/jpeg')
    }
    
    # Make the request
    response = requests.post(
        api_url,
        files=files,
        params=params,
        timeout=30
    )
    
    response.raise_for_status()
    return response.json()


def extract_predictions(result: Dict[str, Any]) -> List[Dict]:
    """
    Extract crack predictions from workflow result.
    Based on the extract_predictions function from realtime_pi5_dual_web.py
    """
    predictions = []
    
    try:
        # Navigate through the nested structure
        if 'outputs' in result:
            for output in result['outputs']:
                if isinstance(output, dict) and 'predictions' in output:
                    for pred in output['predictions']:
                        if pred.get('class') == 'crack':
                            x = pred.get('x', 0)
                            y = pred.get('y', 0)
                            width = pred.get('width', 0)
                            height = pred.get('height', 0)
                            confidence = pred.get('confidence', 0)
                            
                            # Filter by confidence and area
                            area = width * height
                            if confidence >= CONFIDENCE_THRESHOLD and area >= MIN_CRACK_AREA:
                                predictions.append({
                                    'x': x,
                                    'y': y,
                                    'width': width,
                                    'height': height,
                                    'confidence': confidence,
                                    'class': 'crack'
                                })
    except Exception as e:
        print(f"  Warning: Error extracting predictions: {e}")
    
    return predictions


def convert_to_yolo_format(predictions: List[Dict], image_width: int, image_height: int) -> List[str]:
    """
    Convert Roboflow predictions to YOLO format.
    
    Roboflow format: x, y, width, height (pixels, center-based)
    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    
    For crack detection: class_id = 0 (only one class)
    """
    yolo_labels = []
    
    for pred in predictions:
        # Roboflow already provides center coordinates
        x_center = pred['x']
        y_center = pred['y']
        width = pred['width']
        height = pred['height']
        
        # Normalize to 0-1 range
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = width / image_width
        height_norm = height / image_height
        
        # Clamp to valid range [0, 1]
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))
        
        # YOLO format: class_id x_center y_center width height
        yolo_label = f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
        yolo_labels.append(yolo_label)
    
    return yolo_labels


def auto_label_image(image_path: str) -> List[str]:
    """
    Auto-label a single image using the Roboflow workflow.
    Returns list of YOLO format label strings.
    """
    # Load image to get dimensions
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ❌ Error: Could not read image {image_path}")
        return []
    
    image_height, image_width = image.shape[:2]
    
    # Run inference
    try:
        result = run_workflow_inference(str(image_path))
        
        # Extract predictions
        predictions = extract_predictions(result)
        
        if not predictions:
            return []
        
        # Convert to YOLO format
        yolo_labels = convert_to_yolo_format(predictions, image_width, image_height)
        
        return yolo_labels
        
    except Exception as e:
        print(f"  ❌ Error processing image: {e}")
        return []


def main():
    """Main function to auto-label all positive images."""
    print("\n" + "="*80)
    print("AUTO-LABELING WITH ROBOFLOW CRACK DETECTION WORKFLOW")
    print("="*80 + "\n")
    
    # Load positive images list
    if not POSITIVE_IMAGES_JSON.exists():
        print(f"❌ ERROR: {POSITIVE_IMAGES_JSON} not found")
        print("Run prepare_archive_dataset.py first to generate this file.")
        exit(1)
    
    with open(POSITIVE_IMAGES_JSON, 'r') as f:
        positive_images = json.load(f)
    
    print(f"Found {len(positive_images)} positive images to label\n")
    
    # Test with first image to verify API connection
    print("Testing API connection with first image...")
    test_img = positive_images[0]
    test_path = DATASET_ROOT / "images" / test_img['split'] / test_img['filename']
    
    try:
        test_result = run_workflow_inference(str(test_path))
        print(f"✓ API connection successful!")
        print(f"  Response structure: {list(test_result.keys())}")
    except Exception as e:
        print(f"❌ API test failed: {e}")
        print("\nPlease check:")
        print("  1. Your API key is valid")
        print("  2. Your workspace name is correct")
        print("  3. Your workflow ID is correct")
        print("  4. You have internet connection")
        exit(1)
    
    print("\nStarting batch labeling...\n")
    
    # Statistics
    total_images = len(positive_images)
    processed = 0
    labeled = 0
    skipped = 0
    errors = 0
    
    # Process each positive image
    for i, image_info in enumerate(positive_images, 1):
        split = image_info['split']
        filename = image_info['filename']
        
        image_path = DATASET_ROOT / "images" / split / filename
        label_path = DATASET_ROOT / "labels" / split / f"{Path(filename).stem}.txt"
        
        print(f"[{i}/{total_images}] Processing {split}/{filename}...")
        
        if not image_path.exists():
            print(f"  ⚠️  Warning: Image not found, skipping")
            skipped += 1
            continue
        
        # Auto-label the image
        yolo_labels = auto_label_image(image_path)
        
        if yolo_labels:
            # Write labels to file
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels) + '\n')
            print(f"  ✓ Labeled with {len(yolo_labels)} crack detection(s)")
            labeled += 1
        else:
            # No detections - write empty label file (image exists but no cracks detected)
            with open(label_path, 'w') as f:
                f.write('')
            print(f"  ⚠️  No cracks detected")
        
        processed += 1
        
        # Progress update every 50 images
        if i % 50 == 0:
            print(f"\n--- Progress: {i}/{total_images} ({i/total_images*100:.1f}%) ---")
            print(f"    Labeled: {labeled}, No detections: {processed-labeled}\n")
    
    # Final statistics
    print("\n" + "="*80)
    print("AUTO-LABELING COMPLETE")
    print("="*80)
    print(f"Total images: {total_images}")
    print(f"Processed: {processed}")
    print(f"Successfully labeled: {labeled} ({labeled/total_images*100:.1f}%)")
    print(f"No detections: {processed - labeled}")
    print(f"Skipped (not found): {skipped}")
    print(f"Errors: {errors}")
    print("\nNext steps:")
    print("1. Review some labeled images to verify quality")
    print("2. Commit the dataset to GitHub: git add dataset/labels && git commit -m 'Add auto-generated labels'")
    print("3. Clone the repo on your RTX 3070 Ti system")
    print("4. Run scripts/train.py to start training")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
