# pipe_crack_ai

YOLO crack-detection training workspace prepared for NVIDIA GPU training (RTX 3070 Ti class) with Ultralytics.

## 1) Create and activate virtual environment

From the project folder:

- **Linux/macOS**
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
- **Windows (PowerShell)**
  - `py -m venv .venv`
  - `.venv\Scripts\Activate.ps1`

## 2) Install dependencies

Install CUDA-enabled PyTorch first (example uses CUDA 12.1 wheels):

- `pip install --upgrade pip`
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- `pip install ultralytics opencv-python numpy matplotlib pandas pyyaml tqdm`

Or use `requirements.txt` for common packages:

- `pip install -r requirements.txt`

> Note: On macOS, CUDA is not available. Use this project on a Windows/Linux machine with NVIDIA drivers + CUDA runtime for GPU training.

## 3) Verify GPU

Run:

- `python scripts/check_gpu.py`

Expected output includes:

- `CUDA available: True`
- GPU model name
- CUDA version

## 4) Prepare your dataset

The project includes a dataset preparation script that organizes images into YOLO format.

**Setup your raw data:**

1. Create folders in the project root:
   - `cracked/` - put all cracked pipe images here
   - `uncracked/` - put all uncracked pipe images here
   - `cracked_labels/` - put YOLO format `.txt` labels for cracked images here

2. Label format (YOLO):
   ```
   class_id x_center y_center width height
   ```
   Example: `0 0.5 0.5 0.3 0.4`

3. Run the preparation script:
   - `python scripts/prepare_dataset.py`

The script will:
- Rename images with consistent naming (`crack_0001.jpg`, `clean_0001.jpg`)
- Create empty labels for uncracked images
- Split data into train (70%), val (20%), test (10%)
- Organize into the `dataset/` structure
- Update `dataset.yaml` with correct paths

## 5) Start training

Run:

- `python scripts/train.py`

Training is configured as:

- model: `yolo11n.pt`
- data: `dataset.yaml`
- imgsz: `640`
- epochs: `100`
- batch: `16`
- device: `0`

Outputs are saved to:

- `runs/detect/train/`

## 6) Run prediction on a video

After training, use best weights:

- `yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=videos/your_video.mp4`

Predictions are typically saved under `runs/detect/predict/`.

## 7) Export for Raspberry Pi deployment

When model quality is acceptable, export to formats useful for edge deployment:

- ONNX: `yolo export model=runs/detect/train/weights/best.pt format=onnx`
- TorchScript: `yolo export model=runs/detect/train/weights/best.pt format=torchscript`

Then benchmark on Raspberry Pi and choose the best latency/accuracy trade-off.
