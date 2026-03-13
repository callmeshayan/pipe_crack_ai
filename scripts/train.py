from pathlib import Path

from ultralytics import YOLO


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "dataset.yaml"

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,
        project=str(project_root / "runs" / "detect"),
        name="train",
    )
