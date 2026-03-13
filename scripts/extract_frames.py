import argparse
from pathlib import Path

import cv2


def extract_every_nth_frame(video_path: str, output_dir: str, every_n: int = 10, prefix: str = "frame") -> None:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            out_file = output_dir / f"{prefix}_{saved:06d}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved} frames to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract every Nth frame from a video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default="frames", help="Output folder for frames")
    parser.add_argument("--every", type=int, default=10, help="Save every Nth frame")
    parser.add_argument("--prefix", default="frame", help="Output image filename prefix")
    args = parser.parse_args()

    if args.every < 1:
        raise ValueError("--every must be >= 1")

    extract_every_nth_frame(args.video, args.output, args.every, args.prefix)
