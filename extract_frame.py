"""
extract_frame.py – Extract the first frame from a video file.

Use the saved image with Roboflow's PolygonZone tool
(https://roboflow.github.io/polygonzone/) to plan zone coordinates, then paste
the resulting polygons into your zones_config.json file.

Usage
-----
    python extract_frame.py --source path/to/video.mp4
    python extract_frame.py --source path/to/video.mp4 --output frame.png --frame 30
"""

import argparse

import cv2


def extract_frame(source: str, output: str, frame_index: int = 0) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= total_frames:
        raise ValueError(
            f"Frame index {frame_index} is out of range "
            f"(video has {total_frames} frames)."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_index} from {source}.")

    cv2.imwrite(output, frame)
    height, width = frame.shape[:2]
    print(f"Saved frame {frame_index} ({width}x{height}) to: {output}")
    print(
        "Open the image in https://roboflow.github.io/polygonzone/ "
        "to define your zone polygons."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a frame from a video for zone coordinate planning."
    )
    parser.add_argument(
        "--source", required=True, help="Path to the input video file."
    )
    parser.add_argument(
        "--output",
        default="frame.png",
        help="Path for the output PNG image (default: frame.png).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Zero-based frame index to extract (default: 0).",
    )
    args = parser.parse_args()
    extract_frame(args.source, args.output, args.frame)


if __name__ == "__main__":
    main()
