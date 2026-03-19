"""
traffic_counter.py – Count vehicles at intersections / roundabouts.

Detects and tracks vehicles across polygon zones, counts source-to-destination
movements per vehicle class, and annotates the output video with trajectories,
bounding boxes, and a statistics overlay.

Usage
-----
    python traffic_counter.py --source video.mp4 --zones zones_config.json
    python traffic_counter.py --source video.mp4 --zones zones_config.json --output out.mp4
    python traffic_counter.py --source video.mp4 --zones zones_config.json \\
        --weights yolo11n.pt --confidence 0.3 --iou 0.7
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# COCO class IDs for vehicles
# ---------------------------------------------------------------------------
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Colors for source / destination zone overlays (BGR)
SOURCE_ZONE_COLOUR = sv.Color(r=0, g=128, b=255)      # blue
DESTINATION_ZONE_COLOUR = sv.Color(r=0, g=200, b=0)   # green


# ---------------------------------------------------------------------------
# DetectionsManager
# ---------------------------------------------------------------------------
class DetectionsManager:
    """Tracks which source zone each tracker ID first appeared in and records
    source→destination pairs when the vehicle enters a destination zone."""

    def __init__(self) -> None:
        # tracker_id -> source zone index
        self.tracker_id_to_source: Dict[int, int] = {}
        # (source_idx, dest_idx) -> {class_name -> count}
        self.counts: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # (src_idx, dst_idx, tracker_id) keys already recorded – prevents a
        # vehicle that stays in a destination zone across multiple frames from
        # being counted more than once.
        self._counted: set = set()

    def update(
        self,
        detections_in_zones: List[sv.Detections],
        source_detections: List[sv.Detections],
    ) -> None:
        """
        Parameters
        ----------
        detections_in_zones:
            One ``sv.Detections`` per destination zone containing only the
            detections that are currently inside that zone.
        source_detections:
            One ``sv.Detections`` per source zone containing only the
            detections that are currently inside that zone.
        """
        # Register new tracker IDs for source zones
        for src_idx, src_dets in enumerate(source_detections):
            for tid in src_dets.tracker_id:
                if tid not in self.tracker_id_to_source:
                    self.tracker_id_to_source[tid] = src_idx

        # When a known tracker ID appears in a destination zone, record the
        # source→destination pair (once per unique tracker ID per pair).
        for dst_idx, dst_dets in enumerate(detections_in_zones):
            for tid, class_id in zip(
                dst_dets.tracker_id, dst_dets.class_id
            ):
                if tid in self.tracker_id_to_source:
                    src_idx = self.tracker_id_to_source[tid]
                    key = (src_idx, dst_idx, tid)
                    if key not in self._counted:
                        self._counted.add(key)
                        class_name = CLASS_NAMES.get(int(class_id), "vehicle")
                        self.counts[(src_idx, dst_idx)][class_name] += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_zones_config(path: str) -> Tuple[List[dict], List[dict]]:
    with open(path, "r") as f:
        config = json.load(f)
    return config["source_zones"], config["destination_zones"]


def build_polygon_zones(
    zone_configs: List[dict],
    frame_resolution_wh: Tuple[int, int],
) -> List[sv.PolygonZone]:
    zones = []
    for zc in zone_configs:
        polygon = np.array(zc["polygon"], dtype=np.int32)
        zone = sv.PolygonZone(polygon=polygon)
        zones.append(zone)
    return zones


def draw_overlay_text(
    frame: np.ndarray,
    source_names: List[str],
    destination_names: List[str],
    counts: Dict[Tuple[int, int], Dict[str, int]],
) -> np.ndarray:
    """Draw a semi-transparent statistics box in the top-left corner."""
    lines: List[str] = []
    for (src_idx, dst_idx), class_counts in sorted(counts.items()):
        src_name = source_names[src_idx] if src_idx < len(source_names) else f"Source {src_idx}"
        dst_name = destination_names[dst_idx] if dst_idx < len(destination_names) else f"Dest {dst_idx}"
        lines.append(f"{src_name} -> {dst_name}")
        for cls_name, cnt in sorted(class_counts.items()):
            lines.append(f"  {cls_name}: {cnt}")

    if not lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    padding = 8
    line_height = 20

    text_w = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines
    )
    box_w = text_w + padding * 2
    box_h = len(lines) * line_height + padding * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(lines):
        y = padding + (i + 1) * line_height
        color = (255, 255, 255) if not line.startswith("  ") else (200, 200, 200)
        cv2.putText(frame, line, (padding, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def process_video(
    source_path: str,
    zones_path: str,
    output_path: Optional[str],
    weights: str,
    confidence: float,
    iou: float,
) -> None:
    # --- Load zone config ------------------------------------------------
    source_zone_configs, dest_zone_configs = load_zones_config(zones_path)
    source_names = [z["name"] for z in source_zone_configs]
    dest_names = [z["name"] for z in dest_zone_configs]

    # --- Video info -------------------------------------------------------
    video_info = sv.VideoInfo.from_video_path(source_path)
    frame_wh = (video_info.width, video_info.height)

    # --- Build polygon zones ---------------------------------------------
    source_zones = build_polygon_zones(source_zone_configs, frame_wh)
    dest_zones = build_polygon_zones(dest_zone_configs, frame_wh)

    # --- Model & tracker -------------------------------------------------
    model = YOLO(weights)
    tracker = sv.ByteTrack(frame_rate=video_info.fps)

    # --- Annotators -------------------------------------------------------
    trace_annotator = sv.TraceAnnotator(
        color=sv.ColorPalette.DEFAULT,
        thickness=2,
        trace_length=50,
    )
    box_annotator = sv.BoxAnnotator(
        color=sv.ColorPalette.DEFAULT,
        thickness=2,
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.DEFAULT,
        text_scale=0.5,
    )
    # Create one annotator per zone
    source_zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=SOURCE_ZONE_COLOUR,
            thickness=2,
            text_scale=0.6,
        )
        for zone in source_zones
    ]
    dest_zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=DESTINATION_ZONE_COLOUR,
            thickness=2,
            text_scale=0.6,
        )
        for zone in dest_zones
    ]

    detections_manager = DetectionsManager()
    frames_generator = sv.get_video_frames_generator(source_path)
    total_frames = video_info.total_frames

    def process_frame(frame: np.ndarray) -> np.ndarray:
        # Detection
        results = model(
            frame,
            verbose=False,
            conf=confidence,
            iou=iou,
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter to vehicle classes only
        mask = np.isin(detections.class_id, VEHICLE_CLASS_IDS)
        detections = detections[mask]

        # Tracking
        detections = tracker.update_with_detections(detections)

        # Zone membership
        src_dets_per_zone: List[sv.Detections] = []
        for zone in source_zones:
            in_zone = zone.trigger(detections)
            src_dets_per_zone.append(detections[in_zone])

        dst_dets_per_zone: List[sv.Detections] = []
        for zone in dest_zones:
            in_zone = zone.trigger(detections)
            dst_dets_per_zone.append(detections[in_zone])

        # Update counts
        detections_manager.update(dst_dets_per_zone, src_dets_per_zone)

        # --- Annotation ---------------------------------------------------
        # Trajectories
        frame = trace_annotator.annotate(scene=frame, detections=detections)

        # Bounding boxes + labels
        labels = [
            f"#{tid} {CLASS_NAMES.get(int(cid), 'vehicle')}"
            for tid, cid in zip(detections.tracker_id or [], detections.class_id or [])
        ]
        frame = box_annotator.annotate(scene=frame, detections=detections)
        if labels:
            frame = label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )

        # Source zones (blue)
        for annotator, zone_cfg in zip(source_zone_annotators, source_zone_configs):
            frame = annotator.annotate(scene=frame, label=zone_cfg["name"])

        # Destination zones (green)
        for annotator, zone_cfg in zip(dest_zone_annotators, dest_zone_configs):
            frame = annotator.annotate(scene=frame, label=zone_cfg["name"])

        # Statistics overlay
        frame = draw_overlay_text(
            frame,
            source_names,
            dest_names,
            detections_manager.counts,
        )

        return frame

    # --- Video sink / display ---------------------------------------------
    if output_path:
        with sv.VideoSink(output_path, video_info) as sink:
            for frame in tqdm(
                frames_generator,
                total=total_frames,
                desc="Processing",
                unit="frame",
            ):
                annotated = process_frame(frame)
                sink.write_frame(annotated)
        print(f"Output saved to: {output_path}")
    else:
        print("Press 'q' to quit live display.")
        for frame in tqdm(
            frames_generator,
            total=total_frames,
            desc="Processing",
            unit="frame",
        ):
            annotated = process_frame(frame)
            cv2.imshow("Traffic Counter", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count vehicles at intersections / roundabouts."
    )
    parser.add_argument("--source", required=True, help="Input video file path.")
    parser.add_argument(
        "--zones", required=True, help="Zone configuration JSON file path."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video file path (omit for live display).",
    )
    parser.add_argument(
        "--weights",
        default="yolo11n.pt",
        help="YOLO model weights (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="Detection IOU threshold (default: 0.7).",
    )
    args = parser.parse_args()

    if not Path(args.source).is_file():
        parser.error(f"Source video not found: {args.source}")
    if not Path(args.zones).is_file():
        parser.error(f"Zones config not found: {args.zones}")

    process_video(
        source_path=args.source,
        zones_path=args.zones,
        output_path=args.output,
        weights=args.weights,
        confidence=args.confidence,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
