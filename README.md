# Traffic Counting

A traffic counting application for intersections and roundabouts that detects and tracks vehicles, counts movements between defined zones, and draws vehicle trajectories.

## Features

- **Vehicle Detection** – Uses YOLO (via [Ultralytics](https://github.com/ultralytics/ultralytics)) to detect cars, motorcycles, buses, and trucks.
- **Object Tracking** – Uses `sv.ByteTrack` to assign consistent IDs across frames.
- **Zone-Based Counting** – Define polygon zones for source (entry) and destination (exit) areas via a JSON config file.
- **Source → Destination Statistics** – Counts how many vehicles of each class travelled from every source zone to every destination zone.
- **Trajectory Drawing** – Uses `sv.TraceAnnotator` to draw the path each vehicle has taken.
- **Video Output** – Writes an annotated output video or displays results in a live window.
- **Frame Extraction Helper** – `extract_frame.py` saves the first frame of a video so you can plan zone coordinates with [Roboflow's PolygonZone tool](https://roboflow.github.io/polygonzone/).

## Installation

```bash
pip install supervision ultralytics opencv-python numpy tqdm
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install supervision ultralytics opencv-python numpy tqdm
```

## Usage

### 1. Extract a frame for zone planning

```bash
python extract_frame.py --source path/to/video.mp4 --output frame.png
```

Open `frame.png` in [Roboflow's PolygonZone tool](https://roboflow.github.io/polygonzone/) to draw polygons and copy the coordinates into `zones_config.json`.

### 2. Edit the zone configuration

Edit `zones_config.json` to match your intersection layout (see [Zone Configuration](#zone-configuration) below).

### 3. Run the traffic counter

```bash
# Write annotated output video
python traffic_counter.py --source video.mp4 --zones zones_config.json --output output.mp4

# Live display (no output file)
python traffic_counter.py --source video.mp4 --zones zones_config.json

# Custom model and thresholds
python traffic_counter.py \
    --source video.mp4 \
    --zones zones_config.json \
    --output output.mp4 \
    --weights yolo11n.pt \
    --confidence 0.3 \
    --iou 0.7
```

### CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--source` | ✅ | – | Path to the input video file |
| `--zones` | ✅ | – | Path to the zone configuration JSON file |
| `--output` | ❌ | – | Path to the output video file (omit for live display) |
| `--weights` | ❌ | `yolo11n.pt` | YOLO model weights file |
| `--confidence` | ❌ | `0.3` | Detection confidence threshold |
| `--iou` | ❌ | `0.7` | Detection IOU threshold |

## Zone Configuration

Zones are defined in a JSON file. Each zone is a named polygon expressed as a list of `[x, y]` pixel coordinates.

```json
{
  "source_zones": [
    {"name": "North Entry", "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]},
    {"name": "South Entry", "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}
  ],
  "destination_zones": [
    {"name": "East Exit",  "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]},
    {"name": "West Exit",  "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}
  ]
}
```

- **source_zones** – Areas where vehicles *enter* the scene (entry lanes).
- **destination_zones** – Areas where vehicles *leave* the scene (exit lanes).

See `zones_config.json` for a ready-to-edit example.

## Example Output

The output video shows:
- Colored polygon overlays for each source and destination zone.
- Bounding boxes and class labels for every tracked vehicle.
- Trajectory lines showing each vehicle's path.
- A count table in the top-left corner, e.g.:

```
North Entry → East Exit
  car: 12  truck: 3
South Entry → West Exit
  car: 8   motorcycle: 2
```

## Requirements

- Python 3.8+
- supervision
- ultralytics
- opencv-python
- numpy
- tqdm
