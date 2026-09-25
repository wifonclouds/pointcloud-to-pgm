# Point Cloud to PGM

Tools for converting 3D point clouds into 2D occupancy/map images in PGM format, with ROS2-compatible YAML metadata.

## Project structure

```text
pointcloud-to-pgm/
├── src/
│   └── cloud_to_pgm.py
├── config/
├── tests/
├── data/
│   ├── input/
│   └── output/
├── .gitignore
└── README.md
```

## Goal

Initial pipeline:

```text
PLY / PCD
   ↓
height filtering
   ↓
XY projection
   ↓
occupancy image
   ↓
PGM + ROS2 YAML
```

The project is intended to evolve toward a more robust LiDAR/SLAM occupancy-map generation pipeline with ground/obstacle classification and configurable processing parameters.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python src/cloud_to_pgm.py data/input/map.ply data/output/map.pgm --resolution 0.05
```

See `python src/cloud_to_pgm.py --help` for all options.


## Polyline section extraction

Extract a point-cloud section around a DXF polyline. The DXF polyline is used exactly as exported; the script does not apply any translation. For a 20 cm-wide section, the default radius is 10 cm and the shortest 3D distance from each point to the polyline must be <= 10 cm.

Usage:

    pip install -r requirements.txt
    python src/extract_polyline_section.py data/input/cloud.ply data/input/profile.dxf data/output/profile_section.ply --width 0.20

If the polyline was moved by +1 m in CloudCompare, export that moved polyline to DXF and pass that DXF to the script. The script uses its stored Z coordinates and does not move it again.
