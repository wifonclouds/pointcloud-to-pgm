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
