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
PGM + ROS2 YAML metadata
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage: automatic bounds

Without reference files, the output bounds are derived from the current cloud. This mode is useful for standalone maps, but different clouds may produce different origins and dimensions.

```bash
python src/cloud_to_pgm.py data/input/map.ply data/output/map.pgm --resolution 0.05
```

## Usage: align output to an existing ROS map

To generate a PGM on the exact same raster grid as an existing map, pass BOTH its YAML and PGM:

```bash
python src/cloud_to_pgm.py data/input/new_cloud.ply data/output/new_map.pgm \
  --reference-yaml data/input/map.yaml \
  --reference-pgm data/input/map.pgm \
  --min-height 0.1 --max-height 2.0
```

In reference mode the script reads `resolution` and `origin` from the reference YAML, and the raster width/height from the reference PGM. Each point is projected using those fixed world-coordinate bounds. Points outside the reference map extent are clipped; they do not expand the image or shift the origin. Do not pass `--padding` in this mode.

**Important:** this guarantees identical grid geometry, not automatic registration of two different point clouds. The new cloud must already be in the same world/map coordinate frame as the reference map. If the cloud has a translation, rotation, or SLAM drift relative to the old map, register/transform the cloud first. Keep the original YAML as the metadata for the generated PGM because the output uses the reference grid's origin and resolution.

The script writes only the PGM. It does not create or modify the YAML.

See `python src/cloud_to_pgm.py --help` for all options.

## Polyline section extraction

Extract a point-cloud section around a DXF polyline. The DXF polyline is used exactly as exported; the script does not apply any translation. For a 20 cm-wide section, the default radius is 10 cm and the shortest 3D distance from each point to the polyline must be <= 10 cm.

Usage:

    pip install -r requirements.txt
    python src/extract_polyline_section.py data/input/cloud.ply data/input/profile.dxf data/output/profile_section.ply --width 0.20

If the polyline was moved by +1 m in CloudCompare, export that moved polyline to DXF and pass that DXF to the script. The script uses its stored Z coordinates and does not move it again.
