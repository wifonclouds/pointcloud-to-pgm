#!/usr/bin/env python3
"""Convert a point cloud to a PGM map, optionally on a fixed ROS map grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml
from PIL import Image


def _load_reference_grid(reference_yaml: Path, reference_pgm: Path) -> tuple[float, float, float, int, int]:
    """Return resolution, origin x/y, width and height from a reference map."""
    with reference_yaml.open("r", encoding="utf-8") as stream:
        metadata: dict[str, Any] = yaml.safe_load(stream) or {}

    try:
        resolution = float(metadata["resolution"])
        origin = metadata["origin"]
        origin_x, origin_y = float(origin[0]), float(origin[1])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"Reference YAML must contain numeric 'resolution' and "
            f"'origin: [x, y, yaw]': {reference_yaml}"
        ) from exc

    if resolution <= 0:
        raise ValueError("Reference map resolution must be > 0")

    with Image.open(reference_pgm) as reference_image:
        width, height = reference_image.size

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid reference PGM dimensions: {reference_pgm}")

    return resolution, origin_x, origin_y, width, height


def point_cloud_to_pgm(
    input_path: Path,
    output_path: Path,
    resolution: float = 0.05,
    min_height: float | None = None,
    max_height: float | None = None,
    padding: float = 0.0,
    reference_yaml: Path | None = None,
    reference_pgm: Path | None = None,
) -> tuple[int, int]:
    """Project a point cloud onto XY and save a grayscale PGM.

    If both reference files are supplied, the output uses the reference map's
    exact resolution, origin and image dimensions. Points outside that fixed
    map extent are clipped rather than shifting/resizing the raster.

    Occupied cells are black (0); the background is white (255).
    """
    if (reference_yaml is None) != (reference_pgm is None):
        raise ValueError("--reference-yaml and --reference-pgm must be used together")
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if padding < 0:
        raise ValueError("padding must be >= 0")
    if reference_yaml is not None and padding != 0:
        raise ValueError("--padding cannot be used with a reference map; the grid extent is fixed")

    cloud = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(cloud.points)

    if points.size == 0:
        raise ValueError(f"Point cloud is empty: {input_path}")

    if min_height is not None:
        points = points[points[:, 2] >= min_height]
    if max_height is not None:
        points = points[points[:, 2] <= max_height]

    if len(points) == 0:
        raise ValueError("No points remain after height filtering")

    xy = points[:, :2]

    if reference_yaml is not None and reference_pgm is not None:
        grid_resolution, origin_x, origin_y, width, height = _load_reference_grid(
            reference_yaml, reference_pgm
        )
        # ROS map origin is the bottom-left corner in world coordinates.
        # Image row zero is the top, so invert the Y pixel index after mapping.
        px = np.floor((xy[:, 0] - origin_x) / grid_resolution).astype(np.int64)
        y_from_bottom = np.floor((xy[:, 1] - origin_y) / grid_resolution).astype(np.int64)
        py = height - 1 - y_from_bottom
    else:
        min_xy = xy.min(axis=0) - padding
        max_xy = xy.max(axis=0) + padding
        width = int(np.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1
        height = int(np.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1
        px = np.floor((xy[:, 0] - min_xy[0]) / resolution).astype(np.int64)
        y_from_bottom = np.floor((xy[:, 1] - min_xy[1]) / resolution).astype(np.int64)
        py = height - 1 - y_from_bottom

    image = np.full((height, width), 255, dtype=np.uint8)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    image[py[valid], px[valid]] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(output_path, format="PPM")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input PLY/PCD point cloud")
    parser.add_argument("output", type=Path, help="Output PGM path")
    parser.add_argument("--resolution", type=float, default=0.05, help="Meters per pixel (ignored when using a reference map)")
    parser.add_argument("--min-height", type=float, default=None, help="Minimum Z value")
    parser.add_argument("--max-height", type=float, default=None, help="Maximum Z value")
    parser.add_argument("--padding", type=float, default=0.0, help="Padding around the point cloud in meters (not available with a reference map)")
    parser.add_argument("--reference-yaml", type=Path, default=None, help="Existing ROS map YAML; locks origin and resolution")
    parser.add_argument("--reference-pgm", type=Path, default=None, help="Existing ROS map PGM; locks output dimensions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width, height = point_cloud_to_pgm(
        args.input,
        args.output,
        resolution=args.resolution,
        min_height=args.min_height,
        max_height=args.max_height,
        padding=args.padding,
        reference_yaml=args.reference_yaml,
        reference_pgm=args.reference_pgm,
    )
    if args.reference_yaml is not None:
        grid_resolution, origin_x, origin_y, _, _ = _load_reference_grid(
            args.reference_yaml, args.reference_pgm
        )
        print(
            f"Reference grid: resolution={grid_resolution}, "
            f"origin=({origin_x}, {origin_y})"
        )
    print(f"Saved {args.output} ({width} x {height} px)")


if __name__ == "__main__":
    main()
