#!/usr/bin/env python3
"""Convert a point cloud into a simple top-down PGM occupancy image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def point_cloud_to_pgm(
    input_path: Path,
    output_path: Path,
    resolution: float = 0.05,
    min_height: float | None = None,
    max_height: float | None = None,
    padding: float = 0.0,
) -> tuple[int, int]:
    """Project a point cloud onto the XY plane and save a grayscale PGM.

    Occupied cells are black (0), while the background is white (255).
    This is a projection, not a full probabilistic occupancy mapping algorithm.
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

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
    min_xy = xy.min(axis=0) - padding
    max_xy = xy.max(axis=0) + padding

    width = int(np.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1
    height = int(np.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1

    # White background, black projected points.
    image = np.full((height, width), 255, dtype=np.uint8)

    px = np.floor((xy[:, 0] - min_xy[0]) / resolution).astype(np.int64)
    py = np.floor((xy[:, 1] - min_xy[1]) / resolution).astype(np.int64)

    # PGM row 0 is the top of the image, so invert Y for a conventional map view.
    py = height - 1 - py

    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    image[py[valid], px[valid]] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(output_path, format="PPM")

    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input PLY/PCD point cloud")
    parser.add_argument("output", type=Path, help="Output PGM path")
    parser.add_argument("--resolution", type=float, default=0.05, help="Meters per pixel")
    parser.add_argument("--min-height", type=float, default=None, help="Minimum Z value")
    parser.add_argument("--max-height", type=float, default=None, help="Maximum Z value")
    parser.add_argument("--padding", type=float, default=0.0, help="Padding around the point cloud in meters")
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
    )
    print(f"Saved {args.output} ({width} x {height} px)")


if __name__ == "__main__":
    main()
