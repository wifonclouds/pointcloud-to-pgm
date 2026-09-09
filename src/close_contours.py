#!/usr/bin/env python3
"""Close gaps in contours of a projected point cloud.

The script projects a point cloud onto the XY plane, rasterizes the points,
and applies morphological closing to connect small gaps in walls/contours.
It does not modify the original point cloud or cloud_to_pgm.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy import ndimage


def project_point_cloud(
    input_path: Path,
    resolution: float,
    min_height: float | None = None,
    max_height: float | None = None,
    padding: float = 0.0,
) -> np.ndarray:
    """Project the point cloud onto an XY binary raster."""
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if padding < 0:
        raise ValueError("padding must be >= 0")

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

    occupied = np.zeros((height, width), dtype=bool)

    px = np.floor((xy[:, 0] - min_xy[0]) / resolution).astype(np.int64)
    py = np.floor((xy[:, 1] - min_xy[1]) / resolution).astype(np.int64)
    py = height - 1 - py

    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    occupied[py[valid], px[valid]] = True

    return occupied


def close_contours(occupied: np.ndarray, radius: int) -> np.ndarray:
    """Close small gaps in contours using binary morphological closing."""
    if radius < 1:
        raise ValueError("radius must be >= 1")

    size = radius * 2 + 1
    structure = np.ones((size, size), dtype=bool)
    return ndimage.binary_closing(occupied, structure=structure)


def save_pgm(occupied: np.ndarray, output_path: Path) -> None:
    """Save the binary contour raster as a black-on-white PGM."""
    image = np.where(occupied, 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(output_path, format="PPM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input PLY/PCD point cloud")
    parser.add_argument("output", type=Path, help="Output PGM path")
    parser.add_argument("--resolution", type=float, default=0.05, help="Meters per pixel")
    parser.add_argument("--radius", type=int, default=2, help="Closing radius in pixels")
    parser.add_argument("--min-height", type=float, default=None, help="Minimum Z value")
    parser.add_argument("--max-height", type=float, default=None, help="Maximum Z value")
    parser.add_argument("--padding", type=float, default=0.0, help="Padding around the point cloud in meters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    occupied = project_point_cloud(
        args.input,
        resolution=args.resolution,
        min_height=args.min_height,
        max_height=args.max_height,
        padding=args.padding,
    )

    closed = close_contours(occupied, radius=args.radius)
    save_pgm(closed, args.output)

    print(f"Saved closed contours to {args.output} ({closed.shape[1]} x {closed.shape[0]} px)")
    print(f"Closing radius: {args.radius} px ({args.radius * args.resolution:.3f} m)")


if __name__ == "__main__":
    main()
