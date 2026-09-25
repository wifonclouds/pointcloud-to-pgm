#!/usr/bin/env python3
"""Extract a 3D point-cloud section around a DXF polyline.

The DXF polyline is used exactly as stored. No translation or other
transformation is applied by this script.

For a section width of 0.20 m, the default radius is 0.10 m. A point is
kept when its shortest 3D distance to any polyline segment is <= radius.
This creates a 20 cm diameter tube around the polyline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ezdxf
import numpy as np
import open3d as o3d


def load_polyline(dxf_path: Path) -> np.ndarray:
    """Load the first supported DXF polyline as Nx3 coordinates."""
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    lwpolylines = list(msp.query("LWPOLYLINE"))
    if lwpolylines:
        entity = lwpolylines[0]
        elevation = float(entity.dxf.elevation.z)
        points = [(float(x), float(y), elevation) for x, y, *_ in entity.get_points()]
    else:
        polylines = list(msp.query("POLYLINE"))
        if not polylines:
            raise ValueError(f"No LWPOLYLINE or POLYLINE found in {dxf_path}")

        entity = polylines[0]
        points = [
            (
                float(vertex.dxf.location.x),
                float(vertex.dxf.location.y),
                float(vertex.dxf.location.z),
            )
            for vertex in entity.vertices
        ]

    if len(points) < 2:
        raise ValueError("Polyline must contain at least two vertices.")

    polyline = np.asarray(points, dtype=np.float64)

    # CloudCompare's horizontal section workflow is expected here: the
    # polyline should already have the desired Z elevation in the DXF.
    z_range = np.ptp(polyline[:, 2])
    if z_range > 1e-4:
        print(
            f"Warning: polyline is not horizontal (Z range: {z_range:.6f} m). "
            "Using the full 3D distance to its segments."
        )

    return polyline


def points_within_polyline_radius(
    points: np.ndarray,
    polyline: np.ndarray,
    radius: float,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Return a boolean mask for points within 3D distance of the polyline."""
    if radius <= 0:
        raise ValueError("radius must be > 0")

    starts = polyline[:-1]
    ends = polyline[1:]
    vectors = ends - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)

    if np.any(lengths_sq <= 0):
        valid_segments = lengths_sq > 0
        starts = starts[valid_segments]
        vectors = vectors[valid_segments]
        lengths_sq = lengths_sq[valid_segments]

    if len(starts) == 0:
        raise ValueError("Polyline contains no non-zero-length segments.")

    mask = np.zeros(len(points), dtype=bool)
    radius_sq = radius * radius

    for begin in range(0, len(points), chunk_size):
        end = min(begin + chunk_size, len(points))
        chunk = points[begin:end]

        # Point-to-segment distance:
        # p_projection = a + clamp(dot(p-a, v) / |v|^2, 0, 1) * v
        delta = chunk[:, None, :] - starts[None, :, :]
        t = np.einsum("nsi,si->ns", delta, vectors) / lengths_sq[None, :]
        t = np.clip(t, 0.0, 1.0)

        closest = starts[None, :, :] + t[:, :, None] * vectors[None, :, :]
        diff = chunk[:, None, :] - closest
        distance_sq = np.einsum("nsi,nsi->ns", diff, diff)

        mask[begin:end] = np.min(distance_sq, axis=1) <= radius_sq

    return mask


def extract_section(
    input_path: Path,
    dxf_path: Path,
    output_path: Path,
    width: float,
    chunk_size: int = 100_000,
) -> tuple[int, int]:
    """Extract and save the point-cloud section."""
    if width <= 0:
        raise ValueError("width must be > 0")

    radius = width / 2.0
    polyline = load_polyline(dxf_path)

    print(f"Input cloud: {input_path}")
    print(f"Polyline: {dxf_path}")
    print(f"Polyline vertices: {len(polyline)}")
    print(
        "Polyline bounds: "
        f"X [{polyline[:, 0].min():.3f}, {polyline[:, 0].max():.3f}], "
        f"Y [{polyline[:, 1].min():.3f}, {polyline[:, 1].max():.3f}], "
        f"Z [{polyline[:, 2].min():.3f}, {polyline[:, 2].max():.3f}]"
    )
    print(f"Section width: {width:.3f} m")
    print(f"Section radius: {radius:.3f} m")

    cloud = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(cloud.points)

    if len(points) == 0:
        raise ValueError(f"Point cloud is empty: {input_path}")

    print(f"Input points: {len(points):,}")

    mask = points_within_polyline_radius(
        points,
        polyline,
        radius,
        chunk_size=chunk_size,
    )

    selected = int(np.count_nonzero(mask))
    print(f"Selected points: {selected:,}")
    print(f"Rejected points: {len(points) - selected:,}")

    if selected == 0:
        raise ValueError(
            "No points were selected. Check that the DXF and point cloud "
            "use the same coordinate system and that the polyline was "
            "exported after the desired +Z translation."
        )

    section = o3d.geometry.PointCloud()
    section.points = o3d.utility.Vector3dVector(points[mask])

    colors = np.asarray(cloud.colors)
    if len(colors) == len(points):
        section.colors = o3d.utility.Vector3dVector(colors[mask])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_path), section):
        raise RuntimeError(f"Failed to write output point cloud: {output_path}")

    print(f"Saved: {output_path}")
    return len(points), selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input PLY/PCD point cloud")
    parser.add_argument("polyline", type=Path, help="DXF polyline")
    parser.add_argument("output", type=Path, help="Output PLY/PCD point cloud")
    parser.add_argument(
        "--width",
        type=float,
        default=0.20,
        help="Total section width in meters (default: 0.20)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of cloud points processed per chunk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_section(
        args.input,
        args.polyline,
        args.output,
        width=args.width,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
