#!/usr/bin/env python3
"""Extract a 20 cm thick planar section from a point cloud.

The DXF polyline defines the profile line that lies in the section plane.
The script fits a plane to that 3D polyline and keeps ALL point-cloud
points whose perpendicular distance to that plane is <= width / 2.

For the default width of 0.20 m, the resulting section extends 20 cm
from the profile plane in the positive plane-normal direction.

The DXF polyline is used exactly as stored. No translation is applied.

LAS/LAZ input is handled with laspy and keeps the original LAS point
attributes in the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ezdxf
import laspy
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

    if len(points) < 3:
        raise ValueError("At least three polyline vertices are required to define a section plane.")

    return np.asarray(points, dtype=np.float64)


def fit_section_plane(polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a plane to the 3D polyline using PCA.

    Returns:
        origin: point on the plane (polyline centroid)
        normal: unit normal vector
    """
    origin = polyline.mean(axis=0)
    centered = polyline - origin

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

    if singular_values[1] < 1e-10:
        raise ValueError(
            "Polyline is effectively a straight line, so it does not uniquely "
            "define a section plane."
        )

    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    # Make the normal point upward when the fitted plane has a Z component.
    # This removes the arbitrary sign returned by SVD.
    if normal[2] < 0:
        normal = -normal

    return origin, normal


def points_within_plane_thickness(
    points: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    width: float,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    """Return points from the plane to one side, up to the requested thickness."""
    mask = np.zeros(len(points), dtype=bool)

    for begin in range(0, len(points), chunk_size):
        end = min(begin + chunk_size, len(points))
        chunk = points[begin:end]
        signed_distance = (chunk - plane_origin) @ plane_normal
        mask[begin:end] = (signed_distance >= 0.0) & (signed_distance <= width)

    return mask


def read_cloud(input_path: Path) -> tuple[np.ndarray, object, str]:
    """Read LAS/LAZ with laspy or PLY/PCD with Open3D."""
    suffix = input_path.suffix.lower()

    if suffix in {".las", ".laz"}:
        las = laspy.read(str(input_path))
        points = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        return points, las, "las"

    cloud = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(cloud.points)

    if len(points) == 0:
        raise ValueError(f"Point cloud is empty: {input_path}")

    return points, cloud, "open3d"


def write_cloud(
    output_path: Path,
    source: object,
    source_type: str,
    mask: np.ndarray,
) -> None:
    """Write selected points while preserving LAS attributes when possible."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source_type == "las":
        if output_path.suffix.lower() not in {".las", ".laz"}:
            raise ValueError("LAS/LAZ input requires a .las or .laz output.")

        output = laspy.LasData(source.header)
        output.points = source.points[mask].copy()
        output.write(str(output_path))
        return

    section = o3d.geometry.PointCloud()
    points = np.asarray(source.points)
    section.points = o3d.utility.Vector3dVector(points[mask])

    colors = np.asarray(source.colors)
    if len(colors) == len(points):
        section.colors = o3d.utility.Vector3dVector(colors[mask])

    if not o3d.io.write_point_cloud(str(output_path), section):
        raise RuntimeError(f"Failed to write output point cloud: {output_path}")


def extract_section(
    input_path: Path,
    dxf_path: Path,
    output_path: Path,
    width: float,
    chunk_size: int = 1_000_000,
) -> tuple[int, int]:
    """Extract and save a planar point-cloud section."""
    if width <= 0:
        raise ValueError("width must be > 0")

    polyline = load_polyline(dxf_path)
    plane_origin, plane_normal = fit_section_plane(polyline)

    print(f"Input cloud: {input_path}")
    print(f"Polyline: {dxf_path}")
    print(f"Polyline vertices: {len(polyline)}")
    print(
        "Polyline bounds: "
        f"X [{polyline[:, 0].min():.3f}, {polyline[:, 0].max():.3f}], "
        f"Y [{polyline[:, 1].min():.3f}, {polyline[:, 1].max():.3f}], "
        f"Z [{polyline[:, 2].min():.3f}, {polyline[:, 2].max():.3f}]"
    )
    print(
        "Section plane origin: "
        f"[{plane_origin[0]:.3f}, {plane_origin[1]:.3f}, {plane_origin[2]:.3f}]"
    )
    print(
        "Section plane normal: "
        f"[{plane_normal[0]:.6f}, {plane_normal[1]:.6f}, {plane_normal[2]:.6f}]"
    )
    print(f"Section thickness above profile: {width:.3f} m")

    points, source, source_type = read_cloud(input_path)
    print(f"Input points: {len(points):,}")

    mask = points_within_plane_thickness(
        points,
        plane_origin,
        plane_normal,
        width,
        chunk_size=chunk_size,
    )

    selected = int(np.count_nonzero(mask))
    print(f"Selected points: {selected:,}")
    print(f"Rejected points: {len(points) - selected:,}")

    if selected == 0:
        raise ValueError(
            "No points were selected. Check that the DXF and point cloud "
            "use the same coordinate system."
        )

    write_cloud(output_path, source, source_type, mask)
    print(f"Saved: {output_path}")
    return len(points), selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input LAS/LAZ/PLY/PCD point cloud")
    parser.add_argument("polyline", type=Path, help="DXF polyline defining the section plane")
    parser.add_argument(
        "output",
        type=Path,
        help="Output LAS/LAZ for LAS/LAZ input, or PLY/PCD for Open3D input",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.20,
        help="Total section thickness in meters (default: 0.20)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
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
