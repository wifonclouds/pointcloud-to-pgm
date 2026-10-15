from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import yaml
from PIL import Image

from src.cloud_to_pgm import point_cloud_to_pgm


def _write_cloud(path: Path, points: list[list[float]]) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    o3d.io.write_point_cloud(str(path), cloud)


def test_point_cloud_to_pgm(tmp_path: Path) -> None:
    input_path = tmp_path / "input.ply"
    output_path = tmp_path / "output.pgm"
    _write_cloud(input_path, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    width, height = point_cloud_to_pgm(input_path, output_path, resolution=1.0)

    assert output_path.exists()
    assert (width, height) == (2, 2)
    image = np.asarray(Image.open(output_path))
    assert np.count_nonzero(image == 0) == 3


def test_reference_grid_keeps_origin_resolution_and_dimensions(tmp_path: Path) -> None:
    input_path = tmp_path / "input.ply"
    output_path = tmp_path / "output.pgm"
    reference_pgm = tmp_path / "reference.pgm"
    reference_yaml = tmp_path / "reference.yaml"

    # First point maps to the bottom-left pixel; second to column 1, top row.
    # Third point is outside the reference map and must be clipped.
    _write_cloud(input_path, [[-1, -2, 0], [0, 0, 0], [10, 10, 0]])
    Image.fromarray(np.full((3, 4), 255, dtype=np.uint8)).save(reference_pgm)
    reference_yaml.write_text(
        yaml.safe_dump(
            {"image": "reference.pgm", "mode": "trinary", "resolution": 1.0,
             "origin": [-1.0, -2.0, 0.0], "negate": 0,
             "occupied_thresh": 0.65, "free_thresh": 0.196}
        ),
        encoding="utf-8",
    )

    width, height = point_cloud_to_pgm(
        input_path,
        output_path,
        reference_yaml=reference_yaml,
        reference_pgm=reference_pgm,
    )

    image = np.asarray(Image.open(output_path).convert("L"))
    assert (width, height) == (4, 3)
    assert image.shape == (3, 4)
    assert image[2, 0] == 0
    assert image[0, 1] == 0
    assert np.count_nonzero(image == 0) == 2


def test_reference_files_must_be_supplied_together(tmp_path: Path) -> None:
    input_path = tmp_path / "input.ply"
    output_path = tmp_path / "output.pgm"
    _write_cloud(input_path, [[0, 0, 0]])

    with pytest.raises(ValueError, match="must be used together"):
        point_cloud_to_pgm(input_path, output_path, reference_yaml=tmp_path / "map.yaml")
