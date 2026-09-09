from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from src.cloud_to_pgm import point_cloud_to_pgm


def test_point_cloud_to_pgm(tmp_path: Path) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    )
    input_path = tmp_path / "input.ply"
    output_path = tmp_path / "output.pgm"
    o3d.io.write_point_cloud(str(input_path), cloud)

    width, height = point_cloud_to_pgm(input_path, output_path, resolution=1.0)

    assert output_path.exists()
    assert width == 2
    assert height == 2
    image = np.asarray(Image.open(output_path))
    assert np.count_nonzero(image == 0) == 3
