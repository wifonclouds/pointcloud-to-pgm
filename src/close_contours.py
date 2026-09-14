#!/usr/bin/env python3
"""Close small gaps in contours of an existing PGM occupancy map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def close_contours(input_path: Path, output_path: Path, radius: int = 2) -> None:
    """Close small gaps in black contours of an existing PGM map."""
    if radius < 1:
        raise ValueError("radius must be >= 1")

    image = Image.open(input_path).convert("L")
    data = np.asarray(image)

    # Black pixels are occupied/contour; white pixels are background.
    occupied = data < 128

    size = radius * 2 + 1
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    structure = (xx * xx + yy * yy) <= radius * radius

    closed = ndimage.binary_closing(occupied, structure=structure)

    output = np.full(data.shape, 255, dtype=np.uint8)
    output[closed] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="L").save(output_path, format="PPM")

    print(f"Saved closed contours to {output_path} ({output.shape[1]} x {output.shape[0]} px)")
    print(f"Closing radius: {radius} px")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input PGM map")
    parser.add_argument("output", type=Path, help="Output PGM map")
    parser.add_argument("--radius", type=int, default=2, help="Closing radius in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    close_contours(args.input, args.output, radius=args.radius)


if __name__ == "__main__":
    main()
