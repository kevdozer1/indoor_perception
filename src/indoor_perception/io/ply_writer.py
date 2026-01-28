"""PLY file writer for point clouds with semantic labels."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from plyfile import PlyData, PlyElement


def write_ply(
    filename: Union[str, Path],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> None:
    """
    Write a point cloud to a PLY file.

    Args:
        filename: Output PLY file path
        points: Nx3 array of point coordinates (x, y, z)
        colors: Optional Nx3 array of RGB colors (0-255)
        labels: Optional N array of semantic labels (integer IDs)

    Raises:
        ValueError: If array shapes are incompatible
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be Nx3 array, got shape {points.shape}")

    n_points = len(points)

    # Validate colors if provided
    if colors is not None:
        if colors.shape != (n_points, 3):
            raise ValueError(
                f"Colors must be Nx3 array matching points, got shape {colors.shape}"
            )
        # Ensure colors are uint8
        if colors.dtype != np.uint8:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        # Default to white if no colors provided
        colors = np.full((n_points, 3), 255, dtype=np.uint8)

    # Validate labels if provided
    if labels is not None:
        if labels.shape != (n_points,):
            raise ValueError(
                f"Labels must be N array matching points, got shape {labels.shape}"
            )
        labels = labels.astype(np.int32)

    # Build vertex dtype based on what data we have
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    if labels is not None:
        vertex_dtype.append(("semantic_label", "i4"))

    # Create structured array
    vertex_data = np.empty(n_points, dtype=vertex_dtype)
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]

    if labels is not None:
        vertex_data["semantic_label"] = labels

    # Create PLY element and write to file
    vertex_element = PlyElement.describe(vertex_data, "vertex")
    ply_data = PlyData([vertex_element], text=True)

    # Ensure parent directory exists
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ply_data.write(str(output_path))


def read_ply(filename: Union[str, Path]) -> dict:
    """
    Read a point cloud from a PLY file.

    Args:
        filename: Input PLY file path

    Returns:
        Dictionary with keys:
            - 'points': Nx3 array of point coordinates
            - 'colors': Nx3 array of RGB colors (if present)
            - 'labels': N array of semantic labels (if present)
    """
    ply_data = PlyData.read(str(filename))
    vertex = ply_data["vertex"]

    result = {
        "points": np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1),
    }

    # Check for colors
    if "red" in vertex.data.dtype.names:
        result["colors"] = np.stack(
            [vertex["red"], vertex["green"], vertex["blue"]], axis=1
        ).astype(np.uint8)

    # Check for labels
    if "semantic_label" in vertex.data.dtype.names:
        result["labels"] = vertex["semantic_label"].astype(np.int32)

    return result
