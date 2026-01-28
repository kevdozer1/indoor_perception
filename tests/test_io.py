"""Tests for I/O module (PLY reading/writing)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from indoor_perception.io import read_ply, write_ply


class TestPLYWriter:
    """Tests for PLY file writing and reading."""

    def test_write_basic_point_cloud(self):
        """Test writing a basic point cloud without colors or labels."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = f.name

        try:
            write_ply(filepath, points)
            assert Path(filepath).exists()

            # Read back and verify
            data = read_ply(filepath)
            np.testing.assert_array_almost_equal(data["points"], points)
            assert data["colors"].shape == (4, 3)  # Default white colors

        finally:
            Path(filepath).unlink()

    def test_write_with_colors(self):
        """Test writing point cloud with colors."""
        points = np.random.rand(100, 3).astype(np.float32)
        colors = np.random.randint(0, 256, (100, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = f.name

        try:
            write_ply(filepath, points, colors=colors)

            data = read_ply(filepath)
            np.testing.assert_array_almost_equal(data["points"], points)
            np.testing.assert_array_equal(data["colors"], colors)

        finally:
            Path(filepath).unlink()

    def test_write_with_labels(self):
        """Test writing point cloud with semantic labels."""
        points = np.random.rand(50, 3).astype(np.float32)
        labels = np.random.randint(0, 10, 50, dtype=np.int32)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = f.name

        try:
            write_ply(filepath, points, labels=labels)

            data = read_ply(filepath)
            np.testing.assert_array_almost_equal(data["points"], points)
            np.testing.assert_array_equal(data["labels"], labels)

        finally:
            Path(filepath).unlink()

    def test_write_with_colors_and_labels(self):
        """Test writing point cloud with both colors and labels."""
        points = np.random.rand(75, 3).astype(np.float32)
        colors = np.random.randint(0, 256, (75, 3), dtype=np.uint8)
        labels = np.random.randint(0, 5, 75, dtype=np.int32)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = f.name

        try:
            write_ply(filepath, points, colors=colors, labels=labels)

            data = read_ply(filepath)
            np.testing.assert_array_almost_equal(data["points"], points, decimal=5)
            np.testing.assert_array_equal(data["colors"], colors)
            np.testing.assert_array_equal(data["labels"], labels)

        finally:
            Path(filepath).unlink()

    def test_invalid_points_shape(self):
        """Test that invalid points shape raises ValueError."""
        invalid_points = np.array([[1, 2]])  # Should be Nx3

        with tempfile.NamedTemporaryFile(suffix=".ply") as f:
            with pytest.raises(ValueError, match="Points must be Nx3"):
                write_ply(f.name, invalid_points)

    def test_mismatched_colors_shape(self):
        """Test that mismatched colors shape raises ValueError."""
        points = np.random.rand(10, 3)
        colors = np.random.randint(0, 256, (5, 3), dtype=np.uint8)  # Wrong size

        with tempfile.NamedTemporaryFile(suffix=".ply") as f:
            with pytest.raises(ValueError, match="Colors must be Nx3"):
                write_ply(f.name, points, colors=colors)

    def test_mismatched_labels_shape(self):
        """Test that mismatched labels shape raises ValueError."""
        points = np.random.rand(10, 3)
        labels = np.array([1, 2, 3])  # Wrong size

        with tempfile.NamedTemporaryFile(suffix=".ply") as f:
            with pytest.raises(ValueError, match="Labels must be N array"):
                write_ply(f.name, points, labels=labels)

    def test_auto_create_parent_directory(self):
        """Test that parent directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir1" / "subdir2" / "test.ply"
            points = np.random.rand(10, 3)

            write_ply(str(filepath), points)
            assert filepath.exists()

    def test_color_dtype_conversion(self):
        """Test that colors are properly converted to uint8."""
        points = np.random.rand(10, 3)
        colors_float = np.random.rand(10, 3) * 255  # Float colors

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            filepath = f.name

        try:
            write_ply(filepath, points, colors=colors_float)

            data = read_ply(filepath)
            assert data["colors"].dtype == np.uint8

        finally:
            Path(filepath).unlink()
