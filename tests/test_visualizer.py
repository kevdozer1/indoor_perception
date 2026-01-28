"""Tests for visualization module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from indoor_perception.io import write_ply
from indoor_perception.visualizer import (
    apply_label_colors,
    create_segmentation_overlay,
    generate_label_colors,
    render_point_cloud_to_image,
    visualize_ply_with_labels,
)


class TestLabelColors:
    """Tests for label color generation."""

    def test_generate_label_colors(self):
        """Test generating distinct colors for labels."""
        colors = generate_label_colors(10, seed=42)

        assert colors.shape == (10, 3)
        assert colors.min() >= 0.0
        assert colors.max() <= 1.0

    def test_apply_label_colors(self):
        """Test applying colors to labels."""
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        colors = apply_label_colors(labels, num_labels=4)

        assert colors.shape == (8, 3)
        assert colors.min() >= 0.0
        assert colors.max() <= 1.0

        # Same labels should have same colors
        np.testing.assert_array_equal(colors[0], colors[1])
        np.testing.assert_array_equal(colors[2], colors[3])

    def test_apply_custom_label_colors(self):
        """Test applying custom color map."""
        labels = np.array([0, 1, 2])
        custom_colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
        ])

        colors = apply_label_colors(labels, label_colors=custom_colors)

        np.testing.assert_array_almost_equal(colors[0], custom_colors[0])
        np.testing.assert_array_almost_equal(colors[1], custom_colors[1])
        np.testing.assert_array_almost_equal(colors[2], custom_colors[2])


class TestPointCloudRendering:
    """Tests for point cloud rendering."""

    def test_render_simple_point_cloud(self):
        """Test rendering a simple point cloud."""
        # Create a small cube of points
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=np.float32)

        colors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=np.float32)

        image = render_point_cloud_to_image(points, colors, image_size=(400, 300))

        assert image.shape == (300, 400, 3)
        assert image.dtype == np.uint8

    def test_render_with_custom_view(self):
        """Test rendering with custom view parameters."""
        points = np.random.rand(100, 3).astype(np.float32)
        colors = np.random.rand(100, 3).astype(np.float32)

        view_params = {
            "zoom": 0.8,
            "front": [0, 0, -1],
            "lookat": [0.5, 0.5, 0.5],
            "up": [0, 1, 0],
        }

        image = render_point_cloud_to_image(
            points, colors, image_size=(640, 480), view_params=view_params
        )

        assert image.shape == (480, 640, 3)


class TestPLYVisualization:
    """Tests for PLY file visualization."""

    def test_visualize_ply_with_rgb_colors(self):
        """Test visualizing PLY with RGB colors."""
        points = np.random.rand(50, 3).astype(np.float32)
        colors = np.random.randint(0, 256, (50, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply_path = f.name

        try:
            write_ply(ply_path, points, colors=colors)

            image = visualize_ply_with_labels(
                ply_path,
                use_label_colors=False,
                image_size=(400, 300),
            )

            assert image.shape == (300, 400, 3)
            assert image.dtype == np.uint8

        finally:
            Path(ply_path).unlink()

    def test_visualize_ply_with_labels(self):
        """Test visualizing PLY with semantic labels."""
        points = np.random.rand(50, 3).astype(np.float32)
        colors = np.random.randint(0, 256, (50, 3), dtype=np.uint8)
        labels = np.random.randint(0, 5, 50, dtype=np.int32)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply_path = f.name

        try:
            write_ply(ply_path, points, colors=colors, labels=labels)

            image = visualize_ply_with_labels(
                ply_path,
                use_label_colors=True,
                image_size=(400, 300),
            )

            assert image.shape == (300, 400, 3)

        finally:
            Path(ply_path).unlink()

    def test_visualize_and_save(self):
        """Test rendering and saving to file."""
        points = np.random.rand(30, 3).astype(np.float32)
        colors = np.random.randint(0, 256, (30, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            write_ply(ply_path, points, colors=colors)

            visualize_ply_with_labels(
                ply_path,
                output_path=output_path,
                use_label_colors=False,
            )

            assert Path(output_path).exists()

        finally:
            Path(ply_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()


class TestSegmentationOverlay:
    """Tests for segmentation overlay creation."""

    def test_create_overlay(self):
        """Test creating segmentation overlay."""
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        seg_mask = np.random.randint(0, 5, (100, 100), dtype=np.int32)

        overlay = create_segmentation_overlay(rgb_image, seg_mask, alpha=0.5)

        assert overlay.shape == rgb_image.shape
        assert overlay.dtype == np.uint8

    def test_overlay_alpha_transparency(self):
        """Test overlay with different alpha values."""
        rgb_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        seg_mask = np.zeros((50, 50), dtype=np.int32)

        # Alpha = 0 should be close to original
        overlay_0 = create_segmentation_overlay(rgb_image, seg_mask, alpha=0.0)
        assert np.allclose(overlay_0, rgb_image, atol=5)

        # Alpha = 1 should be fully segmentation colors
        overlay_1 = create_segmentation_overlay(rgb_image, seg_mask, alpha=1.0)
        assert not np.allclose(overlay_1, rgb_image)

    def test_overlay_with_custom_colors(self):
        """Test overlay with custom label colors."""
        rgb_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        seg_mask = np.random.randint(0, 3, (50, 50), dtype=np.int32)

        custom_colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        overlay = create_segmentation_overlay(
            rgb_image, seg_mask, alpha=0.5, label_colors=custom_colors
        )

        assert overlay.shape == rgb_image.shape


class TestGridVisualization:
    """Tests for grid visualization."""

    def test_create_grid_basic(self):
        """Test creating basic grid visualization."""
        from indoor_perception.visualizer import create_grid_visualization

        rgb_image = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        seg_mask = np.random.randint(0, 5, (240, 320), dtype=np.int32)
        pc_image = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)

        grid = create_grid_visualization(rgb_image, seg_mask, pc_image)

        assert grid.dtype == np.uint8
        assert len(grid.shape) == 3
        assert grid.shape[2] == 3  # RGB

    def test_create_grid_with_save(self):
        """Test creating and saving grid visualization."""
        from indoor_perception.visualizer import create_grid_visualization

        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        seg_mask = np.random.randint(0, 3, (100, 100), dtype=np.int32)
        pc_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            create_grid_visualization(
                rgb_image, seg_mask, pc_image, output_path=output_path
            )

            assert Path(output_path).exists()

        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()
