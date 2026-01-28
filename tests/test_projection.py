"""Tests for 3D projection module."""

import numpy as np
import pytest

from indoor_perception.projection import DepthProjector, compute_intrinsics_matrix


class TestComputeIntrinsics:
    """Tests for intrinsics matrix computation."""

    def test_compute_intrinsics_matrix(self):
        """Test creating an intrinsics matrix."""
        K = compute_intrinsics_matrix(fx=525.0, fy=525.0, cx=319.5, cy=239.5)

        assert K.shape == (3, 3)
        assert K[0, 0] == 525.0  # fx
        assert K[1, 1] == 525.0  # fy
        assert K[0, 2] == 319.5  # cx
        assert K[1, 2] == 239.5  # cy
        assert K[2, 2] == 1.0
        assert K[0, 1] == 0.0
        assert K[1, 0] == 0.0


class TestDepthProjector:
    """Tests for DepthProjector class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Simple intrinsics for a 640x480 image
        self.intrinsics = compute_intrinsics_matrix(
            fx=525.0, fy=525.0, cx=319.5, cy=239.5
        )

    def test_project_single_pixel(self):
        """Test projecting a single pixel at the center."""
        # 3x3 depth map with center pixel at 1 meter
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 1] = 1.0  # Center pixel

        # Intrinsics for 3x3 image (center at 1, 1)
        K = compute_intrinsics_matrix(fx=1.0, fy=1.0, cx=1.0, cy=1.0)

        projector = DepthProjector(K)
        points, colors = projector.project_to_3d(depth)

        # Should have 1 valid point (the center)
        assert len(points) == 1
        # Center pixel projects to (0, 0, 1)
        np.testing.assert_array_almost_equal(points[0], [0.0, 0.0, 1.0])

    def test_project_planar_surface(self):
        """Test projecting a planar depth surface."""
        h, w = 10, 10
        depth = np.ones((h, w), dtype=np.float32) * 2.0  # Constant 2 meters

        K = compute_intrinsics_matrix(fx=1.0, fy=1.0, cx=4.5, cy=4.5)

        projector = DepthProjector(K)
        points, colors = projector.project_to_3d(depth)

        # All points should have Z = 2.0
        assert len(points) == h * w
        np.testing.assert_array_almost_equal(points[:, 2], np.full(h * w, 2.0))

    def test_filter_invalid_depth(self):
        """Test that zero/negative depth values are filtered out."""
        depth = np.array([
            [1.0, 0.0, 1.0],
            [0.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)

        K = compute_intrinsics_matrix(fx=1.0, fy=1.0, cx=1.0, cy=1.0)

        projector = DepthProjector(K)
        points, colors = projector.project_to_3d(depth)

        # Should have 6 valid points (positive depths only)
        assert len(points) == 6

    def test_project_with_rgb(self):
        """Test projecting with RGB colors."""
        depth = np.ones((5, 5), dtype=np.float32)
        rgb = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)

        K = compute_intrinsics_matrix(fx=1.0, fy=1.0, cx=2.0, cy=2.0)

        projector = DepthProjector(K)
        points, colors = projector.project_to_3d(depth, rgb=rgb)

        assert len(points) == 25
        assert colors is not None
        assert colors.shape == (25, 3)

    def test_apply_segmentation(self):
        """Test applying segmentation labels to points."""
        h, w = 5, 5
        depth = np.ones((h, w), dtype=np.float32)
        segmentation = np.arange(h * w, dtype=np.int32).reshape(h, w)

        K = compute_intrinsics_matrix(fx=1.0, fy=1.0, cx=2.0, cy=2.0)

        projector = DepthProjector(K)
        points, _ = projector.project_to_3d(depth)
        points, labels = projector.apply_segmentation(points, segmentation, depth)

        assert len(labels) == h * w
        assert labels.min() >= 0
        assert labels.max() <= h * w - 1

    def test_project_segmented_scene(self):
        """Test the complete pipeline: project + segment."""
        h, w = 10, 10
        depth = np.ones((h, w), dtype=np.float32) * 1.5
        rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        segmentation = np.random.randint(0, 5, (h, w), dtype=np.int32)

        K = compute_intrinsics_matrix(fx=100.0, fy=100.0, cx=5.0, cy=5.0)

        projector = DepthProjector(K)
        points, colors, labels = projector.project_segmented_scene(
            depth, segmentation, K, rgb
        )

        assert len(points) == h * w
        assert colors is not None
        assert len(colors) == h * w
        assert len(labels) == h * w

    def test_no_intrinsics_raises_error(self):
        """Test that missing intrinsics raises ValueError."""
        depth = np.ones((5, 5), dtype=np.float32)
        projector = DepthProjector()  # No intrinsics

        with pytest.raises(ValueError, match="Camera intrinsics must be provided"):
            projector.project_to_3d(depth)

    def test_invalid_intrinsics_shape(self):
        """Test that invalid intrinsics shape raises ValueError."""
        depth = np.ones((5, 5), dtype=np.float32)
        bad_intrinsics = np.eye(4)  # Should be 3x3

        projector = DepthProjector(bad_intrinsics)

        with pytest.raises(ValueError, match="Intrinsics must be 3x3 matrix"):
            projector.project_to_3d(depth)

    def test_mismatched_rgb_depth_shape(self):
        """Test that mismatched RGB and depth shapes raise ValueError."""
        depth = np.ones((5, 5), dtype=np.float32)
        rgb = np.ones((10, 10, 3), dtype=np.uint8)  # Wrong size

        K = compute_intrinsics_matrix(fx=1.0, fy=1.0, cx=2.0, cy=2.0)
        projector = DepthProjector(K)

        with pytest.raises(ValueError, match="doesn't match depth shape"):
            projector.project_to_3d(depth, rgb=rgb)

    def test_mismatched_segmentation_depth_shape(self):
        """Test that mismatched segmentation and depth shapes raise ValueError."""
        depth = np.ones((5, 5), dtype=np.float32)
        segmentation = np.ones((10, 10), dtype=np.int32)  # Wrong size

        K = compute_intrinsics_matrix(fx=1.0, fy=1.0, cx=2.0, cy=2.0)
        projector = DepthProjector(K)
        points, _ = projector.project_to_3d(depth)

        with pytest.raises(ValueError, match="doesn't match depth shape"):
            projector.apply_segmentation(points, segmentation, depth)
