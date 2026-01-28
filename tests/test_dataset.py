"""Tests for dataset loading."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from indoor_perception.dataset import ScanNetDataset, create_sample_scene


class TestScanNetDataset:
    """Tests for ScanNet dataset loader."""

    def test_create_sample_scene(self):
        """Test creating a synthetic sample scene."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = create_sample_scene(
                output_dir=tmpdir,
                scene_id="test_scene",
                num_frames=5,
                image_size=(640, 480),
            )

            assert scene_path.exists()
            assert (scene_path / "color").exists()
            assert (scene_path / "depth").exists()
            assert (scene_path / "intrinsic").exists()

            # Check that files were created
            color_files = list((scene_path / "color").glob("*.jpg"))
            depth_files = list((scene_path / "depth").glob("*.png"))

            assert len(color_files) == 5
            assert len(depth_files) == 5

            # Check intrinsics file
            intrinsic_file = scene_path / "intrinsic" / "intrinsic_color.txt"
            assert intrinsic_file.exists()

    def test_load_synthetic_dataset(self):
        """Test loading a synthetic dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_scene(
                output_dir=tmpdir,
                scene_id="scene0000_00",
                num_frames=3,
            )

            dataset = ScanNetDataset(data_root=tmpdir)

            assert len(dataset) == 3

    def test_get_frame(self):
        """Test getting a frame from the dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_scene(
                output_dir=tmpdir,
                scene_id="scene0000_00",
                num_frames=5,
                image_size=(320, 240),
            )

            dataset = ScanNetDataset(data_root=tmpdir)
            frame = dataset[0]

            # Check that all required keys are present
            assert "rgb" in frame
            assert "depth" in frame
            assert "intrinsics" in frame
            assert "frame_id" in frame
            assert "scene_id" in frame

            # Check data types and shapes
            assert frame["rgb"].shape == (240, 320, 3)
            assert frame["rgb"].dtype == np.uint8
            assert frame["depth"].shape == (240, 320)
            assert frame["depth"].dtype == np.float32
            assert frame["intrinsics"].shape == (3, 3)

    def test_depth_scaling(self):
        """Test that depth values are properly scaled from mm to meters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_scene(
                output_dir=tmpdir,
                scene_id="scene0000_00",
                num_frames=1,
            )

            dataset = ScanNetDataset(data_root=tmpdir, depth_scale=1000.0)
            frame = dataset[0]

            # Depth should be in reasonable range for meters (< 10m typically)
            assert frame["depth"].max() < 10.0
            assert frame["depth"].min() >= 0.0

    def test_multiple_scenes(self):
        """Test loading multiple scenes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_scene(tmpdir, "scene0000_00", num_frames=3)
            create_sample_scene(tmpdir, "scene0000_01", num_frames=2)

            dataset = ScanNetDataset(data_root=tmpdir)

            # Should have 5 total frames
            assert len(dataset) == 5

    def test_specific_scenes(self):
        """Test loading specific scene IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_scene(tmpdir, "scene0000_00", num_frames=3)
            create_sample_scene(tmpdir, "scene0000_01", num_frames=2)

            # Load only one scene
            dataset = ScanNetDataset(
                data_root=tmpdir,
                scene_ids=["scene0000_00"],
            )

            assert len(dataset) == 3

    def test_max_frames_per_scene(self):
        """Test limiting frames per scene."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_scene(tmpdir, "scene0000_00", num_frames=10)

            dataset = ScanNetDataset(
                data_root=tmpdir,
                max_frames_per_scene=3,
            )

            assert len(dataset) == 3

    def test_get_frame_path(self):
        """Test getting frame file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_scene(tmpdir, "scene0000_00", num_frames=2)

            dataset = ScanNetDataset(data_root=tmpdir)
            path = dataset.get_frame_path(0)

            assert isinstance(path, str)
            assert Path(path).exists()
            assert path.endswith(".jpg")

    def test_invalid_index(self):
        """Test that invalid index raises IndexError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_scene(tmpdir, "scene0000_00", num_frames=2)

            dataset = ScanNetDataset(data_root=tmpdir)

            with pytest.raises(IndexError):
                _ = dataset[10]

            with pytest.raises(IndexError):
                _ = dataset[-1]

    def test_nonexistent_data_root(self):
        """Test that nonexistent data root raises ValueError."""
        with pytest.raises(ValueError, match="Data root does not exist"):
            ScanNetDataset(data_root="/nonexistent/path")

    def test_empty_dataset(self):
        """Test that empty dataset raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory but no scenes
            with pytest.raises(ValueError, match="No frames found"):
                ScanNetDataset(data_root=tmpdir)

    def test_intrinsics_loading(self):
        """Test that intrinsics are properly loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_scene(tmpdir, "scene0000_00", num_frames=1)

            dataset = ScanNetDataset(data_root=tmpdir)
            frame = dataset[0]

            K = frame["intrinsics"]

            # Check that it's a valid intrinsics matrix
            assert K.shape == (3, 3)
            assert K[2, 2] == 1.0  # Bottom-right should be 1
            assert K[0, 0] > 0  # fx should be positive
            assert K[1, 1] > 0  # fy should be positive
