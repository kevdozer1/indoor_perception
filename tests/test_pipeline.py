"""Integration tests for the complete pipeline."""

import tempfile
from pathlib import Path

import pytest

from indoor_perception.dataset import ScanNetDataset, create_sample_scene
from indoor_perception.pipeline import ScenePerceptionPipeline


class TestScenePerceptionPipeline:
    """Integration tests for the full pipeline."""

    def setup_method(self):
        """Set up test dataset."""
        self.tmpdir = tempfile.mkdtemp()
        create_sample_scene(
            output_dir=self.tmpdir,
            scene_id="test_scene",
            num_frames=3,
            image_size=(320, 240),
        )

        self.dataset = ScanNetDataset(data_root=self.tmpdir)

    @pytest.mark.slow
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = ScenePerceptionPipeline(
            dataset=self.dataset,
            model_name="facebook/mask2former-swin-tiny-coco-panoptic",
            device="cpu",
        )

        assert pipeline.dataset is not None
        assert pipeline.segmentation_model is not None
        assert pipeline.projector is not None

    @pytest.mark.slow
    def test_process_single_frame(self):
        """Test processing a single frame."""
        pipeline = ScenePerceptionPipeline(
            dataset=self.dataset,
            model_name="facebook/mask2former-swin-tiny-coco-panoptic",
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "test_output.ply"

            result = pipeline.process_frame(
                idx=0,
                output_path=str(output_path),
                save_ply=True,
            )

            # Check result structure
            assert "points" in result
            assert "colors" in result
            assert "labels" in result
            assert "segment_info" in result
            assert "frame_id" in result
            assert "output_file" in result

            # Check that output file was created
            assert output_path.exists()

            # Check data shapes
            n_points = len(result["points"])
            assert result["points"].shape == (n_points, 3)
            assert result["colors"].shape == (n_points, 3)
            assert result["labels"].shape == (n_points,)

    @pytest.mark.slow
    def test_process_all_frames(self):
        """Test processing all frames in dataset."""
        pipeline = ScenePerceptionPipeline(
            dataset=self.dataset,
            model_name="facebook/mask2former-swin-tiny-coco-panoptic",
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as output_dir:
            results = pipeline.process_all(
                output_dir=output_dir,
                save_metadata=True,
            )

            # Check that all frames were processed
            assert len(results) == len(self.dataset)

            # Check that output files were created
            output_path = Path(output_dir)
            ply_files = list(output_path.glob("*.ply"))
            json_files = list(output_path.glob("*.json"))

            assert len(ply_files) == len(self.dataset)
            assert len(json_files) == len(self.dataset)

    @pytest.mark.slow
    def test_process_limited_frames(self):
        """Test processing with max_frames limit."""
        pipeline = ScenePerceptionPipeline(
            dataset=self.dataset,
            model_name="facebook/mask2former-swin-tiny-coco-panoptic",
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as output_dir:
            results = pipeline.process_all(
                output_dir=output_dir,
                max_frames=2,
            )

            assert len(results) == 2

    def test_get_statistics(self):
        """Test getting pipeline statistics."""
        pytest.skip("Requires model download - run manually if needed")

        pipeline = ScenePerceptionPipeline(
            dataset=self.dataset,
            model_name="facebook/mask2former-swin-tiny-coco-panoptic",
            device="cpu",
        )

        stats = pipeline.get_statistics()

        assert "num_frames" in stats
        assert "model_name" in stats
        assert "device" in stats
        assert "num_classes" in stats

        assert stats["num_frames"] == len(self.dataset)
        assert stats["device"] == "cpu"

    def test_process_without_saving(self):
        """Test processing frame without saving PLY."""
        pytest.skip("Requires model download - run manually if needed")

        pipeline = ScenePerceptionPipeline(
            dataset=self.dataset,
            device="cpu",
        )

        result = pipeline.process_frame(idx=0, save_ply=False)

        assert result["output_file"] is None
        assert "points" in result
        assert "colors" in result
        assert "labels" in result


class TestPipelineWithPreInitializedModel:
    """Test pipeline with pre-initialized segmentation model."""

    @pytest.mark.slow
    def test_use_existing_model(self):
        """Test pipeline with pre-initialized model."""
        from indoor_perception.segmentation import PanopticSegmentationModel

        # Initialize model separately
        model = PanopticSegmentationModel(
            model_name="facebook/mask2former-swin-tiny-coco-panoptic",
            device="cpu",
        )

        # Create dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_scene(tmpdir, "scene0000_00", num_frames=1)
            dataset = ScanNetDataset(data_root=tmpdir)

            # Pass model to pipeline
            pipeline = ScenePerceptionPipeline(
                dataset=dataset,
                segmentation_model=model,
            )

            assert pipeline.segmentation_model is model
