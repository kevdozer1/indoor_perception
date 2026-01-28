"""Main pipeline for indoor scene perception."""

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from indoor_perception.dataset.base import RGBDDataset
from indoor_perception.io.ply_writer import write_ply
from indoor_perception.projection.projector import DepthProjector
from indoor_perception.segmentation.model import PanopticSegmentationModel


class ScenePerceptionPipeline:
    """
    Complete pipeline for indoor scene perception.

    Orchestrates: data loading → segmentation → 3D projection → output
    """

    def __init__(
        self,
        dataset: RGBDDataset,
        segmentation_model: Optional[PanopticSegmentationModel] = None,
        model_name: str = "facebook/mask2former-swin-large-coco-panoptic",
        device: str = "auto",
    ):
        """
        Initialize the perception pipeline.

        Args:
            dataset: RGB-D dataset to process
            segmentation_model: Optional pre-initialized segmentation model.
                               If None, will create one with model_name.
            model_name: Model name to use if segmentation_model is None
            device: Device for model inference ('cuda', 'cpu', or 'auto')
        """
        self.dataset = dataset
        self.projector = DepthProjector()

        # Initialize or use provided segmentation model
        if segmentation_model is None:
            print("Initializing segmentation model...")
            self.segmentation_model = PanopticSegmentationModel(
                model_name=model_name,
                device=device,
            )
        else:
            self.segmentation_model = segmentation_model

        print(f"Pipeline initialized with {len(dataset)} frames")

    def process_frame(
        self,
        idx: int,
        output_path: Optional[str] = None,
        save_ply: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single frame through the complete pipeline.

        Args:
            idx: Frame index in the dataset
            output_path: Optional path to save PLY file. If None and save_ply=True,
                        generates default path.
            save_ply: Whether to save the point cloud to a PLY file

        Returns:
            Dictionary containing:
                - 'points': Nx3 array of 3D points
                - 'colors': Nx3 array of RGB colors
                - 'labels': N array of segment IDs
                - 'segment_info': Metadata about segments
                - 'frame_id': Frame identifier
                - 'output_file': Path to saved PLY (if save_ply=True)

        Example:
            >>> pipeline = ScenePerceptionPipeline(dataset)
            >>> result = pipeline.process_frame(0)
            >>> print(f"Generated {len(result['points'])} points")
        """
        # Load RGB-D frame
        frame = self.dataset[idx]
        rgb = frame["rgb"]
        depth = frame["depth"]
        intrinsics = frame["intrinsics"]
        frame_id = frame["frame_id"]

        print(f"\nProcessing frame {idx} (ID: {frame_id})")
        print(f"  Image size: {rgb.shape[:2]}")

        # Run panoptic segmentation
        print("  Running segmentation...")
        segmentation_map, segment_info = self.segmentation_model.segment(rgb)
        print(f"  Found {len(segment_info)} segments")

        # Project to 3D with segmentation labels
        print("  Projecting to 3D...")
        points, colors, labels = self.projector.project_segmented_scene(
            depth=depth,
            segmentation_mask=segmentation_map,
            intrinsics=intrinsics,
            rgb=rgb,
        )
        print(f"  Generated {len(points)} 3D points")

        # Save PLY if requested
        output_file = None
        if save_ply:
            if output_path is None:
                scene_id = frame.get("scene_id", "scene")
                output_path = f"{scene_id}_{frame_id}.ply"

            print(f"  Saving to {output_path}")
            write_ply(output_path, points, colors, labels)
            output_file = output_path

        return {
            "points": points,
            "colors": colors,
            "labels": labels,
            "segment_info": segment_info,
            "frame_id": frame_id,
            "output_file": output_file,
        }

    def process_all(
        self,
        output_dir: str,
        max_frames: Optional[int] = None,
        save_metadata: bool = True,
    ) -> list:
        """
        Process all frames in the dataset.

        Args:
            output_dir: Directory to save output PLY files
            max_frames: Optional limit on number of frames to process
            save_metadata: Whether to save segment metadata as JSON

        Returns:
            List of result dictionaries for each processed frame

        Example:
            >>> pipeline = ScenePerceptionPipeline(dataset)
            >>> results = pipeline.process_all("output/", max_frames=10)
            >>> print(f"Processed {len(results)} frames")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_frames = len(self.dataset)
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)

        print(f"\n{'='*60}")
        print(f"Processing {num_frames} frames")
        print(f"Output directory: {output_path}")
        print(f"{'='*60}")

        results = []
        for idx in range(num_frames):
            try:
                # Generate output filename
                frame = self.dataset[idx]
                scene_id = frame.get("scene_id", "scene")
                frame_id = frame["frame_id"]
                output_file = output_path / f"{scene_id}_{frame_id}.ply"

                # Process frame
                result = self.process_frame(
                    idx,
                    output_path=str(output_file),
                    save_ply=True,
                )

                # Save metadata if requested
                if save_metadata:
                    metadata_file = output_file.with_suffix(".json")
                    self._save_metadata(result, metadata_file)

                results.append(result)

            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Completed: {len(results)}/{num_frames} frames processed successfully")
        print(f"{'='*60}\n")

        return results

    def _save_metadata(self, result: Dict[str, Any], output_file: Path) -> None:
        """Save segment metadata to JSON file."""
        import json

        metadata = {
            "frame_id": result["frame_id"],
            "num_points": len(result["points"]),
            "segments": {
                str(seg_id): {
                    "label_name": info["label_name"],
                    "label_id": int(info["label_id"]),
                    "is_thing": info["is_thing"],
                    "score": float(info["score"]),
                }
                for seg_id, info in result["segment_info"].items()
            },
        }

        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            "num_frames": len(self.dataset),
            "model_name": self.segmentation_model.model_name,
            "device": self.segmentation_model.device,
            "num_classes": self.segmentation_model.get_num_classes(),
        }
