"""Visualize pipeline results with grid views."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from indoor_perception.dataset import ScanNetDataset
from indoor_perception.io import read_ply
from indoor_perception.visualizer import visualize_pipeline_result


def create_visualizations(data_dir: str, output_dir: str, ply_dir: str):
    """Create visualizations for processed frames."""
    output_path = Path(output_dir)
    viz_dir = output_path / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset to get RGB images
    dataset = ScanNetDataset(data_root=data_dir)

    print(f"Creating visualizations for {len(dataset)} frames...")

    for idx in range(len(dataset)):
        frame = dataset[idx]
        scene_id = frame["scene_id"]
        frame_id = frame["frame_id"]
        full_frame_id = f"{scene_id}_{frame_id}"

        print(f"\n  Visualizing {full_frame_id}...")

        # Get PLY path
        ply_path = Path(ply_dir) / f"{full_frame_id}.ply"

        if not ply_path.exists():
            print(f"    Warning: PLY file not found: {ply_path}")
            continue

        # Create dummy segmentation mask (spatial regions)
        h, w = frame["rgb"].shape[:2]
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        segmentation_mask = (x_coords // 80 + y_coords // 60) % 15  # More segments

        # Create visualizations
        visualize_pipeline_result(
            rgb_image=frame["rgb"],
            segmentation_mask=segmentation_mask,
            ply_path=str(ply_path),
            output_path=str(viz_dir),
            frame_id=full_frame_id,
        )

    print(f"\n[OK] Visualizations saved to {viz_dir}")

    # List output files
    print("\nGenerated files:")
    for img_file in sorted(viz_dir.glob("*_grid.png")):
        print(f"  [IMG] {img_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize pipeline results")
    parser.add_argument("--data", required=True, help="Data directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--ply-dir", required=True, help="Directory with PLY files")

    args = parser.parse_args()

    create_visualizations(args.data, args.output, args.ply_dir)


if __name__ == "__main__":
    main()
