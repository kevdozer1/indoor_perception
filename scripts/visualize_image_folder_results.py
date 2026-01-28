"""Visualize image-folder pipeline results with grid views."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from indoor_perception.visualizer import visualize_pipeline_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize image-folder pipeline results")
    parser.add_argument("--image-dir", required=True, help="Image directory")
    parser.add_argument("--pattern", default="*.*", help="Glob pattern for images")
    parser.add_argument("--ply-dir", required=True, help="Directory with PLY files")
    parser.add_argument("--output", required=True, help="Output directory for visualizations")
    parser.add_argument("--segmentation-dir", required=False, help="Directory with segmentation maps")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    ply_dir = Path(args.ply_dir)
    segmentation_dir = Path(args.segmentation_dir) if args.segmentation_dir else ply_dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in image_dir.glob(args.pattern) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.max_images:
        images = images[: args.max_images]

    if not images:
        print(f"No images found in {image_dir}")
        return

    scene_id = image_dir.name
    for image_path in images:
        frame_id = image_path.stem
        full_frame_id = f"{scene_id}_{frame_id}"

        ply_path = ply_dir / f"{full_frame_id}.ply"
        if not ply_path.exists():
            print(f"Skipping missing PLY: {ply_path}")
            continue

        seg_path = segmentation_dir / f"{full_frame_id}.segmentation.npy"
        if not seg_path.exists():
            print(f"Skipping missing segmentation: {seg_path}")
            continue

        rgb = np.array(Image.open(image_path).convert("RGB"))
        segmentation_mask = np.load(seg_path)

        visualize_pipeline_result(
            rgb_image=rgb,
            segmentation_mask=segmentation_mask,
            ply_path=ply_path,
            output_path=output_dir,
            frame_id=full_frame_id,
        )


if __name__ == "__main__":
    main()
