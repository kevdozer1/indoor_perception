"""Run the pipeline on a folder of images using SAM 2 segmentation."""

import argparse
from pathlib import Path

from indoor_perception.dataset import ImageFolderDataset
from indoor_perception.pipeline import ScenePerceptionPipeline
from indoor_perception.segmentation import Sam2SegmentationModel


def build_midas_depth_estimator(device: str, depth_min: float, depth_max: float):
    import torch

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    midas.to(device)
    midas.eval()

    def estimate(rgb: "np.ndarray") -> "np.ndarray":
        import numpy as np

        input_batch = midas_transforms(rgb).to(device)
        with torch.inference_mode():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy()
        depth_min_val = np.min(depth)
        depth_max_val = np.max(depth)
        if depth_max_val - depth_min_val < 1e-6:
            norm = np.zeros_like(depth)
        else:
            norm = (depth - depth_min_val) / (depth_max_val - depth_min_val)
        # MiDaS gives inverse depth-like values (higher = closer), so invert to meters.
        depth_m = depth_max - norm * (depth_max - depth_min)
        return depth_m.astype(np.float32)

    return estimate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline on image folder with SAM2 masks.")
    parser.add_argument("--input-dir", required=True, help="Folder of images")
    parser.add_argument("--pattern", default="*.*", help="Glob pattern (default: *.*)")
    parser.add_argument("--output", required=True, help="Output directory for PLY files")
    parser.add_argument("--checkpoint", required=True, help="SAM2 checkpoint path")
    parser.add_argument(
        "--model-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM2 config name or path",
    )
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--constant-depth", type=float, default=2.0, help="Constant depth in meters")
    parser.add_argument(
        "--depth-mode",
        choices=["constant", "midas"],
        default="constant",
        help="Depth generation mode (default: constant)",
    )
    parser.add_argument("--depth-min", type=float, default=0.5, help="Min depth for MiDaS scaling (m)")
    parser.add_argument("--depth-max", type=float, default=4.0, help="Max depth for MiDaS scaling (m)")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images")
    parser.add_argument("--save-segmentation-maps", action="store_true", help="Save segmentation maps")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have output PLY files",
    )

    args = parser.parse_args()

    depth_estimator = None
    if args.depth_mode == "midas":
        depth_estimator = build_midas_depth_estimator(args.device, args.depth_min, args.depth_max)

    dataset = ImageFolderDataset(
        image_dir=args.input_dir,
        pattern=args.pattern,
        constant_depth_m=args.constant_depth,
        depth_mode=args.depth_mode,
        depth_estimator=depth_estimator,
    )

    if args.max_images is not None:
        dataset.images = dataset.images[: args.max_images]
    if args.skip_existing:
        scene_id = dataset.image_dir.name
        output_dir = Path(args.output)
        existing = {
            p.stem.replace(f"{scene_id}_", "", 1)
            for p in output_dir.glob(f"{scene_id}_*.ply")
        }
        dataset.images = [p for p in dataset.images if p.stem not in existing]

    segmentation_model = Sam2SegmentationModel(
        checkpoint_path=args.checkpoint,
        model_config=args.model_config,
        device=args.device,
    )

    pipeline = ScenePerceptionPipeline(
        dataset=dataset,
        segmentation_model=segmentation_model,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline.process_all(
        output_dir=str(output_dir),
        save_metadata=True,
        save_segmentation_maps=args.save_segmentation_maps,
    )


if __name__ == "__main__":
    main()
