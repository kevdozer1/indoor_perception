"""Run SAM 2 automatic mask generation on a folder of images."""

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def list_images(input_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])


def normalize_image(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def save_overlay(image: np.ndarray, masks: List[dict], output_path: Path, alpha: float) -> None:
    overlay = image.astype(np.float32).copy()
    rng = np.random.default_rng(42)

    for mask in masks:
        seg = mask.get("segmentation")
        if seg is None:
            continue
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        overlay[seg] = (1 - alpha) * overlay[seg] + alpha * color

    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


def save_mask_index_map(masks: List[dict], shape: tuple, output_path: Path) -> None:
    mask_map = np.zeros(shape[:2], dtype=np.int32)
    for idx, mask in enumerate(masks, start=1):
        seg = mask.get("segmentation")
        if seg is None:
            continue
        mask_map[seg] = idx
    np.save(output_path, mask_map)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM 2 automatic mask generation on images.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of images to process")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam2.1_hiera_large.pt",
        help="Path to SAM 2 checkpoint",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM 2 model config",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images")
    parser.add_argument("--save-overlays", action="store_true", help="Save color overlays")
    parser.add_argument("--save-mask-map", action="store_true", help="Save mask index map as .npy")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as e:
        raise RuntimeError(
            "SAM 2 is not installed. Install it from https://github.com/facebookresearch/sam2 "
            "and ensure its dependencies are available."
        ) from e

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    checkpoint_path = Path(args.checkpoint)
    model_config_path = Path(args.model_config)

    fallback_root = Path(".cache") / "sam2"
    if not checkpoint_path.exists():
        candidate = fallback_root / checkpoint_path
        if candidate.exists():
            checkpoint_path = candidate

    if not model_config_path.exists():
        candidate = fallback_root / model_config_path
        if candidate.exists():
            model_config_path = candidate

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    # Hydra expects config names relative to the sam2 package (e.g. configs/sam2.1/...)
    config_name = str(model_config_path)
    sam2_root = fallback_root / "sam2" / "configs"
    try:
        if sam2_root in model_config_path.parents:
            config_name = str(model_config_path.relative_to(fallback_root / "sam2"))
    except Exception:
        pass

    print(f"Loading SAM 2 model on {device}...")
    model = build_sam2(config_name, str(checkpoint_path), device=device)
    mask_generator = SAM2AutomaticMaskGenerator(model)

    images = list_images(input_dir)
    if args.max_images:
        images = images[: args.max_images]

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} image(s)")

    for image_path in images:
        print(f"\nProcessing {image_path.name}")
        image = normalize_image(Image.open(image_path))

        with torch.inference_mode():
            masks = mask_generator.generate(image)

        base_name = image_path.stem
        result_dir = output_dir / base_name
        result_dir.mkdir(parents=True, exist_ok=True)

        if args.save_overlays:
            overlay_path = result_dir / f"{base_name}_overlay.png"
            save_overlay(image, masks, overlay_path, alpha=args.alpha)

        if args.save_mask_map:
            mask_map_path = result_dir / f"{base_name}_mask_map.npy"
            save_mask_index_map(masks, image.shape, mask_map_path)

        # Save raw masks as numpy for downstream use
        np.save(result_dir / f"{base_name}_masks.npy", masks, allow_pickle=True)

    print(f"\n[OK] SAM 2 results saved to {output_dir}")


if __name__ == "__main__":
    main()
