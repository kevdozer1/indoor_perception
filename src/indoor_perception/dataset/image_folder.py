"""Image-only dataset loader with synthetic depth."""

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image

from indoor_perception.dataset.base import RGBDDataset


class ImageFolderDataset(RGBDDataset):
    """Load images from a folder and generate synthetic depth."""

    def __init__(
        self,
        image_dir: str,
        pattern: str = "*.*",
        constant_depth_m: float = 2.0,
        depth_mode: str = "constant",
        depth_estimator=None,
    ) -> None:
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")

        self.depth_mode = depth_mode
        self.constant_depth_m = float(constant_depth_m)
        self.depth_estimator = depth_estimator

        self.images: List[Path] = sorted(self.image_dir.glob(pattern))
        self.images = [p for p in self.images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        if not self.images:
            raise ValueError(f"No images found in {self.image_dir} with pattern {pattern}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range [0, {len(self.images)})")

        image_path = self.images[idx]
        rgb = np.array(Image.open(image_path).convert("RGB"))
        h, w = rgb.shape[:2]

        intrinsics = np.array(
            [
                [525.0, 0.0, w / 2.0],
                [0.0, 525.0, h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        if self.depth_mode == "constant":
            depth = np.ones((h, w), dtype=np.float32) * self.constant_depth_m
        elif self.depth_mode == "midas":
            if self.depth_estimator is None:
                raise ValueError("depth_estimator is required for depth_mode='midas'")
            depth = self.depth_estimator(rgb)
        else:
            raise ValueError(f"Unsupported depth mode: {self.depth_mode}")

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "frame_id": image_path.stem,
            "scene_id": self.image_dir.name,
        }

    def get_frame_path(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range [0, {len(self.images)})")
        return str(self.images[idx])
