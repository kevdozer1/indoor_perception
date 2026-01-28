"""SAM 2-based segmentation model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Sam2MaskInfo:
    label_id: int
    score: float
    is_thing: bool
    label_name: str


class Sam2SegmentationModel:
    """Run SAM 2 automatic mask generation and return a segmentation map."""

    def __init__(
        self,
        checkpoint_path: str,
        model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
    ) -> None:
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "SAM 2 is not installed. Install it from https://github.com/facebookresearch/sam2 "
                "and ensure its dependencies are available."
            ) from exc

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.model_config = model_config
        self.checkpoint_path = checkpoint_path

        self._model = build_sam2(model_config, checkpoint_path, device=device)
        self._mask_generator = SAM2AutomaticMaskGenerator(
            self._model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )

    def segment(self, rgb: np.ndarray) -> Tuple[np.ndarray, Dict[int, Dict]]:
        """Return segmentation map and segment metadata."""
        image = rgb
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        masks = self._mask_generator.generate(image)

        if not masks:
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.int32), {}

        # Sort by area so larger masks get assigned first.
        masks_sorted: List[dict] = sorted(masks, key=lambda m: m.get("area", 0), reverse=True)
        h, w = image.shape[:2]
        segmentation_map = np.zeros((h, w), dtype=np.int32)
        segment_info: Dict[int, Dict] = {}

        current_id = 1
        for mask in masks_sorted:
            seg = mask.get("segmentation")
            if seg is None:
                continue
            fill = np.logical_and(seg, segmentation_map == 0)
            if not np.any(fill):
                continue
            segmentation_map[fill] = current_id
            score = float(mask.get("predicted_iou", 0.0))
            segment_info[current_id] = {
                "label_id": current_id,
                "label_name": f"sam2_mask_{current_id}",
                "is_thing": True,
                "score": score,
            }
            current_id += 1

        return segmentation_map, segment_info
