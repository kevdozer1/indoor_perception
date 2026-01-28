"""Panoptic segmentation using pretrained models."""

from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


class PanopticSegmentationModel:
    """
    Wrapper for panoptic segmentation using Mask2Former.

    Uses Hugging Face transformers for easy model loading and inference.
    """

    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-large-coco-panoptic",
        device: str = "auto",
    ):
        """
        Initialize the segmentation model.

        Args:
            model_name: Hugging Face model identifier
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading segmentation model: {model_name}")
        print(f"Using device: {self.device}")

        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    @torch.no_grad()
    def segment(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[int, Dict[str, any]]]:
        """
        Run panoptic segmentation on an RGB image.

        Args:
            image: HxWx3 RGB image (numpy array, uint8)

        Returns:
            Tuple of:
                - segmentation_map: HxW array with segment IDs (int32)
                - segment_info: Dictionary mapping segment_id to metadata:
                    - 'label_id': Category ID
                    - 'label_name': Category name
                    - 'is_thing': Whether this is a 'thing' (object) or 'stuff' (background)
                    - 'score': Confidence score

        Example:
            >>> model = PanopticSegmentationModel()
            >>> seg_map, info = model.segment(rgb_image)
            >>> print(f"Found {len(info)} segments")
            >>> for seg_id, meta in info.items():
            ...     print(f"Segment {seg_id}: {meta['label_name']}")
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Preprocess image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        outputs = self.model(**inputs)

        # Post-process to get panoptic segmentation
        result = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[pil_image.size[::-1]],  # (height, width)
        )[0]

        # Extract segmentation map and segment info
        segmentation_map = result["segmentation"].cpu().numpy().astype(np.int32)
        segments_info = result["segments_info"]

        # Build segment metadata dictionary
        segment_metadata = {}
        for seg in segments_info:
            seg_id = seg["id"]
            label_id = seg["label_id"]
            label_name = self.model.config.id2label[label_id]

            segment_metadata[seg_id] = {
                "label_id": label_id,
                "label_name": label_name,
                "is_thing": seg.get("isthing", True),
                "score": seg.get("score", 1.0),
            }

        return segmentation_map, segment_metadata

    def get_label_name(self, label_id: int) -> str:
        """Get the category name for a label ID."""
        return self.model.config.id2label.get(label_id, f"unknown_{label_id}")

    def get_num_classes(self) -> int:
        """Get the number of classes the model can predict."""
        return len(self.model.config.id2label)


class SemanticSegmentationModel:
    """
    Wrapper for semantic segmentation (simpler than panoptic).

    This can be used as an alternative to panoptic segmentation
    if you only need per-pixel class labels without instance separation.
    """

    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-large-ade-semantic",
        device: str = "auto",
    ):
        """
        Initialize the semantic segmentation model.

        Args:
            model_name: Hugging Face model identifier
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading segmentation model: {model_name}")
        print(f"Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    @torch.no_grad()
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Run semantic segmentation on an RGB image.

        Args:
            image: HxWx3 RGB image (numpy array, uint8)

        Returns:
            Tuple of:
                - segmentation_map: HxW array with class IDs
                - class_names: Dictionary mapping class_id to class name
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Post-process for semantic segmentation
        result = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        segmentation_map = result.cpu().numpy().astype(np.int32)

        # Build class name mapping
        class_names = {
            class_id: self.model.config.id2label[class_id]
            for class_id in np.unique(segmentation_map)
            if class_id in self.model.config.id2label
        }

        return segmentation_map, class_names
