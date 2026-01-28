"""3D projection utilities for converting depth maps to point clouds."""

from typing import Optional, Tuple

import numpy as np


class DepthProjector:
    """Projects 2D depth maps to 3D point clouds using camera intrinsics."""

    def __init__(self, intrinsics: Optional[np.ndarray] = None):
        """
        Initialize the depth projector.

        Args:
            intrinsics: Optional 3x3 camera intrinsics matrix.
                       If None, must be provided in project_to_3d calls.
        """
        self.intrinsics = intrinsics

    def project_to_3d(
        self,
        depth: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Project a depth map to 3D points using camera intrinsics.

        Uses the pinhole camera model:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth

        Args:
            depth: HxW depth map in meters
            intrinsics: Optional 3x3 camera intrinsics matrix.
                       Uses self.intrinsics if not provided.
            rgb: Optional HxWx3 RGB image to get point colors

        Returns:
            Tuple of:
                - points: Nx3 array of 3D points (x, y, z)
                - colors: Nx3 array of RGB colors if rgb provided, else None

        Raises:
            ValueError: If no intrinsics provided or shapes don't match
        """
        K = intrinsics if intrinsics is not None else self.intrinsics
        if K is None:
            raise ValueError("Camera intrinsics must be provided")

        if K.shape != (3, 3):
            raise ValueError(f"Intrinsics must be 3x3 matrix, got shape {K.shape}")

        h, w = depth.shape
        if rgb is not None and rgb.shape[:2] != (h, w):
            raise ValueError(
                f"RGB shape {rgb.shape[:2]} doesn't match depth shape {depth.shape}"
            )

        # Extract intrinsic parameters
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create pixel coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Unproject to 3D (vectorized)
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack coordinates and reshape to Nx3
        points_3d = np.stack([x, y, z], axis=-1)
        points_3d = points_3d.reshape(-1, 3)

        # Filter out invalid points (zero or negative depth)
        valid_mask = points_3d[:, 2] > 0
        points_3d = points_3d[valid_mask]

        # Extract colors if RGB image provided
        colors = None
        if rgb is not None:
            colors = rgb.reshape(-1, 3)[valid_mask]

        return points_3d, colors

    def apply_segmentation(
        self,
        points: np.ndarray,
        segmentation_mask: np.ndarray,
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply segmentation labels to 3D points.

        Args:
            points: Nx3 array of 3D points
            segmentation_mask: HxW segmentation mask with integer segment IDs
            depth: HxW depth map (used to align segmentation with points)
            rgb: Optional HxWx3 RGB image (used to filter valid points)

        Returns:
            Tuple of:
                - points: Filtered Nx3 array of points (only valid depths)
                - labels: N array of segment IDs for each point

        Raises:
            ValueError: If shapes don't match
        """
        h, w = depth.shape
        if segmentation_mask.shape != (h, w):
            raise ValueError(
                f"Segmentation shape {segmentation_mask.shape} doesn't match "
                f"depth shape {depth.shape}"
            )

        if rgb is not None and rgb.shape[:2] != (h, w):
            raise ValueError(
                f"RGB shape {rgb.shape[:2]} doesn't match depth shape {depth.shape}"
            )

        # Flatten segmentation mask
        labels_flat = segmentation_mask.reshape(-1)

        # Create valid mask (same logic as project_to_3d)
        valid_mask = depth.reshape(-1) > 0
        labels = labels_flat[valid_mask]

        if len(labels) != len(points):
            raise ValueError(
                f"Number of labels ({len(labels)}) doesn't match "
                f"number of points ({len(points)})"
            )

        return points, labels

    def project_segmented_scene(
        self,
        depth: np.ndarray,
        segmentation_mask: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Complete pipeline: project depth to 3D and apply segmentation labels.

        Args:
            depth: HxW depth map in meters
            segmentation_mask: HxW segmentation mask with integer segment IDs
            intrinsics: Optional 3x3 camera intrinsics matrix
            rgb: Optional HxWx3 RGB image

        Returns:
            Tuple of:
                - points: Nx3 array of 3D points
                - colors: Nx3 array of RGB colors (or None if no RGB)
                - labels: N array of segment IDs

        Example:
            >>> projector = DepthProjector(intrinsics)
            >>> points, colors, labels = projector.project_segmented_scene(
            ...     depth, segmentation_mask, rgb=rgb_image
            ... )
        """
        # Project to 3D
        points, colors = self.project_to_3d(depth, intrinsics, rgb)

        # Apply segmentation
        points, labels = self.apply_segmentation(points, segmentation_mask, depth, rgb)

        return points, colors, labels


def compute_intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Construct a 3x3 camera intrinsics matrix.

    Args:
        fx: Focal length in x direction (pixels)
        fy: Focal length in y direction (pixels)
        cx: Principal point x coordinate (pixels)
        cy: Principal point y coordinate (pixels)

    Returns:
        3x3 intrinsics matrix K

    Example:
        >>> K = compute_intrinsics_matrix(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
    """
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float32)
    return K
