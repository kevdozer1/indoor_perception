"""Abstract base class for RGB-D datasets."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class RGBDDataset(ABC):
    """Abstract base class for RGB-D datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single frame from the dataset.

        Args:
            idx: Frame index

        Returns:
            Dictionary containing:
                - 'rgb': HxWx3 numpy array (uint8) - RGB image
                - 'depth': HxW numpy array (float32) - Depth map in meters
                - 'intrinsics': 3x3 numpy array (float32) - Camera intrinsics
                - 'frame_id': str - Unique frame identifier
                - Additional dataset-specific metadata
        """
        pass

    @abstractmethod
    def get_frame_path(self, idx: int) -> str:
        """
        Get the file path for a frame.

        Args:
            idx: Frame index

        Returns:
            Path to the frame's RGB or depth file
        """
        pass
