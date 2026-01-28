"""ScanNet RGB-D dataset loader."""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image

from indoor_perception.dataset.base import RGBDDataset


class ScanNetDataset(RGBDDataset):
    """
    ScanNet RGB-D dataset loader.

    Expected directory structure:
        data_root/
        ├── scene0000_00/
        │   ├── color/
        │   │   ├── 0.jpg
        │   │   ├── 1.jpg
        │   │   └── ...
        │   ├── depth/
        │   │   ├── 0.png
        │   │   ├── 1.png
        │   │   └── ...
        │   └── intrinsic/
        │       ├── intrinsic_color.txt
        │       └── intrinsic_depth.txt
        └── scene0000_01/
            └── ...
    """

    def __init__(
        self,
        data_root: str,
        scene_ids: Optional[List[str]] = None,
        depth_scale: float = 1000.0,
        max_frames_per_scene: Optional[int] = None,
    ):
        """
        Initialize ScanNet dataset.

        Args:
            data_root: Root directory containing scene folders
            scene_ids: List of scene IDs to load (e.g., ['scene0000_00']).
                      If None, loads all scenes in data_root.
            depth_scale: Scale factor for depth values (ScanNet uses mm, so 1000.0 to get meters)
            max_frames_per_scene: Maximum frames to load per scene (for testing/debugging)
        """
        self.data_root = Path(data_root)
        self.depth_scale = depth_scale
        self.max_frames_per_scene = max_frames_per_scene

        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")

        # Find all scenes
        if scene_ids is None:
            scene_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        else:
            scene_dirs = [self.data_root / scene_id for scene_id in scene_ids]

        # Build frame list
        self.frames: List[Dict[str, Any]] = []
        for scene_dir in scene_dirs:
            if not scene_dir.exists():
                print(f"Warning: Scene directory not found: {scene_dir}")
                continue
            self._load_scene(scene_dir)

        if len(self.frames) == 0:
            raise ValueError(f"No frames found in {self.data_root}")

        print(f"Loaded {len(self.frames)} frames from {len(scene_dirs)} scenes")

    def _load_scene(self, scene_dir: Path) -> None:
        """Load all frames from a scene directory."""
        color_dir = scene_dir / "color"
        depth_dir = scene_dir / "depth"
        intrinsic_file = scene_dir / "intrinsic" / "intrinsic_color.txt"

        if not color_dir.exists() or not depth_dir.exists():
            print(f"Warning: Missing color or depth directory in {scene_dir}")
            return

        # Load intrinsics for this scene
        if intrinsic_file.exists():
            intrinsics = self._load_intrinsics(intrinsic_file)
        else:
            print(f"Warning: No intrinsics found for {scene_dir}, using default")
            # Default intrinsics for ScanNet (approximate)
            intrinsics = np.array([
                [577.87, 0, 319.5],
                [0, 577.87, 239.5],
                [0, 0, 1]
            ], dtype=np.float32)

        # Find all color frames
        color_files = sorted(color_dir.glob("*.jpg"))
        if self.max_frames_per_scene:
            color_files = color_files[:self.max_frames_per_scene]

        for color_file in color_files:
            frame_id = color_file.stem  # e.g., "0", "1", etc.
            depth_file = depth_dir / f"{frame_id}.png"

            if not depth_file.exists():
                continue

            self.frames.append({
                "scene_id": scene_dir.name,
                "frame_id": frame_id,
                "color_path": color_file,
                "depth_path": depth_file,
                "intrinsics": intrinsics,
            })

    def _load_intrinsics(self, intrinsic_file: Path) -> np.ndarray:
        """
        Load camera intrinsics from ScanNet format.

        ScanNet intrinsic files contain a 4x4 matrix with:
            fx  0   cx  0
            0   fy  cy  0
            0   0   1   0
            0   0   0   1
        """
        with open(intrinsic_file, 'r') as f:
            lines = f.readlines()

        # Parse 4x4 matrix
        matrix = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            values = [float(x) for x in line.split()]
            matrix.append(values)

        K = np.array(matrix, dtype=np.float32)

        # Extract 3x3 intrinsics
        if K.shape == (4, 4):
            K = K[:3, :3]
        elif K.shape != (3, 3):
            raise ValueError(f"Invalid intrinsics shape: {K.shape}")

        return K

    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single frame from the dataset.

        Args:
            idx: Frame index

        Returns:
            Dictionary containing:
                - 'rgb': HxWx3 numpy array (uint8)
                - 'depth': HxW numpy array (float32) in meters
                - 'intrinsics': 3x3 numpy array (float32)
                - 'frame_id': str
                - 'scene_id': str
        """
        if idx < 0 or idx >= len(self.frames):
            raise IndexError(f"Index {idx} out of range [0, {len(self.frames)})")

        frame_info = self.frames[idx]

        # Load RGB
        rgb = np.array(Image.open(frame_info["color_path"]))

        # Load depth (stored as uint16 in mm, convert to float32 in meters)
        depth_raw = np.array(Image.open(frame_info["depth_path"]))
        depth = depth_raw.astype(np.float32) / self.depth_scale

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": frame_info["intrinsics"],
            "frame_id": frame_info["frame_id"],
            "scene_id": frame_info["scene_id"],
        }

    def get_frame_path(self, idx: int) -> str:
        """Get the RGB file path for a frame."""
        if idx < 0 or idx >= len(self.frames):
            raise IndexError(f"Index {idx} out of range [0, {len(self.frames)})")
        return str(self.frames[idx]["color_path"])
