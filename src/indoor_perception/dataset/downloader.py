"""Utilities for downloading sample RGB-D data."""

import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from tqdm import tqdm


class DownloadProgressBar:
    """Progress bar for URL downloads."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading')

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_sample_scannet(
    output_dir: str,
    url: Optional[str] = None,
    extract: bool = True,
) -> Path:
    """
    Download sample ScanNet data.

    Note: Official ScanNet data requires registration at http://www.scan-net.org/
    This function can download from a provided URL or use a pre-prepared sample.

    Args:
        output_dir: Directory to save the data
        url: Optional URL to download from. If None, provides instructions.
        extract: Whether to extract the downloaded archive

    Returns:
        Path to the downloaded/extracted data directory

    Raises:
        ValueError: If no URL provided and no default available
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if url is None:
        # Provide instructions for obtaining ScanNet data
        instructions = """
ScanNet Dataset Download Instructions:
======================================

Official ScanNet data requires registration and agreement to terms of use:
1. Visit http://www.scan-net.org/
2. Register and request access
3. Download sample scenes using the provided download script

Alternative for testing:
- You can create a small sample dataset manually following this structure:

  data/
  └── scene0000_00/
      ├── color/
      │   ├── 0.jpg
      │   ├── 1.jpg
      │   └── ...
      ├── depth/
      │   ├── 0.png  (uint16, depth in millimeters)
      │   ├── 1.png
      │   └── ...
      └── intrinsic/
          └── intrinsic_color.txt  (4x4 camera matrix)

- Or use this function with a URL parameter pointing to a prepared sample archive
        """
        print(instructions)
        raise ValueError(
            "No download URL provided. Please see instructions above for obtaining ScanNet data."
        )

    # Download the file
    download_path = output_path / "scannet_sample.zip"
    print(f"Downloading from {url}...")

    try:
        urlretrieve(url, download_path, DownloadProgressBar())
        print(f"Downloaded to {download_path}")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    # Extract if requested
    if extract:
        print(f"Extracting to {output_path}...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print(f"Extraction complete")

        # Clean up zip file
        download_path.unlink()

    return output_path


def create_sample_scene(
    output_dir: str,
    scene_id: str = "scene0000_00",
    num_frames: int = 5,
    image_size: tuple = (640, 480),
) -> Path:
    """
    Create a minimal synthetic sample scene for testing.

    This creates dummy RGB and depth images with proper directory structure
    for testing the dataset loader without real ScanNet data.

    Args:
        output_dir: Directory to create the sample scene
        scene_id: Scene identifier
        num_frames: Number of frames to generate
        image_size: (width, height) of images

    Returns:
        Path to the created scene directory
    """
    import numpy as np
    from PIL import Image

    output_path = Path(output_dir) / scene_id
    color_dir = output_path / "color"
    depth_dir = output_path / "depth"
    intrinsic_dir = output_path / "intrinsic"

    # Create directories
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    width, height = image_size

    # Create sample intrinsics (typical values for ScanNet)
    fx = fy = 577.87  # focal length in pixels
    cx, cy = width / 2, height / 2
    intrinsics = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # Save intrinsics
    intrinsic_file = intrinsic_dir / "intrinsic_color.txt"
    np.savetxt(intrinsic_file, intrinsics, fmt='%.6f')

    # Generate sample frames
    for i in range(num_frames):
        # Create synthetic RGB image (gradient pattern)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = (np.linspace(0, 255, width) * np.ones((height, 1))).astype(np.uint8)
        rgb[:, :, 1] = (np.linspace(0, 255, height)[:, None] * np.ones((1, width))).astype(np.uint8)
        rgb[:, :, 2] = 128

        # Create synthetic depth (planar surface with noise)
        base_depth = 2000  # 2 meters in mm
        noise = np.random.randint(-100, 100, (height, width))
        depth = np.clip(base_depth + noise, 500, 5000).astype(np.uint16)

        # Save images
        Image.fromarray(rgb).save(color_dir / f"{i}.jpg")
        Image.fromarray(depth).save(depth_dir / f"{i}.png")

    print(f"Created sample scene with {num_frames} frames at {output_path}")
    return output_path
