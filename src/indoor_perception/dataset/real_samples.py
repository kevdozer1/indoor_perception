"""Download real RGB-D indoor scene samples from public sources."""

from pathlib import Path
from typing import Tuple
import ssl
from urllib.request import urlopen, Request

import numpy as np
from PIL import Image
from tqdm import tqdm

TUM_DEPTH_SCALE = 5000.0
SYNTHETIC_DEPTH_SCALE = 1000.0


def download_file(url: str, output_path: Path, ssl_context: ssl.SSLContext | None = None) -> None:
    """Download a file with progress bar."""
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    with urlopen(req, context=ssl_context) as response:
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            if total_size:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                f.write(response.read())


def download_tum_samples(output_dir: str, insecure: bool = False) -> Tuple[Path, float]:
    """
    Download real RGB-D samples from TUM RGB-D dataset.

    Returns:
        (scene_dir, depth_scale)
    """
    output_path = Path(output_dir)
    scene_dir = output_path / "scene_real_tum"

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    intrinsic_dir = scene_dir / "intrinsic"

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading real indoor RGB-D samples from TUM dataset...")
    print("Source: https://vision.in.tum.de/data/datasets/rgbd-dataset")

    # TUM RGB-D dataset uses Microsoft Kinect
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5

    intrinsics = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    intrinsic_file = intrinsic_dir / "intrinsic_color.txt"
    np.savetxt(intrinsic_file, intrinsics, fmt='%.6f')

    # Sample frames from TUM fr3/cabinet sequence (publicly available)
    base_url = "https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_cabinet/"

    samples = [
        ("rgb/1341847980.722988.png", "depth/1341847980.723020.png", "0"),
        ("rgb/1341847982.226762.png", "depth/1341847982.226830.png", "1"),
        ("rgb/1341847985.730384.png", "depth/1341847985.730451.png", "2"),
    ]

    print(f"\nDownloading {len(samples)} real indoor scene frames...")

    ssl_context = ssl._create_unverified_context() if insecure else None
    downloaded_frames = 0

    try:
        for rgb_path, depth_path, frame_id in samples:
            print(f"\nFrame {frame_id}:")

            # Download RGB
            rgb_url = base_url + rgb_path
            rgb_file = color_dir / f"{frame_id}.png"

            try:
                print("  Downloading RGB...")
                download_file(rgb_url, rgb_file, ssl_context=ssl_context)

                # Convert PNG to JPG for consistency
                img = Image.open(rgb_file)
                jpg_file = color_dir / f"{frame_id}.jpg"
                img.convert('RGB').save(jpg_file, quality=95)
                rgb_file.unlink()

            except Exception as e:
                print(f"  Warning: Could not download RGB frame: {e}")
                continue

            # Download Depth
            depth_url = base_url + depth_path
            depth_file = depth_dir / f"{frame_id}.png"

            try:
                print("  Downloading depth...")
                download_file(depth_url, depth_file, ssl_context=ssl_context)
            except Exception as e:
                print(f"  Warning: Could not download depth frame: {e}")
                continue

            downloaded_frames += 1

        if downloaded_frames == 0:
            raise RuntimeError(
                "No frames were downloaded. Try --insecure if your SSL certificates are missing."
            )

        print(f"\n[OK] Successfully downloaded {downloaded_frames} frame(s) to {scene_dir}")
        return scene_dir, TUM_DEPTH_SCALE

    except Exception as e:
        print(f"\nError downloading TUM samples: {e}")
        print("\nAlternative: Downloading Open3D Sample TUM RGB-D image...")
        return download_open3d_sample(output_path, sample="tum")


def download_open3d_sample(output_dir: Path, sample: str = "tum") -> Tuple[Path, float]:
    """Download a real RGB-D sample via Open3D's dataset helper."""
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError(f"Open3D is required for Open3D sample downloads: {e}") from e

    sample_map = {
        "tum": (o3d.data.SampleTUMRGBDImage, TUM_DEPTH_SCALE),
    }

    if sample not in sample_map:
        raise ValueError(f"Unsupported Open3D sample: {sample}")

    dataset_cls, depth_scale = sample_map[sample]
    dataset = dataset_cls()

    output_path = Path(output_dir)
    scene_dir = output_path / f"scene_open3d_{sample}"
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    intrinsic_dir = scene_dir / "intrinsic"

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    color_src = Path(dataset.color_path)
    depth_src = Path(dataset.depth_path)

    img = Image.open(color_src)
    img.convert('RGB').save(color_dir / "0.jpg", quality=95)
    Image.open(depth_src).save(depth_dir / "0.png")

    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5
    intrinsics = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    np.savetxt(intrinsic_dir / "intrinsic_color.txt", intrinsics, fmt='%.6f')

    print(f"\n[OK] Downloaded Open3D sample to {scene_dir}")
    return scene_dir, depth_scale


def create_realistic_samples(output_dir: Path) -> Tuple[Path, float]:
    """Create high-quality synthetic samples with realistic indoor scene features."""
    scene_dir = output_dir / "scene_real_synthetic"

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    intrinsic_dir = scene_dir / "intrinsic"

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5

    intrinsics = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    np.savetxt(intrinsic_dir / "intrinsic_color.txt", intrinsics, fmt='%.6f')

    print("\nCreating high-quality synthetic indoor scenes...")

    for frame_id in range(3):
        print(f"  Creating frame {frame_id}...")

        rgb, depth = create_indoor_scene_with_furniture(frame_id, (640, 480))

        Image.fromarray(rgb).save(color_dir / f"{frame_id}.jpg", quality=95)
        Image.fromarray(depth).save(depth_dir / f"{frame_id}.png")

    print(f"\n[OK] Created realistic indoor scenes at {scene_dir}")
    return scene_dir, SYNTHETIC_DEPTH_SCALE


def create_indoor_scene_with_furniture(seed: int, size: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Create a realistic indoor scene with furniture-like objects."""
    np.random.seed(seed * 42)
    width, height = size

    rgb = np.ones((height, width, 3), dtype=np.uint8) * 220
    depth = np.ones((height, width), dtype=np.uint16) * 3500

    colors = {
        'wall': (230, 225, 215),
        'floor': (180, 140, 100),
        'table': (139, 90, 43),
        'chair': (100, 80, 60),
        'cabinet': (180, 140, 110),
        'plant': (60, 120, 60),
        'laptop': (50, 50, 55),
        'book': (200, 180, 160),
    }

    rgb[:int(height*0.6), :] = colors['wall']

    for y in range(int(height*0.6), height):
        floor_color = np.array(colors['floor'])
        darkness = 1.0 - (y - height*0.6) / (height * 0.4) * 0.3
        rgb[y, :] = (floor_color * darkness).astype(np.uint8)

        depth[y, :] = int(1500 + (height - y) * 4)

    objects = [
        (50, 300, 200, 150, 2000, 'table'),
        (280, 320, 80, 130, 1800, 'chair'),
        (400, 280, 180, 200, 2500, 'cabinet'),
        (120, 350, 60, 40, 1900, 'laptop'),
        (450, 320, 50, 80, 2400, 'plant'),
    ]

    for x, y, w, h, d, color_key in objects:
        x2, y2 = min(x + w, width), min(y + h, height)

        base_color = np.array(colors[color_key])
        variation = np.random.randint(-15, 15, (y2-y, x2-x, 3))
        obj_color = np.clip(base_color + variation, 0, 255).astype(np.uint8)
        rgb[y:y2, x:x2] = obj_color

        depth[y:y2, x:x2] = d

        if y + 5 < height:
            shadow_color = (rgb[y:y+5, x:x2].astype(np.int16) * 0.7).astype(np.uint8)
            rgb[y:y+5, x:x2] = shadow_color

    for x in range(width):
        light_factor = 0.9 + 0.2 * (x / width)
        rgb[:, x] = np.clip(rgb[:, x].astype(np.float32) * light_factor, 0, 255).astype(np.uint8)

    noise_rgb = np.random.randint(-5, 5, (height, width, 3))
    rgb = np.clip(rgb.astype(np.int16) + noise_rgb, 0, 255).astype(np.uint8)

    noise_depth = np.random.randint(-30, 30, (height, width))
    depth = np.clip(depth.astype(np.int32) + noise_depth, 500, 5000).astype(np.uint16)

    return rgb, depth
