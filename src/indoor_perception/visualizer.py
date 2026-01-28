"""Visualization utilities for point clouds and segmentation results."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Force headless rendering
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image

from indoor_perception.io import read_ply


def generate_label_colors(num_labels: int, seed: int = 42) -> np.ndarray:
    """
    Generate distinct colors for semantic labels.

    Args:
        num_labels: Number of unique labels
        seed: Random seed for reproducibility

    Returns:
        Nx3 array of RGB colors (0-1 range)
    """
    np.random.seed(seed)
    colors = np.random.rand(num_labels, 3)

    # Ensure colors are reasonably bright and distinct
    colors = np.clip(colors * 0.8 + 0.2, 0, 1)

    return colors


def apply_label_colors(
    labels: np.ndarray,
    label_colors: Optional[np.ndarray] = None,
    num_labels: Optional[int] = None,
) -> np.ndarray:
    """
    Map semantic labels to RGB colors.

    Args:
        labels: N array of integer labels
        label_colors: Optional Lx3 array of colors for each label
        num_labels: Number of unique labels (used if label_colors not provided)

    Returns:
        Nx3 array of RGB colors (0-1 range)
    """
    unique_labels = np.unique(labels)

    if label_colors is None:
        if num_labels is None:
            num_labels = len(unique_labels)
        label_colors = generate_label_colors(num_labels)

    # Map labels to colors
    colors = np.zeros((len(labels), 3))
    for label in unique_labels:
        mask = labels == label
        color_idx = label % len(label_colors)
        colors[mask] = label_colors[color_idx]

    return colors


def render_point_cloud_to_image(
    points: np.ndarray,
    colors: np.ndarray,
    image_size: Tuple[int, int] = (800, 600),
    view_params: Optional[Dict] = None,
) -> np.ndarray:
    """
    Render a point cloud to an image without GUI.

    Args:
        points: Nx3 array of point coordinates
        colors: Nx3 array of RGB colors (0-1 range)
        image_size: (width, height) of output image
        view_params: Optional dict with view parameters:
            - 'zoom': Camera zoom factor
            - 'front': Camera front vector [x, y, z]
            - 'lookat': Camera look-at point [x, y, z]
            - 'up': Camera up vector [x, y, z]

    Returns:
        HxWx3 image array (uint8)
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create visualizer (offscreen)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=image_size[0], height=image_size[1])
    vis.add_geometry(pcd)

    # Set view parameters
    view_control = vis.get_view_control()

    if view_params is not None:
        if "zoom" in view_params:
            view_control.set_zoom(view_params["zoom"])
        if "front" in view_params:
            view_control.set_front(view_params["front"])
        if "lookat" in view_params:
            view_control.set_lookat(view_params["lookat"])
        if "up" in view_params:
            view_control.set_up(view_params["up"])
    else:
        # Default: view from angle
        view_control.set_zoom(0.6)
        view_control.set_front([0.5, -0.5, -0.6])
        view_control.set_lookat(np.mean(points, axis=0))
        view_control.set_up([0, -1, 0])

    # Render
    vis.poll_events()
    vis.update_renderer()

    # Capture image
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    # Convert to uint8 array
    image_array = (np.asarray(image) * 255).astype(np.uint8)

    return image_array


def visualize_ply_with_labels(
    ply_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    image_size: Tuple[int, int] = (800, 600),
    use_label_colors: bool = True,
) -> np.ndarray:
    """
    Load a PLY file and render it with semantic label colors.

    Args:
        ply_path: Path to PLY file
        output_path: Optional path to save rendered image
        image_size: (width, height) of output image
        use_label_colors: If True, color by labels; otherwise use RGB colors from PLY

    Returns:
        Rendered image as HxWx3 array (uint8)
    """
    # Load PLY
    data = read_ply(ply_path)
    points = data["points"]

    # Choose colors
    if use_label_colors and "labels" in data:
        labels = data["labels"]
        colors = apply_label_colors(labels)
    elif "colors" in data:
        colors = data["colors"] / 255.0  # Convert to 0-1 range
    else:
        # Default white
        colors = np.ones((len(points), 3))

    # Render
    image = render_point_cloud_to_image(points, colors, image_size)

    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output_path)

    return image


def create_segmentation_overlay(
    rgb_image: np.ndarray,
    segmentation_mask: np.ndarray,
    alpha: float = 0.5,
    label_colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create an overlay of segmentation mask on RGB image.

    Args:
        rgb_image: HxWx3 RGB image (uint8)
        segmentation_mask: HxW segmentation mask with integer labels
        alpha: Transparency of overlay (0=transparent, 1=opaque)
        label_colors: Optional Lx3 array of colors (0-1 range) for labels

    Returns:
        HxWx3 overlay image (uint8)
    """
    h, w = segmentation_mask.shape

    # Generate colors if not provided
    unique_labels = np.unique(segmentation_mask)
    if label_colors is None:
        label_colors = generate_label_colors(len(unique_labels))

    # Create colored segmentation mask
    seg_colored = np.zeros((h, w, 3), dtype=np.float32)
    for i, label in enumerate(unique_labels):
        mask = segmentation_mask == label
        color_idx = label % len(label_colors)
        seg_colored[mask] = label_colors[color_idx]

    # Convert to uint8
    seg_colored = (seg_colored * 255).astype(np.uint8)

    # Blend with RGB image
    overlay = (alpha * seg_colored + (1 - alpha) * rgb_image).astype(np.uint8)

    return overlay


def create_grid_visualization(
    rgb_image: np.ndarray,
    segmentation_mask: np.ndarray,
    point_cloud_image: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    titles: Optional[Tuple[str, str, str]] = None,
) -> np.ndarray:
    """
    Create a grid visualization with RGB, segmentation overlay, and 3D render.

    Args:
        rgb_image: HxWx3 RGB input image
        segmentation_mask: HxW segmentation mask
        point_cloud_image: HxWx3 rendered point cloud image
        output_path: Optional path to save the grid
        titles: Optional tuple of (rgb_title, seg_title, 3d_title)

    Returns:
        Grid image as array (uint8)
    """
    if titles is None:
        titles = ("RGB Input", "Segmentation Overlay", "3D Point Cloud")

    # Create segmentation overlay
    seg_overlay = create_segmentation_overlay(rgb_image, segmentation_mask, alpha=0.5)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # RGB input
    axes[0].imshow(rgb_image)
    axes[0].set_title(titles[0], fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Segmentation overlay
    axes[1].imshow(seg_overlay)
    axes[1].set_title(titles[1], fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # 3D point cloud
    axes[2].imshow(point_cloud_image)
    axes[2].set_title(titles[2], fontsize=14, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    grid_image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    return grid_image


def visualize_pipeline_result(
    rgb_image: np.ndarray,
    segmentation_mask: np.ndarray,
    ply_path: Union[str, Path],
    output_path: Union[str, Path],
    frame_id: str = "frame",
) -> None:
    """
    Create complete visualization for a pipeline result.

    Generates:
    - Grid visualization (RGB + segmentation + 3D)
    - Standalone 3D render with label colors
    - Segmentation overlay

    Args:
        rgb_image: HxWx3 RGB input image
        segmentation_mask: HxW segmentation mask
        ply_path: Path to output PLY file
        output_path: Base path for output images
        frame_id: Frame identifier for filenames
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Render point cloud with label colors
    print(f"  Rendering 3D point cloud for {frame_id}...")
    point_cloud_image = visualize_ply_with_labels(
        ply_path,
        output_path=output_path / f"{frame_id}_3d.png",
        use_label_colors=True,
    )

    # Create segmentation overlay
    print(f"  Creating segmentation overlay for {frame_id}...")
    seg_overlay = create_segmentation_overlay(rgb_image, segmentation_mask, alpha=0.5)
    Image.fromarray(seg_overlay).save(output_path / f"{frame_id}_segmentation.png")

    # Create grid visualization
    print(f"  Creating grid visualization for {frame_id}...")
    create_grid_visualization(
        rgb_image=rgb_image,
        segmentation_mask=segmentation_mask,
        point_cloud_image=point_cloud_image,
        output_path=output_path / f"{frame_id}_grid.png",
        titles=(
            f"RGB Input ({frame_id})",
            f"Segmentation Overlay ({frame_id})",
            f"3D Point Cloud ({frame_id})",
        ),
    )

    print(f"  [OK] Visualizations saved to {output_path}")


def create_comparison_grid(
    image_paths: list,
    titles: list,
    output_path: Union[str, Path],
    grid_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Create a comparison grid from multiple images.

    Args:
        image_paths: List of paths to images
        titles: List of titles for each image
        output_path: Path to save the comparison grid
        grid_size: Optional (rows, cols) for grid layout
    """
    n_images = len(image_paths)

    if grid_size is None:
        # Auto-determine grid size
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx, (img_path, title) in enumerate(zip(image_paths, titles)):
        row = idx // cols
        col = idx % cols

        img = Image.open(img_path)
        axes[row, col].imshow(img)
        axes[row, col].set_title(title, fontsize=12, fontweight="bold")
        axes[row, col].axis("off")

    # Hide unused subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
