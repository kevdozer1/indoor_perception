"""Indoor scene perception with RGB-D processing and panoptic segmentation."""

__version__ = "0.1.0"

from indoor_perception.pipeline import ScenePerceptionPipeline
from indoor_perception.visualizer import (
    create_grid_visualization,
    visualize_pipeline_result,
    visualize_ply_with_labels,
)

__all__ = [
    "ScenePerceptionPipeline",
    "visualize_ply_with_labels",
    "create_grid_visualization",
    "visualize_pipeline_result",
]
