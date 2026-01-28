# Indoor Scene Perception

A modular Python pipeline for indoor scene perception using RGB-D data and panoptic segmentation. Processes RGB-D images from datasets like ScanNet, performs semantic/panoptic segmentation, and projects results into 3D point clouds with semantic labels.

## Features

- **RGB-D Dataset Loading**: Support for ScanNet format with extensible architecture
- **Panoptic Segmentation**: Using pretrained Mask2Former models from Hugging Face
- **3D Projection**: Convert 2D segmentation + depth to 3D point clouds
- **PLY Export**: Standard 3D format with colors and semantic labels
- **Advanced Visualization**: Open3D-based rendering with semantic label coloring
- **Grid Visualizations**: Side-by-side RGB, segmentation, and 3D views
- **Headless Rendering**: Generate visualizations without GUI (perfect for servers)
- **Complete Demo**: End-to-end demo script with visual outputs
- **Modular Design**: Clean separation of concerns for easy extension
- **Comprehensive Tests**: Full test coverage for all components

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/indoor-perception.git
cd indoor-perception

# Use main branch
git checkout main

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Run the Complete Demo

The fastest way to see the pipeline in action:

```bash
# Run complete demo (creates data, runs pipeline, generates visualizations)
python scripts/demo.py

# Output will be in output/demo/ with grid visualizations you can screenshot
```

This will:
1. Create synthetic RGB-D data
2. Run panoptic segmentation
3. Generate 3D point clouds
4. Create visualization grids (RGB + segmentation + 3D renders)

### Run the Demo on Real Indoor RGB-D Data (TUM)

```bash
# Download TUM RGB-D samples and run the pipeline
python scripts/demo.py --data-source tum --data data/real
```

### Run the Demo on Real Indoor RGB-D Data (Open3D Sample)

```bash
# Download an Open3D-provided real RGB-D sample and run the pipeline
python scripts/demo.py --data-source open3d --data data/real
```

### Manual Usage

### 1. Download or Create Sample Data

```bash
# Create synthetic sample data for testing
python scripts/download_sample.py --output data/ --create-synthetic --num-frames 10

# Or download from a URL (if you have ScanNet access)
python scripts/download_sample.py --output data/ --url <your-scannet-url>

# Download a few real indoor RGB-D samples from TUM
python scripts/download_real_samples.py --output data/real --source tum

# If SSL certificate errors occur:
python scripts/download_real_samples.py --output data/real --source tum --insecure

# Download a real RGB-D sample via Open3D
python scripts/download_real_samples.py --output data/real --source open3d --open3d-sample tum
```

### 2. Run the Pipeline

```bash
# Process all frames
python scripts/run_pipeline.py --data data/ --output output/

# Process specific scenes
python scripts/run_pipeline.py --data data/ --output output/ --scenes scene0000_00

# Limit frames for testing
python scripts/run_pipeline.py --data data/ --output output/ --max-frames 5

# Save segmentation maps for later visualization
python scripts/run_pipeline.py --data data/ --output output/ --save-segmentation-maps
```

### 3. Visualize Results

Open the generated PLY files in:
- [MeshLab](https://www.meshlab.net/)
- [CloudCompare](https://www.cloudcompare.org/)
- Any PLY viewer

## Usage Examples

### Basic Pipeline Usage

```python
from indoor_perception.dataset import ScanNetDataset
from indoor_perception.pipeline import ScenePerceptionPipeline

# Load dataset
dataset = ScanNetDataset(data_root="data/")

# Initialize pipeline
pipeline = ScenePerceptionPipeline(
    dataset=dataset,
    model_name="facebook/mask2former-swin-large-coco-panoptic",
    device="auto",  # Uses CUDA if available
)

# Process a single frame
result = pipeline.process_frame(idx=0, output_path="output/frame_0.ply")

print(f"Generated {len(result['points'])} 3D points")
print(f"Found {len(result['segment_info'])} segments")

# Process all frames
results = pipeline.process_all(output_dir="output/")
```

### Custom Dataset

```python
from indoor_perception.dataset import create_sample_scene

# Create a synthetic scene for testing
scene_path = create_sample_scene(
    output_dir="data/",
    scene_id="my_scene",
    num_frames=10,
    image_size=(640, 480),
)
```

### Manual Processing

```python
from indoor_perception.segmentation import PanopticSegmentationModel
from indoor_perception.projection import DepthProjector
from indoor_perception.io import write_ply
import numpy as np

# Load your RGB-D data
rgb = ...  # HxWx3 numpy array
depth = ...  # HxW numpy array (in meters)
intrinsics = ...  # 3x3 camera matrix

# Run segmentation
model = PanopticSegmentationModel(device="cuda")
seg_map, seg_info = model.segment(rgb)

# Project to 3D
projector = DepthProjector(intrinsics)
points, colors, labels = projector.project_segmented_scene(
    depth, seg_map, rgb=rgb
)

# Save as PLY
write_ply("output.ply", points, colors, labels)
```

### Visualization

```python
from indoor_perception.visualizer import (
    visualize_ply_with_labels,
    create_grid_visualization,
    visualize_pipeline_result,
)

# Render a PLY file to image (headless, no GUI)
image = visualize_ply_with_labels(
    "output.ply",
    output_path="visualization.png",
    use_label_colors=True,  # Color by semantic labels
    image_size=(800, 600),
)

# Create grid visualization with RGB, segmentation, and 3D view
create_grid_visualization(
    rgb_image=rgb,
    segmentation_mask=seg_map,
    point_cloud_image=image,
    output_path="grid.png",
)

# Complete pipeline visualization (all-in-one)
visualize_pipeline_result(
    rgb_image=rgb,
    segmentation_mask=seg_map,
    ply_path="output.ply",
    output_path="viz/",
    frame_id="frame_001",
)
# This generates: frame_001_grid.png, frame_001_3d.png, frame_001_segmentation.png
```

You can also visualize a batch with the CLI (uses saved segmentation maps if present):

```bash
python scripts/visualize_results.py --data data/ --output output/viz --ply-dir output/
```

### SAM 2 Masks for Image-Only Tests

You can run SAM 2 automatic mask generation on a folder of images (e.g., `data/facebook/`)
and inspect the overlays to compare mask quality.

Install SAM 2 (official repo) and download checkpoints:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

# Download checkpoints
cd checkpoints
./download_ckpts.sh
```

Note: SAM 2 requires Python 3.10+ and recent PyTorch. On Windows, the SAM 2
team recommends using WSL for installation.

Run SAM 2 on local images:

```bash
python scripts/run_sam2_auto_masks.py \
  --input-dir data/facebook \
  --output-dir output/sam2/facebook \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --save-overlays \
  --save-mask-map
```

### SAM 2 + Point Cloud (Image Folder)

Use SAM 2 masks plus a synthetic constant depth to generate point clouds from
image-only datasets like `data/facebook`:

```bash
python scripts/run_image_folder_pipeline.py \
  --input-dir data/facebook \
  --pattern "*_01.jpg" \
  --output output/facebook_sam2_pcd \
  --checkpoint .cache/sam2/checkpoints/sam2.1_hiera_small.pt \
  --model-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --device cpu \
  --constant-depth 2.0 \
  --save-segmentation-maps
```

### Add Screenshots to This README

1. Run a demo or pipeline to generate visualization PNGs.
2. Copy the images you want into `docs/images/`.
3. Add markdown image tags to this README.

Example (update filenames as needed):
```markdown
![Demo grid](docs/images/scene_open3d_tum_0_grid.png)
```

## Project Structure

```
indoor-perception/
├── src/indoor_perception/
│   ├── dataset/          # Dataset loaders
│   ├── segmentation/     # Segmentation models
│   ├── projection/       # 3D projection utilities
│   ├── io/              # Point cloud I/O
│   ├── visualizer.py    # Visualization utilities
│   └── pipeline.py      # Main pipeline
├── tests/               # Comprehensive tests
├── scripts/             # CLI utilities
│   ├── demo.py          # Complete demo script
│   ├── download_sample.py
│   └── run_pipeline.py
├── examples/            # Usage examples
└── pyproject.toml       # Project configuration
```

## Architecture

### Components

1. **Dataset Module** ([src/indoor_perception/dataset/](src/indoor_perception/dataset/))
   - Abstract `RGBDDataset` interface
   - `ScanNetDataset` implementation
   - Sample data download utilities

2. **Segmentation Module** ([src/indoor_perception/segmentation/](src/indoor_perception/segmentation/))
   - Mask2Former wrapper for panoptic segmentation
   - Support for multiple pretrained models
   - GPU/CPU inference

3. **Projection Module** ([src/indoor_perception/projection/](src/indoor_perception/projection/))
   - Depth-based 3D projection using camera intrinsics
   - Segment label assignment to 3D points
   - Pinhole camera model implementation

4. **I/O Module** ([src/indoor_perception/io/](src/indoor_perception/io/))
   - PLY format read/write
   - Support for colors and semantic labels

5. **Visualization Module** ([src/indoor_perception/visualizer.py](src/indoor_perception/visualizer.py))
   - Open3D-based headless rendering
   - Semantic label coloring
   - Grid visualizations (RGB + segmentation + 3D)
   - PNG export without GUI

6. **Pipeline** ([src/indoor_perception/pipeline.py](src/indoor_perception/pipeline.py))
   - End-to-end orchestration
   - Batch processing
   - Metadata export

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_projection.py -v

# Skip slow tests (model downloads)
pytest tests/ -v -m "not slow"

# Run with coverage
pytest tests/ --cov=indoor_perception --cov-report=html
```

## Models

The pipeline supports any Mask2Former model from Hugging Face. Recommended models:

- `facebook/mask2former-swin-large-coco-panoptic` (default, 80 classes)
- `facebook/mask2former-swin-base-coco-panoptic` (smaller, faster)
- `facebook/mask2former-swin-large-ade-semantic` (150 classes, indoor-focused)

## Data Format

### ScanNet Structure

```
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
```

### Output PLY Format

PLY files contain:
- `x, y, z`: 3D coordinates
- `red, green, blue`: RGB colors (0-255)
- `semantic_label`: Segment ID (optional)

Accompanying JSON files contain segment metadata (label names, scores).

## Performance

- **GPU**: ~1-2 seconds per frame (640x480) on RTX 3090
- **CPU**: ~10-15 seconds per frame (640x480)
- **Memory**: ~4GB GPU memory for large Mask2Former model

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Support for Replica dataset
- [ ] Instance segmentation tracking across frames
- [ ] Mesh reconstruction from point clouds
- [ ] Integration with SLAM systems
- [ ] Real-time processing optimization

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{indoor_perception,
  title={Indoor Scene Perception Pipeline},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/indoor-perception}
}
```

## Acknowledgments

- [ScanNet Dataset](http://www.scan-net.org/)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## Troubleshooting

### CUDA Out of Memory

Use a smaller model or reduce image resolution:

```python
pipeline = ScenePerceptionPipeline(
    dataset=dataset,
    model_name="facebook/mask2former-swin-base-coco-panoptic",  # Smaller model
    device="cuda",
)
```

### Slow Processing

Enable GPU acceleration and use batch processing:

```bash
python scripts/run_pipeline.py --data data/ --output output/ --device cuda
```

### ScanNet Data Access

ScanNet requires registration at [http://www.scan-net.org/](http://www.scan-net.org/). Alternatively, use the synthetic data generator for testing:

```bash
python scripts/download_sample.py --output data/ --create-synthetic
```
