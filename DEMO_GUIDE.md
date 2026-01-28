# Demo Guide - Indoor Scene Perception

This guide will help you run the complete demo and generate visual outputs.

## Quick Start

### 1. Install Dependencies

```bash
# Install the package with all dependencies (including open3d and matplotlib)
pip install -e .
```

### 2. Run the Demo

```bash
# Run the complete demo - this will take a few minutes on first run
python scripts/demo.py
```

**What the demo does:**
1. Creates synthetic RGB-D data (3 frames by default)
2. Loads the Mask2Former segmentation model (~2GB download on first run)
3. Runs panoptic segmentation on each frame
4. Projects segmentation to 3D point clouds
5. Generates visualizations with Open3D (headless, no GUI)
6. Saves everything to `output/demo/`

### 3. View Results

After running, check these directories:

```
output/demo/
├── point_clouds/          # PLY files (3D point clouds)
├── visualizations/        # PNG images you can screenshot
│   ├── *_grid.png        # 3-panel grid (RGB, segmentation, 3D)
│   ├── *_3d.png          # 3D point cloud render
│   └── *_segmentation.png # Segmentation overlay
└── DEMO_SUMMARY.txt      # Summary of results
```

**Files to screenshot:**
- `output/demo/visualizations/*_grid.png` - Main visualization grids

## Demo Options

### Run with custom settings

```bash
# Create more scenes and frames
python scripts/demo.py --num-scenes 2 --num-frames 5

# Use CPU instead of GPU
python scripts/demo.py --device cpu

# Use a smaller/faster model
python scripts/demo.py --model facebook/mask2former-swin-base-coco-panoptic

# Custom output directory
python scripts/demo.py --output my_demo_output/
```

### Demo with existing data

If you have real ScanNet data:

```bash
# Skip synthetic data creation and use your own
python scripts/run_pipeline.py --data path/to/scannet --output output/real_data
```

Then visualize the results:

```python
from indoor_perception.visualizer import visualize_ply_with_labels

# Render any PLY file
visualize_ply_with_labels(
    "output/real_data/scene0000_00_0.ply",
    output_path="visualization.png",
    use_label_colors=True,
)
```

## Expected Output

### Grid Visualization

Each `*_grid.png` file contains 3 panels:
1. **Left**: Original RGB input image
2. **Middle**: Segmentation overlay (colored regions)
3. **Right**: 3D point cloud render (colored by semantic labels)

### Performance

**First run (with model download):**
- Download time: ~5-10 minutes (2GB model)
- Processing: ~30-60 seconds per frame (CPU)
- Processing: ~5-10 seconds per frame (GPU)

**Subsequent runs:**
- No download needed
- Same processing times

## Troubleshooting

### Model Download Issues

If the model download fails or is slow:

```bash
# Use a smaller model
python scripts/demo.py --model facebook/mask2former-swin-tiny-coco-panoptic
```

### Out of Memory

If you get CUDA out of memory errors:

```bash
# Use CPU instead
python scripts/demo.py --device cpu
```

### Visualization Issues

If Open3D rendering fails:

```python
# Check if Open3D is installed correctly
python -c "import open3d as o3d; print(o3d.__version__)"
```

### No Visual Output

The demo runs headless (no GUI windows). All visualizations are saved as PNG files in `output/demo/visualizations/`. Check that directory for the images.

## Testing Without Model Download

To test the project structure without downloading the segmentation model:

```bash
# Create synthetic data only
python scripts/download_sample.py --output data/ --create-synthetic --num-frames 3

# Verify data structure
ls -R data/
```

## Next Steps

After running the demo:

1. **Explore the visualizations** in `output/demo/visualizations/`
2. **Open PLY files** in MeshLab or CloudCompare for interactive viewing
3. **Modify the pipeline** in `scripts/demo.py` for your use case
4. **Use real ScanNet data** for production work

## Example Screenshots

Look for these files:
```
output/demo/visualizations/demo_scene_00_0_grid.png
output/demo/visualizations/demo_scene_00_1_grid.png
output/demo/visualizations/demo_scene_00_2_grid.png
```

Each grid image shows the complete pipeline: RGB input → Segmentation → 3D reconstruction.
