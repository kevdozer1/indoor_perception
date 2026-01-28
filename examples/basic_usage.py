"""Basic usage example for the indoor perception pipeline."""

from indoor_perception.dataset import ScanNetDataset, create_sample_scene
from indoor_perception.pipeline import ScenePerceptionPipeline


def main():
    # Create a small synthetic dataset for testing
    print("Creating synthetic sample data...")
    create_sample_scene(
        output_dir="data",
        scene_id="scene0000_00",
        num_frames=3,
        image_size=(640, 480),
    )

    # Load the dataset
    print("\nLoading dataset...")
    dataset = ScanNetDataset(
        data_root="data",
        scene_ids=["scene0000_00"],
    )

    print(f"Dataset contains {len(dataset)} frames")

    # Initialize the pipeline
    print("\nInitializing pipeline...")
    pipeline = ScenePerceptionPipeline(
        dataset=dataset,
        model_name="facebook/mask2former-swin-large-coco-panoptic",
        device="auto",  # Uses CUDA if available, otherwise CPU
    )

    # Process a single frame
    print("\nProcessing first frame...")
    result = pipeline.process_frame(
        idx=0,
        output_path="output/example_frame.ply",
        save_ply=True,
    )

    print(f"\nResults:")
    print(f"  Number of 3D points: {len(result['points'])}")
    print(f"  Number of segments: {len(result['segment_info'])}")
    print(f"  Output file: {result['output_file']}")

    print("\nSegments found:")
    for seg_id, info in result['segment_info'].items():
        print(f"  - Segment {seg_id}: {info['label_name']} (score: {info['score']:.3f})")

    # Process all frames
    print("\n" + "="*60)
    print("Processing all frames...")
    results = pipeline.process_all(
        output_dir="output",
        save_metadata=True,
    )

    print(f"\nProcessed {len(results)} frames successfully!")
    print("Point cloud files saved to: output/")
    print("\nYou can visualize the PLY files using:")
    print("  - MeshLab: https://www.meshlab.net/")
    print("  - CloudCompare: https://www.cloudcompare.org/")


if __name__ == "__main__":
    main()
