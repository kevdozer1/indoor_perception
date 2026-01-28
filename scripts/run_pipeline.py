"""Run the complete indoor scene perception pipeline."""

import argparse
from pathlib import Path

from indoor_perception.dataset import ScanNetDataset
from indoor_perception.pipeline import ScenePerceptionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run indoor scene perception pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all frames in a dataset
  python scripts/run_pipeline.py --data data/ --output output/

  # Process specific scenes
  python scripts/run_pipeline.py --data data/ --output output/ --scenes scene0000_00 scene0001_00

  # Process limited frames for testing
  python scripts/run_pipeline.py --data data/ --output output/ --max-frames 10

  # Use CPU instead of GPU
  python scripts/run_pipeline.py --data data/ --output output/ --device cpu

  # Use a different model
  python scripts/run_pipeline.py --data data/ --output output/ --model facebook/mask2former-swin-base-coco-panoptic
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to ScanNet data directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for PLY files",
    )

    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=None,
        help="Specific scene IDs to process (default: all scenes)",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all)",
    )

    parser.add_argument(
        "--max-frames-per-scene",
        type=int,
        default=None,
        help="Maximum frames per scene in dataset (default: all)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="facebook/mask2former-swin-large-coco-panoptic",
        help="Hugging Face model name (default: mask2former-swin-large-coco-panoptic)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference (default: auto)",
    )

    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Depth scale factor (ScanNet uses mm, so 1000.0 for meters) (default: 1000.0)",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't save segment metadata JSON files",
    )
    parser.add_argument(
        "--save-segmentation-maps",
        action="store_true",
        help="Save segmentation maps as .npy files alongside PLY outputs",
    )

    args = parser.parse_args()

    # Validate paths
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data directory does not exist: {data_path}")
        return

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Indoor Scene Perception Pipeline")
    print("="*60)
    print(f"Data directory: {data_path}")
    print(f"Output directory: {output_path}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print("="*60)

    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = ScanNetDataset(
            data_root=str(data_path),
            scene_ids=args.scenes,
            depth_scale=args.depth_scale,
            max_frames_per_scene=args.max_frames_per_scene,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize pipeline
    print("\nInitializing pipeline...")
    try:
        pipeline = ScenePerceptionPipeline(
            dataset=dataset,
            model_name=args.model,
            device=args.device,
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    # Print statistics
    stats = pipeline.get_statistics()
    print("\nPipeline Statistics:")
    print(f"  Number of frames: {stats['num_frames']}")
    print(f"  Number of classes: {stats['num_classes']}")
    print(f"  Model device: {stats['device']}")

    # Process frames
    print("\nStarting processing...")
    try:
        results = pipeline.process_all(
            output_dir=str(output_path),
            max_frames=args.max_frames,
            save_metadata=not args.no_metadata,
            save_segmentation_maps=args.save_segmentation_maps,
        )

        # Print summary
        print("\nProcessing complete!")
        print(f"Successfully processed {len(results)} frames")
        print(f"Output files saved to: {output_path}")

        # Print sample of segment types found
        if results:
            print("\nExample segments found:")
            sample_result = results[0]
            for seg_id, info in list(sample_result["segment_info"].items())[:5]:
                print(f"  - {info['label_name']} (score: {info['score']:.2f})")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
