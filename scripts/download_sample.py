"""Download or create sample ScanNet data for testing."""

import argparse
from pathlib import Path

from indoor_perception.dataset.downloader import create_sample_scene, download_sample_scannet


def main():
    parser = argparse.ArgumentParser(
        description="Download or create sample ScanNet data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create synthetic sample data for testing
  python scripts/download_sample.py --output data/ --create-synthetic

  # Download from a URL
  python scripts/download_sample.py --output data/ --url https://example.com/scannet_sample.zip

  # Create multiple synthetic scenes
  python scripts/download_sample.py --output data/ --create-synthetic --num-scenes 3 --num-frames 10
        """,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/",
        help="Output directory for the data (default: data/)",
    )

    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to download ScanNet sample from",
    )

    parser.add_argument(
        "--create-synthetic",
        action="store_true",
        help="Create synthetic sample scenes for testing",
    )

    parser.add_argument(
        "--num-scenes",
        type=int,
        default=1,
        help="Number of synthetic scenes to create (default: 1)",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of frames per synthetic scene (default: 5)",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=("WIDTH", "HEIGHT"),
        help="Image size for synthetic data (default: 640 480)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.create_synthetic:
        print("Creating synthetic sample data...")
        for i in range(args.num_scenes):
            scene_id = f"scene{i:04d}_00"
            create_sample_scene(
                output_dir=str(output_dir),
                scene_id=scene_id,
                num_frames=args.num_frames,
                image_size=tuple(args.image_size),
            )
        print(f"\nCreated {args.num_scenes} synthetic scene(s) in {output_dir}")
        print(f"Total frames: {args.num_scenes * args.num_frames}")

    elif args.url:
        print("Downloading sample data...")
        download_sample_scannet(
            output_dir=str(output_dir),
            url=args.url,
            extract=True,
        )
        print(f"\nDownloaded and extracted data to {output_dir}")

    else:
        # Print instructions
        download_sample_scannet(output_dir=str(output_dir), url=None)


if __name__ == "__main__":
    main()
