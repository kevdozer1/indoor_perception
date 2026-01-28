"""Download real RGB-D indoor scene samples from public sources."""

import argparse
from pathlib import Path

from indoor_perception.dataset.real_samples import (
    create_realistic_samples,
    download_open3d_sample,
    download_tum_samples,
)


def main():
    parser = argparse.ArgumentParser(description="Download real indoor RGB-D samples")
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["tum", "open3d", "synthetic"],
        default="tum",
        help="Data source: tum (download real), open3d (sample), or synthetic",
    )
    parser.add_argument(
        "--open3d-sample",
        type=str,
        default="tum",
        help="Open3D sample to download (default: tum)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL verification for downloads (use only if certificates fail)",
    )

    args = parser.parse_args()

    if args.source == "tum":
        download_tum_samples(args.output, insecure=args.insecure)
    elif args.source == "open3d":
        download_open3d_sample(Path(args.output), sample=args.open3d_sample)
    else:
        create_realistic_samples(Path(args.output))


if __name__ == "__main__":
    main()
