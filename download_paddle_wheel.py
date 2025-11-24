#!/usr/bin/env python3
"""Helper to download the PaddlePaddle GPU wheel once for caching."""
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download PaddlePaddle GPU wheel (for Modal cache)")
    parser.add_argument("--version", default="3.2.0", help="PaddlePaddle GPU version (default: 3.2.0)")
    parser.add_argument(
        "--cuda-tag",
        default="cu126",
        help="Paddle CUDA tag (matches index URL). Default: cu126",
    )
    parser.add_argument("--output-dir", default="./wheels", help="Directory to store downloaded wheel")
    parser.add_argument(
        "--platform",
        default="manylinux2014_x86_64",
        help="Target platform for the wheel (default: manylinux2014_x86_64)",
    )
    parser.add_argument(
        "--python-version",
        default="310",
        help="Python version tag for the wheel (default: 310 for CPython 3.10)",
    )
    parser.add_argument(
        "--use-cudnn95",
        action="store_true",
        help="Use wheel built with CUDNN 9.5 (from ERNIE Makefile). This fixes CUDNN mismatch issues.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_cudnn95:
        # Use wheel built with CUDNN 9.5 (from ERNIE Makefile)
        # This is the correct wheel that matches the Docker image: cuda12.6-cudnn9.5
        print("Using CUDNN 9.5 wheel (from ERNIE Makefile)")
        wheel_url = "https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-TagBuild-Training-Linux-Gpu-Cuda12.6-Cudnn9.5-Trt10.5-Mkl-Avx-Gcc11-SelfBuiltPypiUse/latest/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl"
        wheel_filename = "paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl"  # Version 0.0.0 indicates latest dev build
        wheel_path = output_dir / wheel_filename
        print(f"  URL: {wheel_url}")
        print(f"  Note: This is a dev build (0.0.0) but includes CUDNN 9.5 support")
    else:
        # PaddlePaddle 3.2.0 is not available via pip index, but the wheel exists on CDN
        # Direct download URL format matches what Modal uses during build
        wheel_filename = f"paddlepaddle_gpu-{args.version}-cp{args.python_version}-cp{args.python_version}-linux_x86_64.whl"
        wheel_url = f"https://paddle-whl.bj.bcebos.com/stable/{args.cuda_tag}/paddlepaddle-gpu/{wheel_filename}"
        wheel_path = output_dir / wheel_filename
    
    print(f"Downloading wheel directly from CDN:")
    print(f"  URL: {wheel_url}")
    print(f"  Destination: {wheel_path}")
    
    # Use urllib to download directly
    import urllib.request
    try:
        print("Downloading... (this may take a few minutes for ~1.9GB file)")
        urllib.request.urlretrieve(wheel_url, wheel_path)
        file_size_mb = wheel_path.stat().st_size / (1024 * 1024)
        print(f"✓ Wheel downloaded successfully!")
        print(f"  File: {wheel_path}")
        print(f"  Size: {file_size_mb:.1f} MB")
        if args.use_cudnn95:
            print(f"\n✓ This wheel is built with CUDNN 9.5 - should fix performance issues!")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"✗ Error: Wheel not found at {wheel_url}")
            if args.use_cudnn95:
                print(f"  CUDNN 9.5 wheel may not be available. Try without --use-cudnn95 flag.")
            else:
                print(f"  This version ({args.version}) may not be available for CUDA {args.cuda_tag}")
                print(f"  Available versions in stable/{args.cuda_tag}/: 3.0.0")
                print(f"\nTrying pip download as fallback (may fail)...")
                index_url = f"https://www.paddlepaddle.org.cn/packages/stable/{args.cuda_tag}/"
                cmd = [
                    "python",
                    "-m",
                    "pip",
                    "download",
                    f"paddlepaddle-gpu=={args.version}",
                    "--index-url",
                    index_url,
                    "--dest",
                    str(output_dir),
                ]
                subprocess.run(cmd, check=True)
        else:
            raise

if __name__ == "__main__":
    main()
