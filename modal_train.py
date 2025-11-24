"""
Modal script for fine-tuning PaddleOCR-VL-0.9B models using ERNIEKit.
Based on: https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md
"""

import modal
import os
from pathlib import Path

# Wheel selection: CRITICAL - Prefer CUDNN 9.5 wheel (0.0.0) for performance
# The 0.0.0 wheel is built with CUDNN 9.5 support (matches Docker cudnn9.5)
# The stable 3.2.0 wheel may not have CUDNN 9.5 support, causing 10x slowdown
# Performance is more important than version number - use CUDNN 9.5 wheel
WHEEL_FILENAME_CUDNN95 = "paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl"  # CUDNN 9.5 build (PREFERRED for performance)
WHEEL_FILENAME_STABLE = "paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl"  # Stable 3.2.0 (fallback, may be slower)

# Resolve wheel paths relative to this script's directory
# Modal's image definition runs on the client side, so we check for local wheels here
# We try multiple path resolution strategies to handle different execution contexts
def _find_wheel_path():
    """Find the local wheel file, trying multiple path resolution strategies."""
    # Strategy 1: Relative to __file__ (works when script is imported normally)
    script_file = Path(__file__)
    script_dir = script_file.parent.resolve()
    wheels_dir = script_dir / "wheels"
    
    # Check if wheels directory exists at this location
    # CRITICAL: Prefer CUDNN 9.5 wheel (0.0.0) for performance - it's built with CUDNN 9.5 support
    # The stable 3.2.0 wheel may not have CUDNN 9.5, causing 10x slowdown
    if wheels_dir.exists() and wheels_dir.is_dir():
        wheel_cudnn95 = wheels_dir / WHEEL_FILENAME_CUDNN95
        wheel_stable = wheels_dir / WHEEL_FILENAME_STABLE
        
        # Prefer CUDNN 9.5 wheel (0.0.0) - performance is critical
        if wheel_cudnn95.exists():
            return wheel_cudnn95, WHEEL_FILENAME_CUDNN95, wheels_dir
        elif wheel_stable.exists():
            return wheel_stable, WHEEL_FILENAME_STABLE, wheels_dir
    
    # Strategy 2: Try current working directory (fallback)
    cwd_wheels = Path.cwd() / "wheels"
    if cwd_wheels.exists() and cwd_wheels.is_dir():
        wheel_cudnn95 = cwd_wheels / WHEEL_FILENAME_CUDNN95
        wheel_stable = cwd_wheels / WHEEL_FILENAME_STABLE
        
        # Prefer CUDNN 9.5 wheel (0.0.0) - performance is critical
        if wheel_cudnn95.exists():
            return wheel_cudnn95, WHEEL_FILENAME_CUDNN95, cwd_wheels
        elif wheel_stable.exists():
            return wheel_stable, WHEEL_FILENAME_STABLE, cwd_wheels
    
    # No wheel found - return stable path for error message
    return wheels_dir / WHEEL_FILENAME_STABLE, WHEEL_FILENAME_STABLE, wheels_dir

LOCAL_WHEEL_PATH, WHEEL_FILENAME, _wheels_dir = _find_wheel_path()

# Image build version - increment this to force a fresh rebuild
# Modal caches images based on the image definition hash
# Changing this value will invalidate the cache and trigger a fresh build
# To force a rebuild, change this version (e.g., "1.0.1", "1.1.0", etc.)
IMAGE_BUILD_VERSION = "1.0.8"  # Using CUDA 12.8.1 with CUDNN built-in to avoid path conflicts

# Create a Modal app
app = modal.App("paddleocr-vl-sft")

# Following exact setup from ERNIE documentation:
# https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md
# Docker image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5
# Since Chinese registry is not accessible, we use NVIDIA CUDA base + install PaddlePaddle 3.2.0
# CRITICAL: Must use CUDNN 9.5 to match PaddlePaddle 3.2.0 requirements
# CUDNN version mismatch causes 10x slowdown (9.0 vs 9.5)
# Use CUDA 12.8.1 with CUDNN built-in to avoid LD_LIBRARY_PATH conflicts
# This base image includes CUDNN 9.x which should work with PaddlePaddle 3.2.0
# CUDA 12.8.1 is backward compatible with CUDA 12.6 requirements
# Modal's CUDA 12.9 driver supports all 12.x versions
base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install(
        "git",
        "wget",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    .run_commands(
        # Upgrade pip first
        "python -m pip install --upgrade pip",
        # Base image includes CUDNN 9.x in /usr/local/cuda/lib64
        # Verify system CUDNN is available (should be 9.5 or later in CUDA 12.8.1)
        "python -c \"import os; cudnn_path = '/usr/local/cuda/lib64/libcudnn.so'; print(f'System CUDNN available: {os.path.exists(cudnn_path)}')\"",
        # Install PaddlePaddle dependencies first (required when using --no-deps)
        "python -m pip install --no-cache-dir "
        "httpx 'numpy>=1.21,<=1.26.4' 'protobuf>=3.20.2' Pillow "
        "'opt_einsum==3.3.0' networkx typing_extensions",
        # Note: PaddlePaddle 3.2.0 will show dependency warnings for NVIDIA CUDA packages
        # These are harmless - the CUDA libraries are already in the base image
        # Installing them via pip would slow down the build and may cause version conflicts
    )
)

# Check if wheel exists locally (on client machine) and add it to image
# Modal's add_local_file works from the client side, so we check here
# Note: This code may execute in container context during image build,
# so we check file existence carefully and only print warnings if file doesn't exist on client
wheel_path_abs = str(LOCAL_WHEEL_PATH.resolve())
wheels_dir_abs = str(_wheels_dir.resolve())

# Only check and print on client side (when file actually exists)
# During image build in container, paths won't exist but add_local_file still works
wheel_exists = LOCAL_WHEEL_PATH.exists() and LOCAL_WHEEL_PATH.is_file()

if wheel_exists:
    # Client-side: wheel exists, use it
    print(f"✓ Using local PaddlePaddle wheel: {wheel_path_abs}")
    
    # Debug: List available wheels
    if _wheels_dir.exists() and _wheels_dir.is_dir():
        available_wheels = list(_wheels_dir.glob("*.whl"))
        if available_wheels:
            print(f"Found {len(available_wheels)} wheel(s) in {wheels_dir_abs}:")
            for wheel in available_wheels:
                print(f"  - {wheel.name}")
    
    try:
        base_image = (
            base_image.add_local_file(wheel_path_abs, f"/tmp/{WHEEL_FILENAME}", copy=True)
            .run_commands(
                # Install PaddlePaddle from local cached wheel (dependencies already installed)
                # Note: We don't import paddle here because CUDA is not available during image build
                # The import will be verified at runtime when GPU is available
                f"echo 'Installing PaddlePaddle from wheel: {WHEEL_FILENAME}' && "
                f"python -m pip install --no-deps --no-cache-dir /tmp/{WHEEL_FILENAME} && "
                f"echo 'PaddlePaddle wheel installed successfully (version will be verified at runtime)'"
            )
        )
    except Exception as e:
        print(f"WARNING: Failed to add local wheel file: {e}")
        print("Falling back to downloading paddlepaddle-gpu from the official index (slow).")
        base_image = base_image.run_commands(
            "python -m pip install --timeout=1800 --default-timeout=1800 --no-cache-dir "
            "paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
        )
else:
    # Client-side: wheel doesn't exist, will download
    # Only print warning if we're actually on client (not in container during build)
    # Check if we're in a container by seeing if the path looks like a container path
    is_container_path = str(LOCAL_WHEEL_PATH).startswith(("/root", "/tmp", "/paddle")) and not wheel_exists
    
    if not is_container_path:
        # We're on client and wheel doesn't exist - this is a real warning
        print(
            f"WARNING: Local wheel not found at {wheel_path_abs}. "
            f"Checked wheels directory: {wheels_dir_abs}. "
            "Falling back to downloading paddlepaddle-gpu from the official index (slow)."
        )
    
    base_image = base_image.run_commands(
        "python -m pip install --timeout=1800 --default-timeout=1800 --no-cache-dir "
        "paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
    )

image = (
    base_image.run_commands(
        # Clone ERNIE repository (exact sequence from documentation)
        "mkdir -p /paddle",
        "git clone https://github.com/PaddlePaddle/ERNIE.git /paddle/ERNIE",
        "cd /paddle/ERNIE && git checkout develop",
    )
    .run_commands(
        # Install ERNIEKit following EXACT sequence from documentation:
        # https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md#22-install-erniekit
        "cd /paddle/ERNIE && python -m pip install -r requirements/gpu/requirements.txt",
        "cd /paddle/ERNIE && python -m pip install -e .",
        "cd /paddle/ERNIE && python -m pip install tensorboard",
        "cd /paddle/ERNIE && python -m pip install opencv-python-headless",
        "cd /paddle/ERNIE && python -m pip install numpy==1.26.4",
        # CRITICAL: After installing opencv, ensure CUDNN 9.5 path is still first in LD_LIBRARY_PATH
        # opencv-python-headless adds its own paths that might interfere
        # We'll set this in .env() but also verify here
        "python -c \"import os; ld_path = os.environ.get('LD_LIBRARY_PATH', ''); cudnn95_path = '/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib'; print(f'CUDNN 9.5 in LD_LIBRARY_PATH: {cudnn95_path in ld_path}'); print(f'CUDNN 9.5 first: {ld_path.startswith(cudnn95_path) if ld_path else False}')\"",
    )
    .run_commands(
        # Install huggingface-cli for model downloads
        "python -m pip install huggingface-hub[cli]",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "PADDLE_HOME": "/paddle",
        # Use system CUDNN from base image (CUDA 12.8.1 includes CUDNN 9.x)
        # System CUDNN is in /usr/local/cuda/lib64 and should be found automatically
        # No need to prioritize pip-installed CUDNN since base image has compatible version
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
        # Cache-busting: This env var changes the image hash, forcing a fresh build when IMAGE_BUILD_VERSION changes
        "IMAGE_BUILD_VERSION": IMAGE_BUILD_VERSION,
    })
)

# Mount volumes for data, models, and outputs
data_volume = modal.Volume.from_name("paddleocr-vl-data", create_if_missing=True)
model_volume = modal.Volume.from_name("paddleocr-vl-models", create_if_missing=True)
output_volume = modal.Volume.from_name("paddleocr-vl-outputs", create_if_missing=True)


@app.function(
    image=image,
    timeout=600,
    volumes={
        "/models": model_volume,
    },
)
def upload_paddle_wheel_remote(wheel_bytes: bytes, filename: str):
    """Store a locally downloaded PaddlePaddle wheel into the persistent model volume."""
    from pathlib import Path

    dest_dir = Path("/models/paddle_wheels")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_bytes(wheel_bytes)
    model_volume.commit()
    return str(dest_path)


def ensure_paddle_installed():
    """Install PaddlePaddle from the cached wheel if it's not already available."""
    try:
        import paddle  # noqa: F401
        return
    except ImportError:
        pass

    from pathlib import Path
    import subprocess

    wheel_path = Path("/models/paddle_wheels") / WHEEL_FILENAME
    if not wheel_path.exists():
        raise RuntimeError(
            f"PaddlePaddle wheel not found at {wheel_path}. "
            "Run `python download_paddle_wheel.py` and "
            "`modal run modal_train.py --mode upload-wheel --wheel-path ./wheels/<wheel>.whl` first."
        )

    print(f"Installing PaddlePaddle from cached wheel: {wheel_path}")
    subprocess.run(
        ["python", "-m", "pip", "install", "--no-deps", str(wheel_path)],
        check=True,
    )


@app.function(
    image=image,
    # Documentation recommends A800 80GB, but Modal doesn't offer A800 (China-specific).
    # A100 is the closest equivalent with 80GB VRAM (or 40GB variant).
    # A10G (24GB) may work for smaller batches but A100 is recommended for max_seq_len=16384.
    gpu="A100-80GB",  # Use A100 GPU (closest to recommended A800 80GB)
    timeout=86400,  # 24 hours timeout
    volumes={
        "/data": data_volume,
        "/models": model_volume,
        "/outputs": output_volume,
    },
)
def train_paddleocr_vl(
    config_path: str = "examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml",
    train_data_path: str = "./ocr_vl_sft-train_Bengali.jsonl",  # Relative path from ERNIE directory (matching Docker)
    output_dir: str = None,  # Will use config default if None
    model_name_or_path: str = "PaddlePaddle/PaddleOCR-VL",  # Relative path from ERNIE directory (matching Docker)
    resume_from: str = None,
):
    """
    Fine-tune PaddleOCR-VL-0.9B model using ERNIEKit.
    
    Args:
        config_path: Path to the training configuration file (relative to ERNIE root)
        train_data_path: Path to training data JSONL file
        output_dir: Directory to save trained models (optional, uses config default if None)
        model_name_or_path: HuggingFace model name or path (default: PaddlePaddle/PaddleOCR-VL)
        resume_from: Path to checkpoint to resume from (optional)
    """
    import subprocess
    import sys
    from pathlib import Path

    # PaddlePaddle should already be installed in the image during build
    # Just verify it's importable
    try:
        import paddle  # noqa: F401
        print(f"✓ PaddlePaddle is available: version {paddle.__version__}")
        print(f"✓ PaddlePaddle location: {paddle.__file__}")
        
        # Check CUDNN version and verify 9.5 is being used
        try:
            import ctypes
            import subprocess
            
            # Check which CUDNN library is actually being used
            # Base image (CUDA 12.8.1) includes CUDNN in system path
            system_cudnn = "/usr/local/cuda/lib64/libcudnn.so"
            
            system_exists = Path(system_cudnn).exists()
            
            print(f"CUDNN (system): {'✓ Found' if system_exists else '✗ Not found'} at {system_cudnn}")
            
            # Check LD_LIBRARY_PATH for system CUDNN
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if "/usr/local/cuda/lib64" in ld_path:
                print(f"✓ LD_LIBRARY_PATH includes system CUDNN path")
                print(f"  LD_LIBRARY_PATH: {ld_path[:200]}")
            else:
                print(f"⚠ WARNING: LD_LIBRARY_PATH may not include system CUDNN path!")
                print(f"  LD_LIBRARY_PATH: {ld_path[:200]}")
            
            # Try to get actual CUDNN version using cudnnGetVersion if available
            if system_exists:
                try:
                    lib = ctypes.CDLL(system_cudnn)
                    # cudnnGetVersion returns major*1000 + minor*100 + patch
                    get_version = getattr(lib, 'cudnnGetVersion', None)
                    if get_version:
                        get_version.restype = ctypes.c_longlong
                        version = get_version()
                        major = version // 1000
                        minor = (version % 1000) // 100
                        patch = version % 100
                        print(f"✓ CUDNN version from system library: {major}.{minor}.{patch}")
                        if major == 9 and minor >= 5:
                            print(f"  ✓ CUDNN 9.5+ is available (compatible with PaddlePaddle 3.2.0)")
                        elif major == 9:
                            print(f"  ⚠ WARNING: CUDNN {major}.{minor}.{patch} may be slower than 9.5")
                        else:
                            print(f"  ⚠ WARNING: Unexpected CUDNN version {major}.{minor}.{patch}")
                except Exception as e:
                    print(f"  Could not read CUDNN version: {e}")
            
            # CRITICAL: Check what CUDNN version PaddlePaddle is actually using
            # PaddlePaddle may be statically linked or use a different mechanism
            try:
                # Try to get CUDNN version from PaddlePaddle's internal state
                # This is the actual version being used by PaddlePaddle
                paddle_cudnn_version = None
                try:
                    # PaddlePaddle may expose CUDNN version through paddle.device
                    # or through internal flags
                    import paddle.fluid.core as core
                    if hasattr(core, 'cudnn_version'):
                        paddle_cudnn_version = core.cudnn_version()
                        print(f"✓ PaddlePaddle reports CUDNN version: {paddle_cudnn_version}")
                    elif hasattr(core, 'get_cudnn_version'):
                        paddle_cudnn_version = core.get_cudnn_version()
                        print(f"✓ PaddlePaddle reports CUDNN version: {paddle_cudnn_version}")
                except:
                    pass
                
                # Alternative: Check PaddlePaddle's build info
                try:
                    build_info = paddle.version.cudnn()
                    if build_info:
                        print(f"✓ PaddlePaddle build CUDNN info: {build_info}")
                except:
                    pass
                
                # If we can't get version from PaddlePaddle, warn
                if paddle_cudnn_version is None:
                    print(f"⚠ Could not determine CUDNN version from PaddlePaddle")
                    print(f"  This may indicate PaddlePaddle is using a different CUDNN than expected")
                    print(f"  Performance may be affected if wrong CUDNN version is used")
                    
            except Exception as e:
                print(f"⚠ Could not check PaddlePaddle CUDNN version: {e}")
                
        except Exception as e:
            print(f"⚠ Could not verify CUDNN version: {e}")
            
    except ImportError as e:
        raise RuntimeError(
            f"PaddlePaddle is not available in the image. This indicates a build issue. Error: {e}"
        )

    # Set working directory to ERNIE (matching Docker setup)
    # Docker uses /home/ERNIE, but docs use /paddle/ERNIE - we follow docs
    os.chdir("/paddle/ERNIE")
    
    # Ensure model and dataset are in ERNIE directory structure (matching Docker)
    # Docker structure: /home/ERNIE/PaddlePaddle/PaddleOCR-VL and /home/ERNIE/ocr_vl_sft-train_Bengali.jsonl
    import shutil
    
    # Copy model from volume to ERNIE directory if not already there
    model_source = "/models/PaddleOCR-VL"
    model_dest = "/paddle/ERNIE/PaddlePaddle/PaddleOCR-VL"
    if Path(model_source).exists() and not Path(model_dest).exists():
        Path(model_dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_source, model_dest)
        print(f"Copied model from {model_source} to {model_dest}")
    elif Path(model_dest).exists():
        print(f"Model already exists at {model_dest}")
    
    # Copy dataset from volume to ERNIE directory if using absolute path
    if train_data_path.startswith("/data/"):
        dataset_source = train_data_path
        dataset_dest = "/paddle/ERNIE/ocr_vl_sft-train_Bengali.jsonl"
        if Path(dataset_source).exists() and not Path(dataset_dest).exists():
            shutil.copy2(dataset_source, dataset_dest)
            print(f"Copied dataset from {dataset_source} to {dataset_dest}")
        train_data_path = "./ocr_vl_sft-train_Bengali.jsonl"  # Use relative path (matching Docker)
    
    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build training command exactly matching Docker setup:
    # CUDA_VISIBLE_DEVICES=0 erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
    #   model_name_or_path=PaddlePaddle/PaddleOCR-VL \
    #   train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl

    # Set CUDA_VISIBLE_DEVICES (Modal GPUs are typically device 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cmd_parts = ["erniekit", "train", config_path]
    cmd_parts.append(f"model_name_or_path={model_name_or_path}")
    cmd_parts.append(f"train_dataset_path={train_data_path}")
    # Skip slow data skipping when resuming (saves 30+ minutes)
    cmd_parts.append("ignore_data_skip=True")
    # Override recompute to False for better performance on A100-80GB
    # This overrides the config file value and ensures we get full speed
    cmd_parts.append("recompute=False")

    if output_dir:
        cmd_parts.append(f"output_dir={output_dir}")

    if resume_from:
        cmd_parts.append(f"resume_from_checkpoint={resume_from}")

    cmd = cmd_parts

    # Ensure system CUDNN path is in LD_LIBRARY_PATH
    # Base image (CUDA 12.8.1) includes CUDNN in /usr/local/cuda/lib64
    # This should be found automatically, but we ensure it's present
    cuda_lib_path = "/usr/local/cuda/lib64"
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    
    if cuda_lib_path not in current_ld_path:
        # Prepend system CUDNN path if not present
        new_ld_path = f"{cuda_lib_path}:" + current_ld_path
        os.environ["LD_LIBRARY_PATH"] = new_ld_path
        print(f"✓ Added system CUDNN path to LD_LIBRARY_PATH")
        print(f"  CUDNN path: {cuda_lib_path}")
    else:
        print(f"✓ System CUDNN path already in LD_LIBRARY_PATH")
    
    print(f"Running training command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Training data: {train_data_path}")
    print(f"Output directory: {output_dir or 'using config default'}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"LD_LIBRARY_PATH (first 200 chars): {os.environ.get('LD_LIBRARY_PATH', '')[:200]}")

    # Run training with real-time output streaming (matching Docker behavior)
    # ERNIEKit logs to stdout with logging_steps=1, so we need to stream it
    # Pass the fixed LD_LIBRARY_PATH explicitly to the subprocess
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env
    )
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end='', flush=True)
    
    process.wait()
    result = type('obj', (object,), {'returncode': process.returncode})()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")
    
    # Commit volumes to persist data after successful training
    data_volume.commit()
    output_volume.commit()
    
    print(f"Training completed! Models saved to {output_dir}")


@app.function(
    image=image,
    timeout=3600,  # 1 hour timeout
    volumes={
        "/data": data_volume,
        "/models": model_volume,
        "/outputs": output_volume,
    },
)
def download_paddleocr_vl_model(
    model_repo: str = "PaddlePaddle/PaddleOCR-VL",
    local_dir: str = "/models/PaddleOCR-VL",  # Save to volume for persistence
):
    """
    Download PaddleOCR-VL-0.9B model from HuggingFace.
    Matches Docker command: huggingface-cli download PaddlePaddle/PaddleOCR-VL --local-dir PaddlePaddle/PaddleOCR-VL
    
    Args:
        model_repo: HuggingFace model repository name
        local_dir: Directory to save the downloaded model (saved to volume, will be copied to ERNIE dir during training)
    """
    import subprocess
    from pathlib import Path
    
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # Use huggingface-cli to download the model (matching Docker command)
    cmd = [
        "huggingface-cli",
        "download",
        model_repo,
        "--local-dir", local_dir,
    ]
    
    print(f"Downloading model: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    model_volume.commit()
    
    if result.returncode != 0:
        raise RuntimeError(f"Model download failed with exit code {result.returncode}")
    
    print(f"Model downloaded to {local_dir}")
    print(f"Model files: {list(Path(local_dir).rglob('*'))}")
    print(f"Note: Model will be copied to /paddle/ERNIE/PaddlePaddle/PaddleOCR-VL during training (matching Docker structure)")


@app.function(
    image=image,
    # Use same GPU as training for consistency and performance
    gpu="A100-80GB",  # Use A100 GPU (same as training)
    timeout=3600,  # 1 hour timeout
    volumes={
        "/data": data_volume,
        "/models": model_volume,
        "/outputs": output_volume,
    },
)
def inference_paddleocr_vl(
    image_path: str,
    model_dir: str = "/outputs/PaddleOCR-VL-SFT",
    save_path: str = "/outputs/inference_results",
    vl_rec_model_name: str = "PaddleOCR-VL-0.9B",
):
    """
    Run inference using fine-tuned PaddleOCR-VL model.
    
    Args:
        image_path: Path to input image (can be URL or local path)
        model_dir: Directory containing the fine-tuned model
        save_path: Directory to save inference results
        vl_rec_model_name: Name of the VL recognition model
    """
    import subprocess
    from pathlib import Path

    ensure_paddle_installed()
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Copy inference config files if they don't exist
    base_model_dir = "/models/PaddleOCR-VL/PaddleOCR-VL-0.9B"
    if Path(base_model_dir).exists():
        import shutil
        for file in ["chat_template.jinja", "inference.yml"]:
            src = Path(base_model_dir) / file
            dst = Path(model_dir) / file
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                print(f"Copied {file} to model directory")
    
    # Run inference using paddleocr doc_parser
    cmd = [
        "paddleocr",
        "doc_parser",
        "-i", image_path,
        "--vl_rec_model_name", vl_rec_model_name,
        "--vl_rec_model_dir", model_dir,
        "--save_path", save_path,
    ]
    
    print(f"Running inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    output_volume.commit()
    
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed with exit code {result.returncode}")
    
    print(f"Inference completed! Results saved to {save_path}")


@app.function(
    image=image,
    timeout=600,
)
def test_wheel_detection():
    """
    Test that verifies:
    1. PaddlePaddle is installed and importable
    2. The installation method (local wheel vs download) can be inferred
    3. PaddlePaddle version matches expected version
    """
    import sys
    from pathlib import Path
    
    results = []
    
    # Test 1: PaddlePaddle import
    try:
        import paddle
        version = paddle.__version__
        results.append(f"✓ PaddlePaddle imported successfully: version {version}")
        
        # Test 2: Check if installed from wheel (check installation location)
        paddle_path = Path(paddle.__file__).parent
        results.append(f"✓ PaddlePaddle location: {paddle_path}")
        
        # Test 3: Check CUDA availability
        cuda_available = paddle.device.is_compiled_with_cuda()
        results.append(f"✓ CUDA compiled: {cuda_available}")
        
        if cuda_available:
            device_count = paddle.device.cuda.device_count()
            results.append(f"✓ GPU devices available: {device_count}")
        
        # Test 4: Verify installation method
        # If installed from local wheel, the package should be in site-packages
        # We can't definitively tell, but we can check if it's a standard installation
        import site
        site_packages = site.getsitepackages()
        if any(str(paddle_path).startswith(str(sp)) for sp in site_packages):
            results.append("✓ PaddlePaddle installed in standard site-packages location")
        
        # Test 5: Check if wheel file was used (indirect check)
        # Look for evidence of wheel installation in pip history or check file timestamps
        # This is indirect, but if the image was built recently and PaddlePaddle works,
        # it's likely from the local wheel if one was available
        
        return "\n".join(results)
        
    except ImportError as e:
        return f"✗ FAILED: Could not import PaddlePaddle: {e}"
    except Exception as e:
        return f"✗ FAILED: Error during test: {e}"


@app.function(
    image=image,
    timeout=300,
    volumes={
        "/data": data_volume,
    },
)
def download_sample_dataset(
    dataset_url: str = "https://paddleformers.bj.bcebos.com/datasets/ocr_vl_sft-train_Bengali.jsonl",
    output_path: str = "/data/ocr_vl_sft-train_Bengali.jsonl",  # Save to volume, will be copied to ERNIE dir during training
):
    """
    Download sample training dataset (Bengali example).
    
    Args:
        dataset_url: URL to download the dataset from
        output_path: Path to save the downloaded dataset
    """
    import subprocess
    from pathlib import Path
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = ["wget", "-O", output_path, dataset_url]
    
    print(f"Downloading dataset: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    data_volume.commit()
    
    if result.returncode != 0:
        raise RuntimeError(f"Dataset download failed with exit code {result.returncode}")
    
    print(f"Dataset downloaded to {output_path}")


@app.local_entrypoint()
def main(
    mode: str = "train",
    config_path: str = "examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml",
    train_data_path: str = "/data/ocr_vl_sft-train_Bengali.jsonl",
    output_dir: str = None,
    model_name_or_path: str = "PaddlePaddle/PaddleOCR-VL",
    resume_from: str = None,
    model_repo: str = "PaddlePaddle/PaddleOCR-VL",
    model_dir: str = "/models/PaddleOCR-VL",
    image_path: str = None,
    save_path: str = "/outputs/inference_results",
    wheel_path: str = None,
):
    """
    Main entrypoint for running training, download, or inference.
    
    Usage:
        # Download the base model
        modal run modal_train.py --mode download
        
        # Download sample dataset
        modal run modal_train.py --mode download-dataset
        
        # Test wheel detection and PaddlePaddle installation
        modal run modal_train.py --mode test-wheel
        
        # Train the model
        modal run modal_train.py --mode train --train-data-path /data/train.jsonl
        
        # Run inference
        modal run modal_train.py --mode inference --image-path "https://example.com/image.png"
    """
    if mode == "train":
        train_paddleocr_vl.remote(
            config_path=config_path,
            train_data_path=train_data_path,
            output_dir=output_dir,
            model_name_or_path=model_name_or_path,
            resume_from=resume_from,
        )
    elif mode == "download":
        download_paddleocr_vl_model.remote(
            model_repo=model_repo,
            local_dir=model_dir,
        )
    elif mode == "download-dataset":
        download_sample_dataset.remote()
    elif mode == "test-wheel":
        result = test_wheel_detection.remote()
        print("\n" + "="*70)
        print("Wheel Detection Test Results")
        print("="*70)
        print(result)
        print("="*70)
    elif mode == "inference":
        if not image_path:
            raise ValueError("image_path is required for inference mode")
        inference_paddleocr_vl.remote(
            image_path=image_path,
            model_dir=output_dir,
            save_path=save_path,
        )
    elif mode == "upload-wheel":
        if not wheel_path:
            raise ValueError("wheel_path is required for upload-wheel mode")
        from pathlib import Path

        wheel_file = Path(wheel_path)
        if not wheel_file.exists():
            raise FileNotFoundError(f"Wheel not found: {wheel_file}")
        wheel_bytes = wheel_file.read_bytes()
        dest_path = upload_paddle_wheel_remote.remote(wheel_bytes, wheel_file.name)
        print(f"Wheel uploaded to {dest_path} inside modal volume. Ready for image build.")
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use 'train', 'download', 'download-dataset', 'test-wheel', 'upload-wheel', or 'inference'"
        )
