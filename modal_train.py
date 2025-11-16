"""
Modal script for fine-tuning PaddleOCR-VL-0.9B models using ERNIEKit.
Based on: https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md
"""

import modal
import os
from pathlib import Path

WHEEL_FILENAME = "paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl"
LOCAL_WHEEL_PATH = Path(__file__).parent / "wheels" / WHEEL_FILENAME

# Create a Modal app
app = modal.App("paddleocr-vl-sft")

# Following exact setup from ERNIE documentation:
# https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md
# Docker image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5
# Since Chinese registry is not accessible, we use NVIDIA CUDA base + install PaddlePaddle 3.2.0
base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04",
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
        # Install PaddlePaddle dependencies first (required when using --no-deps)
        "python -m pip install --no-cache-dir "
        "httpx 'numpy>=1.21,<=1.26.4' 'protobuf>=3.20.2' Pillow "
        "'opt_einsum==3.3.0' networkx typing_extensions",
    )
)

# Check if wheel exists locally (on client machine) and add it to image
# Use absolute path to ensure Modal can find it
wheel_path_abs = str(LOCAL_WHEEL_PATH.resolve())
if LOCAL_WHEEL_PATH.exists():
    print(f"Using local PaddlePaddle wheel: {wheel_path_abs}")
    base_image = (
        base_image.add_local_file(wheel_path_abs, f"/tmp/{WHEEL_FILENAME}", copy=True)
        .run_commands(
            # Install PaddlePaddle from local cached wheel (dependencies already installed)
            f"python -m pip install --no-deps --no-cache-dir /tmp/{WHEEL_FILENAME}"
        )
    )
else:
    print(
        f"WARNING: Local wheel not found at {wheel_path_abs}. "
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
    )
    .run_commands(
        # Install huggingface-cli for model downloads
        "python -m pip install huggingface-hub[cli]",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "PADDLE_HOME": "/paddle",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
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
    gpu="A100-40GB",  # Use A100 GPU (closest to recommended A800 80GB)
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
        print("✓ PaddlePaddle is available")
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

    if output_dir:
        cmd_parts.append(f"output_dir={output_dir}")

    if resume_from:
        cmd_parts.append(f"resume_from_checkpoint={resume_from}")

    cmd = cmd_parts

    print(f"Running training command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Training data: {train_data_path}")
    print(f"Output directory: {output_dir or 'using config default'}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Run training with real-time output (matching Docker behavior)
    result = subprocess.run(
        cmd,
        check=False
    )

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
    gpu="A100-40GB",  # Use A100 GPU (same as training)
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
            f"Unknown mode: {mode}. Use 'train', 'download', 'download-dataset', 'upload-wheel', or 'inference'"
        )
