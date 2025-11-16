# Modal Setup Learnings: Docker to Modal Conversion Guide

This document captures the key learnings from successfully converting the ERNIE Docker setup to Modal for PaddleOCR-VL fine-tuning.

**Reference:** Based on [ERNIE PaddleOCR-VL SFT Documentation](https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md)

## Core Principles

1. **Follow ERNIE documentation exactly** - The Docker setup is the source of truth
2. **Match directory structure** - Use `/paddle/ERNIE` (not `/home/ERNIE`)
3. **Use exact installation sequence** - Follow ERNIE docs step-by-step
4. **Pre-install dependencies** - Avoid pip dependency resolution loops

---

## What Worked: Image Build

### 1. Base Image Selection

```python
# Use NVIDIA CUDA base (Chinese registry not accessible)
base_image = modal.Image.from_registry(
    "nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04",
    add_python="3.10"
)
```

**Why:** The official PaddlePaddle Docker image (`ccr-2vdh3abv-pub.cnc.bj.baidubce.com`) is not accessible outside China, so we use NVIDIA's public CUDA image.

**Docker equivalent:**
```bash
docker run --gpus all ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5
```

### 2. PaddlePaddle Installation Strategy

**Critical:** Pre-install dependencies BEFORE installing PaddlePaddle to avoid dependency resolution loops.

```python
# Step 1: Pre-install dependencies FIRST
.run_commands(
    "python -m pip install --no-cache-dir "
    "httpx 'numpy>=1.21,<=1.26.4' 'protobuf>=3.20.2' Pillow "
    "'opt_einsum==3.3.0' networkx typing_extensions",
)

# Step 2: Install PaddlePaddle with --no-deps (from local wheel if available)
if LOCAL_WHEEL_PATH.exists():
    base_image.add_local_file(wheel_path, f"/tmp/{WHEEL_FILENAME}", copy=True)
    .run_commands(f"python -m pip install --no-deps --no-cache-dir /tmp/{WHEEL_FILENAME}")
else:
    # Fallback to downloading from index (slow)
    base_image.run_commands(
        "python -m pip install --timeout=1800 --default-timeout=1800 --no-cache-dir "
        "paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
    )
```

**Why this works:**
- Pre-installing dependencies prevents pip from downloading multiple 1.7GB wheels during backtracking
- `--no-deps` flag prevents re-downloading dependencies
- Local wheel caching reduces build time from 20+ minutes to ~10 minutes
- Only **ONE** 1.7GB download instead of 4-5!

**PaddlePaddle Version:** Use stable `3.2.0` (cu126), matching the Docker image version.

### 3. ERNIEKit Installation Sequence (EXACT Match)

```python
.run_commands(
    # Clone ERNIE repository
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
    "cd /paddle/ERNIE && python -m pip install numpy==1.26.4",  # Force version AFTER opencv
)
```

**Why:** This exact sequence from ERNIE docs ensures compatibility. Installing `numpy==1.26.4` **after** `opencv-python-headless` resolves version conflicts.

**Docker equivalent:**
```bash
cd ERNIE
python -m pip install -r requirements/gpu/requirements.txt
python -m pip install -e .
python -m pip install tensorboard
python -m pip install opencv-python-headless
python -m pip install numpy==1.26.4
```

### 4. Environment Variables

```python
.env({
    "HF_HOME": "/root/.cache/huggingface",
    "PADDLE_HOME": "/paddle",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
})
```

**Why:** Matches Docker environment setup for HuggingFace cache and CUDA libraries.

---

## What Worked: Training Setup

### 1. Directory Structure Matching

```python
# Set working directory to ERNIE (matching Docker)
os.chdir("/paddle/ERNIE")

# Copy model to expected location
model_dest = "/paddle/ERNIE/PaddlePaddle/PaddleOCR-VL"
if Path(model_source).exists() and not Path(model_dest).exists():
    shutil.copytree(model_source, model_dest)

# Copy dataset to expected location  
dataset_dest = "/paddle/ERNIE/ocr_vl_sft-train_Bengali.jsonl"
if Path(dataset_source).exists() and not Path(dataset_dest).exists():
    shutil.copy2(dataset_source, dataset_dest)
    train_data_path = "./ocr_vl_sft-train_Bengali.jsonl"  # Use relative path!
```

**Why:** ERNIEKit expects files in specific locations relative to `/paddle/ERNIE`. Must match Docker structure exactly.

**Docker structure:**
- Model: `/home/ERNIE/PaddlePaddle/PaddleOCR-VL` (we use `/paddle/ERNIE/PaddlePaddle/PaddleOCR-VL`)
- Dataset: `/home/ERNIE/ocr_vl_sft-train_Bengali.jsonl` (we use `/paddle/ERNIE/ocr_vl_sft-train_Bengali.jsonl`)

### 2. Training Command (Exact Match)

```python
# Build training command exactly matching Docker setup:
# CUDA_VISIBLE_DEVICES=0 erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
#   model_name_or_path=PaddlePaddle/PaddleOCR-VL \
#   train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cmd = [
    "erniekit", "train", 
    "examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml",
    "model_name_or_path=PaddlePaddle/PaddleOCR-VL",
    "train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl",  # Relative path!
]

if output_dir:
    cmd.append(f"output_dir={output_dir}")

subprocess.run(cmd, check=False)
```

**Why:** 
- Use **relative paths** (`./`) matching Docker setup
- Config file path is relative to `/paddle/ERNIE`
- Command-line arguments override config file values

**Docker equivalent:**
```bash
CUDA_VISIBLE_DEVICES=0 erniekit train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
    model_name_or_path=PaddlePaddle/PaddleOCR-VL \
    train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl
```

### 3. Use `--detach` for Long Training

```bash
modal run --detach modal_train.py --mode train
```

**Why:** Keeps training running even if your local client disconnects. Essential for multi-hour training runs.

---

## Critical Fixes That Worked

### 1. PaddlePaddle Version: Use Stable 3.2.0

**Problem:** Initially tried nightly builds, but they caused issues.

**Solution:** Use stable `3.2.0` (cu126) matching the Docker image:
- Docker: `paddle:3.2.0-gpu-cuda12.6-cudnn9.5`
- Modal: `paddlepaddle-gpu==3.2.0` from stable cu126 index

**Download wheel locally first:**
```bash
python download_paddle_wheel.py --output-dir ./wheels
```

### 2. Dependency Installation Order

**Problem:** Pip was downloading multiple 1.7GB PaddlePaddle wheels during dependency resolution.

**Solution:** Pre-install dependencies, then install PaddlePaddle with `--no-deps`:
1. Pre-install PaddlePaddle dependencies (numpy, opt_einsum, etc.)
2. Install PaddlePaddle with `--no-deps`
3. Install ERNIEKit requirements
4. Force `numpy==1.26.4` last

**Result:** Only ONE 1.7GB download instead of 4-5!

### 3. Missing Dependencies: opt_einsum

**Problem:** `ModuleNotFoundError: No module named 'opt_einsum'` after installing PaddlePaddle.

**Solution:** Install `opt_einsum==3.3.0` BEFORE PaddlePaddle:
```python
"python -m pip install --no-cache-dir 'opt_einsum==3.3.0' ..."
```

**Why:** PaddlePaddle requires it but doesn't declare it as a dependency when using `--no-deps`.

### 4. Numpy Version Conflict

**Problem:** 
- `opencv-python-headless` wants `numpy>=2.0`
- `paddleformers` wants `numpy<=1.26.4`

**Solution:** Install `numpy==1.26.4` **AFTER** `opencv-python-headless`:
```python
"cd /paddle/ERNIE && python -m pip install opencv-python-headless",
"cd /paddle/ERNIE && python -m pip install numpy==1.26.4",  # Force version
```

**Why:** Installing numpy last forces the correct version that satisfies paddleformers.

---

## File Structure

```
vlm-training/
├── modal_train.py              # Main Modal script
├── download_paddle_wheel.py    # Download PaddlePaddle wheel locally
├── upload_data.py              # Helper to upload datasets
├── quick_test.py               # Quick validation tests
├── test_setup.py               # Full test suite
├── wheels/                     # Local PaddlePaddle wheel cache
│   └── paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl
└── ERNIE/                      # ERNIE repository (git submodule or clone)
    └── examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

---

## Complete Workflow

### 1. Download Wheel Once (Saves 10+ minutes per build)

```bash
python download_paddle_wheel.py --output-dir ./wheels
```

This downloads the 1.9GB PaddlePaddle wheel locally. Modal will use it during image build.

### 2. Download Model

```bash
modal run modal_train.py --mode download
```

Saves to `/models/PaddleOCR-VL` in Modal volume.

### 3. Download Dataset

```bash
modal run modal_train.py --mode download-dataset
```

Downloads Bengali sample dataset to `/data/ocr_vl_sft-train_Bengali.jsonl`.

### 4. Start Training (Detached)

```bash
modal run --detach modal_train.py --mode train \
  --train-data-path /data/ocr_vl_sft-train_Bengali.jsonl \
  --output-dir /outputs/paddleocr-vl-sft
```

**Important:** Always use `--detach` for long training runs!

### 5. Resume Training (Auto-detects checkpoint)

```bash
modal run --detach modal_train.py --mode train \
  --train-data-path /data/ocr_vl_sft-train_Bengali.jsonl \
  --output-dir /outputs/paddleocr-vl-sft
```

ERNIEKit automatically detects and resumes from the last checkpoint.

---

## Key Learnings

1. **Match Docker setup exactly** - Don't deviate from ERNIE docs
2. **Pre-install dependencies** - Avoid pip dependency resolution
3. **Use local wheel caching** - Saves 10+ minutes per build
4. **Directory structure matters** - ERNIEKit expects specific paths
5. **Use relative paths in training commands** - Match Docker behavior
6. **Always use `--detach` for training** - Prevents disconnection issues
7. **Force numpy==1.26.4 after opencv** - Resolves version conflicts
8. **Install opt_einsum before PaddlePaddle** - Required but not declared

---

## Common Pitfalls to Avoid

### ❌ Don't Pin PaddlePaddle to Nightly

**Wrong:**
```python
"paddlepaddle-gpu==3.0.0.dev20250114"  # Old nightly
```

**Right:**
```python
"paddlepaddle-gpu==3.2.0"  # Stable, matches Docker
```

### ❌ Don't Skip Dependency Pre-installation

**Wrong:**
```python
# Install PaddlePaddle directly - causes multiple downloads
"python -m pip install paddlepaddle-gpu==3.2.0"
```

**Right:**
```python
# Pre-install deps, then install with --no-deps
"python -m pip install httpx numpy opt_einsum ..."
"python -m pip install --no-deps paddlepaddle-gpu==3.2.0"
```

### ❌ Don't Use Absolute Paths in Training Commands

**Wrong:**
```python
"train_dataset_path=/data/ocr_vl_sft-train_Bengali.jsonl"  # Absolute
```

**Right:**
```python
"train_dataset_path=./ocr_vl_sft-train_Bengali.jsonl"  # Relative
```

### ❌ Don't Forget `--detach`

**Wrong:**
```bash
modal run modal_train.py --mode train  # Stops if you disconnect
```

**Right:**
```bash
modal run --detach modal_train.py --mode train  # Continues running
```

### ❌ Don't Install Numpy Before OpenCV

**Wrong:**
```python
"python -m pip install numpy==1.26.4"
"python -m pip install opencv-python-headless"  # Upgrades numpy to 2.0
```

**Right:**
```python
"python -m pip install opencv-python-headless"
"python -m pip install numpy==1.26.4"  # Force correct version
```

---

## Performance Improvements

### Build Time Optimization

**Before:** 20+ minutes (multiple 1.7GB downloads)
**After:** ~10 minutes (single download, cached)

**Optimizations:**
1. Pre-install dependencies → prevents backtracking
2. Use `--no-deps` → prevents re-downloading deps
3. Local wheel caching → reuses downloaded wheel
4. `--no-cache-dir` → reduces image size

### Training Time

- **Expected:** ~2 hours on A100-40GB (matches Docker on A800-80GB)
- **Hyperparameters:** All match official config exactly
- **Checkpoint resumption:** Automatic

---

## Verification Checklist

Before training, verify:

- [ ] PaddlePaddle wheel downloaded locally (`./wheels/`)
- [ ] Model downloaded to Modal volume (`/models/PaddleOCR-VL`)
- [ ] Dataset downloaded to Modal volume (`/data/`)
- [ ] Image builds successfully (check logs)
- [ ] PaddlePaddle imports correctly (`import paddle`)
- [ ] ERNIEKit CLI works (`erniekit --help`)
- [ ] Config file exists (`examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml`)

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: opt_einsum` | Install `opt_einsum==3.3.0` before PaddlePaddle |
| `numpy version conflict` | Install `numpy==1.26.4` AFTER opencv-python-headless |
| Multiple wheel downloads | Pre-install dependencies, use `--no-deps` |
| Training stops on disconnect | Use `--detach` flag |
| Model not found | Copy from `/models/` to `/paddle/ERNIE/PaddlePaddle/` |
| Dataset not found | Copy from `/data/` to `/paddle/ERNIE/` |

---

## References

- [ERNIE PaddleOCR-VL SFT Documentation](https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md)
- [ERNIEKit Installation Guide](https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/erniekit.md)
- [Modal Documentation](https://modal.com/docs)

---

**Last Updated:** 2025-11-16  
**Status:** ✅ Working - Successfully training PaddleOCR-VL on Modal

