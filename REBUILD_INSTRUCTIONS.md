# Rebuild Instructions: CUDA 12.6 + CUDNN 9.5 + A100-80GB

## CUDA Version Compatibility Analysis

### Modal Host Machine
- **CUDA Driver**: 12.9 (from [Modal CUDA docs](https://modal.com/docs/guide/cuda))
- **Supports**: Any CUDA 12.x version (backward compatible)
- **Source**: Modal automatically provides CUDA Driver 12.9 on all GPU machines

### PaddleOCR-VL Requirements (from ERNIE docs)
- **Docker Image**: `cuda12.6-cudnn9.5`
- **PaddlePaddle 3.2.0 requires**: CUDNN 9.5.1.17
- **Source**: [ERNIE PaddleOCR-VL SFT docs](https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md)

### Current Configuration
- **Base Image**: `nvidia/cuda:12.6.0-cudnn9-devel-ubuntu22.04`
- **CUDNN Fix**: Installing `nvidia-cudnn-cu12==9.5.1.17` via pip (matches PaddlePaddle requirement)
- **GPU**: `A100-80GB` (matches A800-80GB memory capacity)

## Changes Made

1. **Updated CUDA version**: `12.3.2` → `12.6.0` (matches Docker)
2. **Added CUDNN 9.5.1.17**: Installing via pip to match PaddlePaddle requirements
3. **Updated GPU**: `A100-40GB` → `A100-80GB` (matches A800-80GB)

## Rebuild Commands

### Step 1: Verify Local Wheel Exists
```bash
ls -lh wheels/paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl
```

If missing, download it:
```bash
python download_paddle_wheel.py --output-dir ./wheels
```

### Step 2: Rebuild Image (with new CUDA/CUDNN)
```bash
# Option 1: Build in detached mode (recommended)
modal run --detach modal_train.py::download_sample_dataset

# Option 2: Build directly (stay connected)
modal run modal_train.py --mode download-dataset
```

This will:
- Pull new CUDA 12.6.0 base image
- Install CUDNN 9.5.1.17
- Install PaddlePaddle 3.2.0 from local wheel
- Install ERNIEKit

### Step 3: Verify CUDNN Version (after build)
```bash
modal run modal_train.py --mode train \
  --train-data-path /data/ocr_vl_sft-train_Bengali.jsonl \
  --output-dir /outputs/paddleocr-vl-sft-test
```

Check logs for CUDNN warning - it should be gone!

### Step 4: Start Training (Detached)
```bash
modal run --detach modal_train.py --mode train \
  --train-data-path /data/ocr_vl_sft-train_Bengali.jsonl \
  --output-dir /outputs/paddleocr-vl-sft
```

## Expected Performance

**Before (A100-40GB + CUDNN 9.0):**
- ~82.7 seconds/step
- ~21.3 hours for 926 steps
- 10.6x slower than expected

**After (A100-80GB + CUDNN 9.5):**
- Expected: ~7.8 seconds/step (matching A800-80GB)
- Expected: ~2 hours for 926 steps
- Should match Docker performance

## Verification Checklist

After rebuild, verify:
- [ ] No CUDNN version mismatch warning in logs
- [ ] Training speed: ~0.13 steps/second (vs 0.012 before)
- [ ] Memory usage: Should have more headroom with 80GB
- [ ] Time per step: ~7-8 seconds (vs 82 seconds before)

## Troubleshooting

If CUDNN warning persists:
1. Check logs for actual CUDNN version detected
2. Verify `nvidia-cudnn-cu12==9.5.1.17` installed correctly
3. May need to set `LD_LIBRARY_PATH` to prioritize pip-installed CUDNN

