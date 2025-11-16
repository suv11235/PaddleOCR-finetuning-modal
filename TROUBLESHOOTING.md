# Troubleshooting Guide

## ImportError: cannot import name 'fused_rms_norm_ext'

### Error Message
```
ImportError: cannot import name 'fused_rms_norm_ext' from 'paddle.incubate.nn.functional'
```

### Cause
This error occurs when using an incompatible version of PaddlePaddle with paddleformers. The `fused_rms_norm_ext` function is required by paddleformers but may not be available in all PaddlePaddle versions.

### Solution
Use the latest nightly build of PaddlePaddle instead of stable releases:

```python
# In modal_train.py
.run_commands(
    "python -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/",
)
```

**Why this works:**
- Nightly builds include the latest features and bug fixes
- The `fused_rms_norm_ext` function is available in recent nightly builds
- ERNIE's paddleformers dependency requires these newer features

## ModuleNotFoundError: No module named 'paddle.distributed.flex_checkpoint'

### Error Message
```
ModuleNotFoundError: No module named 'paddle.distributed.flex_checkpoint'
Traceback (most recent call last):
  File ".../paddleformers/peft/lora/lora_layers.py", line 26, in <module>
    from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
ModuleNotFoundError: No module named 'paddle.distributed.flex_checkpoint'
```

### Cause
This error occurs when using a PaddlePaddle version that doesn't include the `paddle.distributed.flex_checkpoint` module. This module is required by paddleformers for LoRA (Low-Rank Adaptation) training features, but is only available in recent nightly builds of PaddlePaddle.

### Solution
Use the **latest** nightly build of PaddlePaddle (without version pinning) to ensure you get the most recent features:

```python
# In modal_train.py - DO NOT pin to a specific version
.run_commands(
    "python -m pip install --timeout=600 paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/",
)
```

**Why this works:**
- The `flex_checkpoint` module was added in recent PaddlePaddle nightly builds
- Pinning to an older nightly build (e.g., `==3.0.0.dev20250114`) may not include this module
- Using the latest nightly ensures you get all required features for paddleformers

**Important:** Always use the latest nightly build without version pinning for paddleformers compatibility.

## Version Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 12.3.2 | From nvidia/cuda Docker image |
| cuDNN | 9 | Included in CUDA image |
| Python | 3.10 | Required by Modal |
| PaddlePaddle | nightly (cu123) | Latest from nightly repo |
| paddleformers | 0.3.2 | Installed via ERNIE requirements |
| numpy | 1.26.4 | Required by paddleformers |
| transformers | >=4.55.1 | Required by paddleformers |

## Common Issues

### 1. Chinese Registry Not Accessible

**Error:**
```
dial tcp: lookup ccr-2vdh3abv-pub.cnc.bj.bcebce.com: no such host
```

**Solution:**
Use public NVIDIA CUDA image + PaddlePaddle from PyPI (already implemented in current scripts).

### 2. NumPy Version Conflicts

**Error:**
```
opencv-python-headless requires numpy>=2.0
paddleformers requires numpy<=1.26.4
```

**Solution:**
Install numpy==1.26.4 last (after opencv) to ensure correct version (already implemented).

### 3. Modal API Deprecation

**Error:**
```
InvalidError: Duplicate local entrypoint name
```

**Solution:**
- Use `modal.App` instead of `modal.Stub`
- Test scripts import from main app to avoid conflicts

### 4. GPU Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Use A100-80GB instead of A100-40GB
- Reduce `max_seq_len` in config from 16384 to 8192
- Reduce batch size or gradient accumulation steps

### 5. PaddlePaddle Download Timeout

**Error:**
```
Image build terminated due to external shut-down
WARNING: Connection timed out while downloading
```

**Cause:**
The PaddlePaddle nightly wheel is 1.7GB and can timeout during download.

**Solutions:**
```bash
# Option 1: Use detached mode (recommended)
./build_image.sh

# Option 2: Build with explicit detach flag
modal run --detach modal_train.py::download_sample_dataset

# Option 3: Stay connected and wait (10-15 minutes)
# Keep your terminal open during the entire build
```

**Tips:**
- First build takes 10-15 minutes
- Subsequent builds use cached layers (much faster)
- Modal caches the image, so you only build once
- Use `--detach` to disconnect safely

### 6. Model Download Failures

**Error:**
```
HuggingFace Hub connection timeout
```

**Solutions:**
- Set `HF_HOME` environment variable (already set)
- Use `huggingface-cli login` if model requires authentication
- Check Modal's network connectivity

## Debug Tips

### 1. Check PaddlePaddle Installation

```python
# Check version
python -c "import paddle; print(paddle.__version__)"

# Check for fused_rms_norm_ext
python -c "from paddle.incubate.nn.functional import fused_rms_norm_ext; print('✓ fused_rms_norm_ext OK')"

# Check for flex_checkpoint module
python -c "from paddle.distributed.flex_checkpoint import dcp; print('✓ flex_checkpoint OK')"
```

### 2. Verify GPU Access

```python
python -c "import paddle; print(f'CUDA: {paddle.device.is_compiled_with_cuda()}'); print(f'GPUs: {paddle.device.cuda.device_count()}')"
```

### 3. Check ERNIEKit Installation

```bash
which erniekit
erniekit --help
```

### 4. Validate Config File

```bash
python -c "import yaml; print(yaml.safe_load(open('/paddle/ERNIE/examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml')))"
```

## Testing Checklist

Run these tests before training:

```bash
# 1. Quick validation
python quick_test.py

# 2. Full test suite
python test_setup.py

# 3. Specific tests
python test_setup.py paddle        # PaddlePaddle
python test_setup.py paddle-gpu    # GPU access
python test_setup.py ernie-cmd     # ERNIEKit CLI
```

## Getting Help

If issues persist:

1. Check the [ERNIE repository issues](https://github.com/PaddlePaddle/ERNIE/issues)
2. Review [PaddlePaddle installation docs](https://www.paddlepaddle.org.cn/install/quick)
3. Check [Modal documentation](https://modal.com/docs)
4. Verify your Modal GPU quota and limits
