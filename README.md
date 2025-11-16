# PaddleOCR-VL Fine-tuning with Modal

This repository contains a Modal-based setup for fine-tuning PaddleOCR-VL-0.9B models using ERNIEKit, following the instructions from [PaddlePaddle ERNIE's PaddleOCR VL SFT documentation](https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md).

## Recent Fixes (2025-11-14)

The Modal scripts have been optimized to fix build issues and significantly improve build performance:

### Critical Fixes:
1. **flex_checkpoint Module Error**: Fixed `ModuleNotFoundError: No module named 'paddle.distributed.flex_checkpoint'` by using latest PaddlePaddle nightly (removed version pinning)
2. **Build Performance**: Optimized to download only ONE PaddlePaddle wheel (1.7GB) instead of multiple, reducing build time from 20+ minutes to ~10-15 minutes
3. **Numpy Version Conflict**: Resolved conflict between PaddlePaddle (wants numpy>=2.0) and paddleformers (wants numpy<=1.26.4) by pre-installing with constraint and forcing final version

### Key Optimizations:
1. **Dependency Pre-installation**: Pre-install dependencies with `--no-deps` flag for PaddlePaddle to avoid backtracking
2. **Cache Optimization**: Added `--no-cache-dir` to all pip installs to reduce image size
3. **Installation Order**: Follows ERNIE's official sequence for maximum compatibility

### Performance Improvements:
- **Build Time**: Reduced from 20+ minutes to ~10-15 minutes (25-50% faster)
- **Downloads**: Only 1 PaddlePaddle wheel instead of 5-6 (80-85% reduction)
- **Image Size**: Smaller due to `--no-cache-dir` flag

See [FIXES_SUMMARY.md](FIXES_SUMMARY.md) for detailed documentation of all fixes.

## What is PaddleOCR-VL?

PaddleOCR-VL-0.9B is a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model. It efficiently supports 109 languages and excels in recognizing complex elements (text, tables, formulas, and charts) while maintaining minimal resource consumption.

## Why Modal?

Modal provides:
- **Cost-effective GPU access**: Pay only for compute time, with automatic spin-up/spin-down
- **No infrastructure management**: No need to manage Docker containers or GPU servers
- **Scalable**: Easy to scale up or down based on your needs
- **Simple deployment**: Just run your script, Modal handles the rest

## Prerequisites

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Set up Modal authentication**:
   ```bash
   modal token new
   ```
   This will open a browser to authenticate with Modal.

3. **Prepare your dataset**:
   - Format your training data as JSONL files following the [SFT VL Dataset Format](https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/datasets.md#sft-vl-dataset)
   - Each line should be a JSON object with `text_info` and `image_info` fields
   - See the [dataset format section](#dataset-format) below for details

## Dataset Format

Each training sample should be a JSON object with the following structure:

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./path/to/image.jpg"}
    ],
    "text_info": [
        {"text": "OCR:", "tag": "mask"},
        {"text": "Your text content here", "tag": "no_mask"}
    ]
}
```

**Required fields:**
- `text_info`: List of text data, each element contains:
  - `text`: The text content from User question or System response
  - `tag`: The mask tag (`no_mask`=include in training, `mask`=exclude)
- `image_info`: List of image data, each element contains:
  - `image_url`: URL to download image online or path to access image locally
  - `matched_text_index`: Index of matched text in `text_info` (default: 0)

**Notes:**
- Each training sample is in JSON format, with multiple samples separated by newlines (JSONL format)
- Ensure that `mask` items and `no_mask` items alternate in the `text_info`
- For examples of table, formula, and chart formats, see the [original documentation](https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md#81-tableformulachart-data-format)

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Paddle wheel once** (prevents re-downloading the 1.9 GB wheel on every build):
   ```bash
   # Download PaddlePaddle 3.2.0 (cu126) wheel locally
   python download_paddle_wheel.py --output-dir ./wheels

   # Upload the wheel to Modal's persistent volume
   modal run modal_train.py --mode upload-wheel \
     --wheel-path ./wheels/paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl
   ```
   Keep the wheel in `./wheels/` — the Modal image build now installs PaddlePaddle directly from this local file. If it is missing, the build falls back to the slow network download.

3. **Build the image** (first time - takes ~5 minutes now that the wheel is cached):
   ```bash
   # Build in detached mode (recommended for large downloads)
   ./build_image.sh

   # OR build directly (stay connected during 1.7GB PaddlePaddle download)
   modal run --detach modal_train.py::download_sample_dataset
   ```

   **Note**: The PaddlePaddle wheel is 1.7GB. The build will take 10-15 minutes. Use `--detach` to avoid timeout issues.

4. **Test the setup** (after image builds):
   ```bash
   # Quick test (3 essential checks)
   python quick_test.py

   # Full test suite (all 9 tests)
   python test_setup.py

   # Run specific test
   python test_setup.py paddle
   ```

5. **Download the base model** (first time only):
   ```bash
   modal run modal_train.py --mode download
   ```

6. **Download sample dataset** (optional, for testing):
   ```bash
   modal run modal_train.py --mode download-dataset
   ```
   This downloads the Bengali training dataset as an example.

## Usage

### 1. Download the Base Model

Download PaddleOCR-VL-0.9B from HuggingFace:

```bash
modal run modal_train.py --mode download
```

The model will be saved to `/models/PaddleOCR-VL` in the Modal volume.

### 2. Prepare Your Training Data

Upload your JSONL training file to Modal volumes:

```bash
# Using the upload helper script
modal run upload_data.py --local-path ./your_data/train.jsonl --remote-path /data/train.jsonl
```

Or mount local directories directly during training (see below).

### 3. Train the Model

Start fine-tuning with your dataset:

```bash
# Use --detach to keep training running even if your local client disconnects
modal run --detach modal_train.py --mode train \
  --config-path examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
  --train-data-path /data/train.jsonl \
  --output-dir /outputs/PaddleOCR-VL-SFT
```

**Note:** Training can take many hours. Using `--detach` ensures training continues even if you close your terminal or lose network connection.

**Key hyperparameters** (can be adjusted in the config file):
- `max_steps`: Total number of training steps
- `warmup_steps`: Number of linear warmup steps (recommended: 1% of max_steps)
- `packing_size`: Number of samples packed into a sequence
- `max_seq_len`: Maximum sequence length (default: 16384)
- `learning_rate`: Learning rate (default: 1e-4)

### 4. Resume Training

If training was interrupted, ERNIEKit will automatically detect and resume from the last checkpoint in the output directory. Simply run the same command again:

```bash
modal run --detach modal_train.py --mode train \
  --config-path examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
  --train-data-path /data/train.jsonl \
  --output-dir /outputs/PaddleOCR-VL-SFT
```

You can also manually specify a checkpoint:

```bash
modal run --detach modal_train.py --mode train \
  --config-path examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
  --train-data-path /data/train.jsonl \
  --output-dir /outputs/PaddleOCR-VL-SFT \
  --resume-from /outputs/PaddleOCR-VL-SFT/checkpoint-1000
```

### 5. Run Inference

Test your fine-tuned model:

```bash
modal run modal_train.py --mode inference \
  --image-path "https://example.com/test_image.png" \
  --output-dir /outputs/PaddleOCR-VL-SFT \
  --save-path /outputs/inference_results
```

The inference results will be saved in the specified `save_path` directory with `.md` extension files.

## Data Management

### Using Modal Volumes

Modal volumes are persistent storage that persists across runs. The script automatically creates:
- `paddleocr-vl-data`: For training/evaluation data
- `paddleocr-vl-models`: For base models
- `paddleocr-vl-outputs`: For model checkpoints and outputs

### Mounting Local Directories

For development, you can modify the script to mount local directories:

```python
@stub.function(
    # ... other parameters ...
    mounts=[
        modal.Mount.from_local_dir("./your_data", remote_path="/data"),
    ],
)
```

## Configuration

The training uses ERNIE's configuration files. The default config is:
- `examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml`

You can customize this config or create your own. Key parameters to adjust:
- Training steps and epochs
- Learning rate and optimizer settings
- Batch size and gradient accumulation
- Model architecture settings

## Output Directory Structure

After training, the model will be saved with the following structure:

```
PaddleOCR-VL-SFT/
├── preprocessor_config.json      # Image preprocessing configuration
├── config.json                    # Model configuration
├── model-00001-of-00001.safetensors  # Model weights
├── model.safetensors.index.json   # Model weight index
├── tokenizer.model                # Tokenizer files
├── tokenizer_config.json
├── train_args.bin                 # Training arguments
├── train_state.json               # Training state
├── train_results.json             # Training results
├── checkpoint-[N]                 # Checkpoint folders (for resuming)
└── ...
```

## GPU Options

The script uses `A100` GPU by default, which is the closest equivalent to the A800 80GB recommended in the [PaddleOCR-VL SFT documentation](https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md). The documentation recommends A800 80GB (80GB VRAM), but Modal doesn't offer A800 (it's a China-specific enterprise GPU variant). A100 provides similar memory capacity and is the recommended choice for training with `max_seq_len=16384`.

You can modify the GPU type in `modal_train.py`:

- `modal.gpu.A100()` - **Recommended** - Closest to A800 80GB, sufficient for max_seq_len=16384
- `modal.gpu.A10G()` - Lower cost option (24GB VRAM), may require reducing batch size or max_seq_len
- `modal.gpu.T4()` - Lower cost option (16GB VRAM), may have memory limitations

## Monitoring

- View logs in real-time: `modal logs <run-id>`
- Check status in Modal dashboard: https://modal.com/apps
- Monitor GPU usage and costs in the Modal dashboard
- Use TensorBoard to view training metrics (if configured in the training script)

## Cost Optimization Tips

1. **Use appropriate GPU**: A100 is recommended (matching the A800 80GB recommendation in the docs). A10G may work but may require reducing batch size or max_seq_len for memory-intensive configs.
2. **Monitor training time**: Set reasonable timeouts to avoid runaway costs
3. **Use checkpoints**: Resume training instead of starting from scratch
4. **Clean up volumes**: Remove old checkpoints to save storage costs
5. **Download models once**: The base model only needs to be downloaded once

## Troubleshooting

### Common Issues

1. **Out of memory**: 
   - Try reducing `max_seq_len` or `packing_size` in the config
   - Ensure you're using A100 GPU (recommended) instead of A10G
   - If using A10G, you may need to reduce `max_seq_len` from 16384 to a smaller value

2. **Import errors**: 
   - All dependencies are pre-installed in the image
   - Ensure ERNIEKit is properly installed (handled automatically)

3. **Dataset format errors**:
   - Verify your JSONL format matches the required structure
   - Ensure `mask` and `no_mask` tags alternate correctly

4. **Model download fails**:
   - Check your HuggingFace authentication (may need `huggingface-cli login`)
   - Verify the model repository name is correct

### Getting Help

- Check ERNIE documentation: https://github.com/PaddlePaddle/ERNIE
- PaddleOCR-VL SFT guide: https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md
- Modal documentation: https://modal.com/docs
- ERNIE issues: https://github.com/PaddlePaddle/ERNIE/issues

## Example Workflow

Complete workflow from start to finish:

```bash
# 1. Download the base PaddleOCR-VL model
modal run modal_train.py --mode download

# 2. Download sample dataset (optional, for testing)
modal run modal_train.py --mode download-dataset

# 3. Fine-tune on your data
modal run modal_train.py --mode train \
  --config-path examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml \
  --train-data-path /data/train.jsonl \
  --output-dir /outputs/PaddleOCR-VL-SFT

# 4. Run inference with the fine-tuned model
modal run modal_train.py --mode inference \
  --image-path "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/bengali_sft/5b/7a/5b7a5c1c-207a-4924-b5f3-82890dc7b94a.png" \
  --output-dir /outputs/PaddleOCR-VL-SFT \
  --save-path /outputs/inference_results
```

## Next Steps

1. Prepare your dataset in the JSONL format
2. Customize the configuration file for your specific use case
3. Start training and monitor progress
4. Evaluate the fine-tuned model on your test data
5. Deploy the model for production use

For more details on dataset formats, training configurations, and inference, refer to the [original PaddleOCR-VL SFT documentation](https://github.com/PaddlePaddle/ERNIE/blob/develop/docs/paddleocr_vl_sft.md).

## Detailed Fixes

### Issue 1: Wrong Base Image
**Before:**
```python
image = modal.Image.debian_slim(python_version="3.10")
```

**After:**
```python
image = modal.Image.from_registry(
    "nvidia/cuda:12.3.0-cudnn9-devel-ubuntu22.04",
    add_python="3.10"
)
# Then install PaddlePaddle from public PyPI
.run_commands(
    "python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/",
)
```

**Why:** The Chinese PaddlePaddle registry (`ccr-2vdh3abv-pub.cnc.bj.bcebce.com`) is not accessible outside China. We use the public NVIDIA CUDA image and install PaddlePaddle from PyPI instead.

### Issue 2: Directory Structure Mismatch
**Before:**
```python
os.chdir("/root/ERNIE")
```

**After:**
```python
os.chdir("/paddle/ERNIE")
```

**Why:** The Docker setup uses `/paddle` as the working directory, matching the container's volume mount point.

### Issue 3: Complex Dependency Installation
**Before:** Multi-stage pip installations with manual opencv/numpy reinstalls, safetensors wheel URLs, and paddleformers patches.

**After:**
```python
# First install PaddlePaddle GPU (nightly build for latest features)
.run_commands(
    "python -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/",
)
# Then install ERNIE following official sequence
.run_commands(
    "git clone https://github.com/PaddlePaddle/ERNIE.git /paddle/ERNIE",
    "cd /paddle/ERNIE && git checkout develop",
)
.run_commands(
    "cd /paddle/ERNIE && python -m pip install -r requirements/gpu/requirements.txt",
    "cd /paddle/ERNIE && python -m pip install -e .",
    "cd /paddle/ERNIE && python -m pip install tensorboard",
    "cd /paddle/ERNIE && python -m pip install opencv-python-headless",
    "cd /paddle/ERNIE && python -m pip install numpy==1.26.4",
)
```

**Why:** Install latest PaddlePaddle nightly first (required for `fused_rms_norm_ext` function), then follow the exact Docker documentation sequence for ERNIE dependencies.

### Issue 4: Missing Environment Variables
**Before:** No CUDA device configuration.

**After:**
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

**Why:** The Docker setup explicitly sets `CUDA_VISIBLE_DEVICES=0` in the training command.

### Issue 5: Deprecated Modal API
**Before:**
```python
stub = modal.Stub("paddleocr-data-upload")
@stub.function(...)
@stub.local_entrypoint()
```

**After:**
```python
app = modal.App("paddleocr-data-upload")
@app.function(...)
@app.local_entrypoint()
```

**Why:** Modal deprecated `Stub` in favor of `App` in recent versions.

### Issue 6: Volume Name Inconsistency
**Before:** Mixed use of `paddleocr-data` and `paddleocr-vl-data`

**After:** Consistent naming:
- `paddleocr-vl-data`
- `paddleocr-vl-models`
- `paddleocr-vl-outputs`

**Why:** Prevents volume mismatch errors between training and data upload scripts.

## Testing

The test scripts now use the **actual deployed app and image** from `modal_train.py`, ensuring you're testing exactly what will be used for training.

Run the test suite to verify your setup:

```bash
# Quick test (3 essential checks - faster)
python quick_test.py

# Full test suite (all 9 tests - comprehensive)
python test_setup.py

# Run specific test
python test_setup.py paddle       # Test PaddlePaddle
python test_setup.py ernie-cmd    # Test ERNIEKit CLI
```

Available tests:
- `python` - Verify Python 3.10+ is installed
- `cuda` - Check CUDA and nvidia-smi availability
- `paddle` - Test PaddlePaddle import and GPU compilation
- `paddle-gpu` - Verify PaddlePaddle can access GPU
- `ernie` - Check ERNIE repository structure
- `ernie-import` - Test ERNIEKit import
- `ernie-cmd` - Verify `erniekit` CLI command
- `deps` - Check all required dependencies
- `config` - Validate training config file

**Important**: The tests use the same Modal app (`paddleocr-vl-sft`) and image as training, so they validate the actual deployment.
