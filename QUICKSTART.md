# Quick Start: Train PaddleOCR-VL on Modal

## The Simplest Way (Recommended)

Just use `modal run` - it's the easiest:

```bash
# Download sample dataset
modal run modal_train.py --mode download-dataset

# Start training
modal run modal_train.py --mode train
```

Yes, `modal run` rebuilds the image each time, but:
- ✅ With our `--no-deps` optimization, the build is now **fast** (~10 min vs 30+ min before)
- ✅ Simple one-line commands
- ✅ No deployment management needed
- ✅ Perfect for getting started

## Complete Example

```bash
# 1. Download sample Bengali dataset
modal run modal_train.py --mode download-dataset

# 2. Start training (~2 hours)
modal run modal_train.py --mode train

# 3. Monitor progress in another terminal
modal app logs paddleocr-vl-sft --follow

# 4. Run inference after training
modal run modal_train.py --mode inference \
    --image-path "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/bengali_sft/5b/7a/5b7a5c1c-207a-4924-b5f3-82890dc7b94a.png"
```

## With Custom Data

```bash
# Upload your dataset
modal volume put paddleocr-vl-data my_data.jsonl /data/my_data.jsonl

# Train on it
modal run modal_train.py --mode train \
    --train-data-path /data/my_data.jsonl \
    --output-dir /outputs/my-custom-model

# Download trained model
modal volume get paddleocr-vl-outputs \
    /outputs/my-custom-model \
    ./trained_model
```

## Time Estimates

With our optimizations:
- **Image build**: ~10 minutes (down from 30+ minutes!)
- **Training**: ~2 hours on A100-40GB
- **Total first run**: ~2 hours 10 minutes

## The Key Optimization

Our image build is now **3x faster** because:
```python
# Before: pip downloads multiple 1.7GB wheels (slow!)
pip install paddlepaddle-gpu

# After: pre-install deps, then install with --no-deps (fast!)
pip install httpx numpy ... nvidia-cuda-*
pip install --no-deps paddlepaddle-gpu
```

Only **ONE** 1.7GB download instead of 4-5! 🚀
