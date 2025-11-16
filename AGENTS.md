# Repository Guidelines

## Project Structure & Module Organization
This workspace fine-tunes PaddleOCR-VL on Modal. Modal entrypoints live in `modal_train.py`, with helper scripts `run_training.py`, `trigger_training.py`, and dataset utilities such as `upload_data.py` and `download_paddle_wheel.py`. Local ERNIE sources live under `ERNIE/`; treat them as an upstream mirror and avoid editing unless the change is meant to be upstreamed. Cached PaddlePaddle wheels live in `wheels/` and should be checked before builds. Smoke and validation scripts (`quick_test.py`, `test_setup.py`) stay in the repo root so they can run inside the Modal image. Documentation for operators sits in `README.md`, `README_TRAINING.md`, and `QUICKSTART.md`.

## Build, Test, and Development Commands
Install Python deps with `pip install -r requirements.txt`. Cache the GPU wheel locally via `python download_paddle_wheel.py --output-dir ./wheels` and upload it to Modal by running `modal run modal_train.py --mode upload-wheel --wheel-path ./wheels/<wheel.whl>`. Build the image once with `modal deploy modal_train.py`; subsequent `modal run modal_train.py --mode train --train-data-path /data/ocr_vl_sft-train.jsonl` reuse that build. Use `modal run modal_train.py --mode download-dataset` for sample data, `modal run modal_train.py --mode inference --image-path <url>` for sanity checks, and tail logs with `modal app logs paddleocr-vl-sft --follow`.

## Coding Style & Naming Conventions
All Python is 4-space indented, PEP 8 compliant, and typed where practical because Modal decorators validate arguments. File and function names remain snake_case (`download_paddle_wheel`, `train_paddleocr_vl`). Keep CLI flags kebab-case (e.g., `--train-data-path`) to match existing docs. Constants and environment switches should remain uppercase (`VOLUME_NAME`). Use docstrings to explain Modal entrypoints and prefer logging over prints.

## Testing Guidelines
Run `python quick_test.py` before opening a PR to ensure Modal credentials, CUDA, and Paddle imports work. Use `python test_setup.py` for the full suite, or scope to a module (`python test_setup.py paddle` or `python test_setup.py ernie-cmd`). Tests must pass against the same Modal image you plan to train on; paste the relevant command output in the PR if failures require discussion.

## Commit & Pull Request Guidelines
Commits should be short, action-oriented sentences in the imperative mood and scoped to the touched module (e.g., `modal_train: add dataset mirror`). Include context on Modal modes touched and reference docs you changed. Pull requests need: summary of changes, exact commands tested, Modal app/version info, and any dataset or volume paths you touched. Attach screenshots or log snippets when changes alter training output so reviewers can confirm behaviour.
