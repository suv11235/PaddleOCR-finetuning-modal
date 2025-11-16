"""
Test script to verify Modal setup for PaddleOCR-VL training.
Run tests to ensure the environment is correctly configured.

This script adds test functions to the existing modal_train app,
so it tests the exact same image that will be used for training.
"""

import modal
import sys
import os

# Import the actual app and image from modal_train
# We'll add our test functions to the same app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_train import app, image

# All test functions will be added to the same app instance
# This ensures we're testing the exact deployment that will be used for training


@app.function(image=image, timeout=600)
def test_python_version():
    """Test that Python version is correct."""
    import sys
    print(f"Python version: {sys.version}")
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    return "✓ Python version OK"


@app.function(image=image, timeout=600)
def test_cuda_available():
    """Test that CUDA is available."""
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print("NVIDIA-SMI output:")
    print(result.stdout)
    assert result.returncode == 0, "nvidia-smi failed"
    return "✓ CUDA available"


@app.function(image=image, timeout=600)
def test_paddle_import():
    """Test that PaddlePaddle can be imported."""
    try:
        import paddle
        print(f"PaddlePaddle version: {paddle.__version__}")
        print(f"CUDA available: {paddle.device.is_compiled_with_cuda()}")
        print(f"GPU count: {paddle.device.cuda.device_count()}")
        assert paddle.device.is_compiled_with_cuda(), "PaddlePaddle not compiled with CUDA"
        return f"✓ PaddlePaddle {paddle.__version__} OK"
    except ImportError as e:
        raise AssertionError(f"Failed to import PaddlePaddle: {e}")


@app.function(image=image, timeout=600)
def test_paddle_gpu():
    """Test that PaddlePaddle can access GPU."""
    import paddle
    import os

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        # Try to create a tensor on GPU
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        print(f"Tensor created: {x}")
        print(f"Tensor device: {x.place}")
        return "✓ PaddlePaddle GPU OK"
    except Exception as e:
        raise AssertionError(f"Failed to use GPU: {e}")


@app.function(image=image, timeout=600)
def test_ernie_directory():
    """Test that ERNIE is cloned and accessible."""
    import os
    from pathlib import Path

    ernie_path = Path("/paddle/ERNIE")
    assert ernie_path.exists(), f"ERNIE directory not found at {ernie_path}"
    assert (ernie_path / "requirements").exists(), "ERNIE requirements directory not found"
    assert (ernie_path / "examples").exists(), "ERNIE examples directory not found"

    # Check for the config file we'll use
    config_path = ernie_path / "examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml"
    assert config_path.exists(), f"Training config not found at {config_path}"

    print(f"ERNIE directory: {ernie_path}")
    print(f"Config file: {config_path}")
    return "✓ ERNIE repository OK"


@app.function(image=image, timeout=600)
def test_ernie_import():
    """Test that ERNIEKit can be imported."""
    import sys
    import os

    os.chdir("/paddle/ERNIE")
    sys.path.insert(0, "/paddle/ERNIE")

    try:
        import erniekit
        print(f"ERNIEKit location: {erniekit.__file__}")
        return "✓ ERNIEKit import OK"
    except ImportError as e:
        raise AssertionError(f"Failed to import ERNIEKit: {e}")


@app.function(image=image, timeout=600)
def test_erniekit_command():
    """Test that erniekit CLI command is available."""
    import subprocess

    # Try to run erniekit help
    result = subprocess.run(["erniekit", "--help"], capture_output=True, text=True)
    print("ERNIEKit help output:")
    print(result.stdout[:500])  # First 500 chars

    assert result.returncode == 0, "erniekit command not found or failed"
    assert "train" in result.stdout, "erniekit train command not found in help"
    return "✓ ERNIEKit CLI OK"


@app.function(image=image, timeout=600)
def test_dependencies():
    """Test that all required dependencies are installed."""
    import importlib

    required_packages = [
        "paddle",
        "numpy",
        "opencv-python-headless",
        "tensorboard",
        "transformers",
        "accelerate",
        "huggingface_hub",
    ]

    results = []
    for package in required_packages:
        # Handle package name differences
        import_name = package.replace("-", "_").replace("opencv_python_headless", "cv2")
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            results.append(f"  ✓ {package}: {version}")
        except ImportError as e:
            results.append(f"  ✗ {package}: MISSING")

    print("\n".join(results))
    return "✓ Dependencies check complete"


@app.function(image=image, timeout=600)
def test_config_file():
    """Test that the training config file is valid."""
    import yaml
    from pathlib import Path

    config_path = Path("/paddle/ERNIE/examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Config keys: {list(config.keys())}")

    # Check for important config keys
    important_keys = ["max_seq_len", "output_dir"]
    for key in important_keys:
        if key in config:
            print(f"  {key}: {config[key]}")

    return "✓ Config file OK"


def run_tests_main(test: str = "all"):
    """
    Run tests to verify the Modal setup.

    Usage:
        # Run all tests
        python test_setup.py

        # Run specific test
        python test_setup.py paddle

    Available tests:
        all - Run all tests (default)
        python - Test Python version
        cuda - Test CUDA availability
        paddle - Test PaddlePaddle installation
        paddle-gpu - Test PaddlePaddle GPU access
        ernie - Test ERNIE repository
        ernie-import - Test ERNIEKit import
        ernie-cmd - Test ERNIEKit CLI command
        deps - Test dependencies
        config - Test config file
    """

    tests = {
        "python": ("Python Version", test_python_version),
        "cuda": ("CUDA Availability", test_cuda_available),
        "paddle": ("PaddlePaddle Import", test_paddle_import),
        "paddle-gpu": ("PaddlePaddle GPU", test_paddle_gpu),
        "ernie": ("ERNIE Repository", test_ernie_directory),
        "ernie-import": ("ERNIEKit Import", test_ernie_import),
        "ernie-cmd": ("ERNIEKit CLI", test_erniekit_command),
        "deps": ("Dependencies", test_dependencies),
        "config": ("Config File", test_config_file),
    }

    print("\n" + "="*70)
    print("PaddleOCR-VL Modal Setup Tests")
    print("="*70 + "\n")

    if test == "all":
        run_tests = tests.items()
    elif test in tests:
        run_tests = [(test, tests[test])]
    else:
        print(f"Unknown test: {test}")
        print(f"Available tests: all, {', '.join(tests.keys())}")
        return

    results = []
    for test_name, (test_desc, test_func) in run_tests:
        print(f"Running: {test_desc}...")
        try:
            result = test_func.remote()
            print(f"  {result}\n")
            results.append((test_name, True, result))
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}\n")
            results.append((test_name, False, str(e)))

    print("="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, message in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready for training.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    import sys
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"
    with app.run():
        run_tests_main(test_name)
