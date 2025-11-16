"""
Quick test to verify basic Modal image build.
Tests only essential components to speed up validation.

This imports from test_setup.py which uses the actual modal_train app.
"""

import sys
import os

# Import from test_setup which uses the actual app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_setup import (
    app,
    test_python_version,
    test_paddle_import,
    test_ernie_directory,
)


if __name__ == "__main__":
    """Quick validation of essential components."""
    print("\n" + "="*50)
    print("Quick Setup Validation")
    print("="*50 + "\n")

    tests = [
        ("Python Version", test_python_version),
        ("PaddlePaddle Import", test_paddle_import),
        ("ERNIE Repository", test_ernie_directory),
    ]

    with app.run():
        passed = 0
        for name, test_func in tests:
            print(f"Testing {name}...")
            try:
                result = test_func.remote()
                print(f"  {result}\n")
                passed += 1
            except Exception as e:
                print(f"  ✗ FAILED: {str(e)}\n")

        print("="*50)
        if passed == len(tests):
            print(f"✓ Quick tests passed ({passed}/{len(tests)})")
            print("\nRun full test suite:")
            print("  python test_setup.py")
        else:
            print(f"⚠️  Some tests failed ({passed}/{len(tests)})")
            print("Check errors above.")
