#!/usr/bin/env python3
"""
Test script to validate the VeriX Docker environment setup.
This script checks all dependencies and verifies GPU availability.
"""

import sys
import os

def test_python_version():
    """Check Python version."""
    print("=" * 60)
    print("Testing Python Version")
    print("=" * 60)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible with Marabou")
    else:
        print("✗ Warning: Python version may not be fully compatible")
    print()
    return True

def test_tensorflow():
    """Test TensorFlow import and GPU availability."""
    print("=" * 60)
    print("Testing TensorFlow")
    print("=" * 60)
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow imported successfully: {tf.__version__}")

        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU(s) available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("⚠ No GPU detected (CPU-only mode)")
            print("  Note: This may be expected if not running with --gpus flag")

        # Test basic operation
        with tf.device('/CPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print("✓ TensorFlow basic operations working")
        print()
        return True
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        print()
        return False

def test_keras():
    """Test Keras import."""
    print("=" * 60)
    print("Testing Keras")
    print("=" * 60)
    try:
        import keras
        print(f"✓ Keras imported successfully: {keras.__version__}")
        print()
        return True
    except Exception as e:
        print(f"✗ Keras import failed: {e}")
        print()
        return False

def test_onnx():
    """Test ONNX and ONNXRuntime."""
    print("=" * 60)
    print("Testing ONNX")
    print("=" * 60)
    try:
        import onnx
        print(f"✓ ONNX imported successfully: {onnx.__version__}")

        import onnxruntime as ort
        print(f"✓ ONNXRuntime imported successfully: {ort.__version__}")

        # Check available providers
        providers = ort.get_available_providers()
        print(f"✓ Available providers: {', '.join(providers)}")
        if 'CUDAExecutionProvider' in providers:
            print("✓ GPU execution provider available")
        else:
            print("⚠ GPU execution provider not available (CPU only)")

        print()
        return True
    except Exception as e:
        print(f"✗ ONNX test failed: {e}")
        print()
        return False

def test_tf2onnx():
    """Test tf2onnx."""
    print("=" * 60)
    print("Testing tf2onnx")
    print("=" * 60)
    try:
        import tf2onnx
        print(f"✓ tf2onnx imported successfully: {tf2onnx.__version__}")
        print()
        return True
    except Exception as e:
        print(f"✗ tf2onnx import failed: {e}")
        print()
        return False

def test_scientific_packages():
    """Test numpy, matplotlib, scikit-image."""
    print("=" * 60)
    print("Testing Scientific Packages")
    print("=" * 60)
    try:
        import numpy as np
        print(f"✓ NumPy imported successfully: {np.__version__}")

        import matplotlib
        print(f"✓ Matplotlib imported successfully: {matplotlib.__version__}")

        from skimage import __version__ as skimage_version
        print(f"✓ scikit-image imported successfully: {skimage_version}")

        print()
        return True
    except Exception as e:
        print(f"✗ Scientific packages test failed: {e}")
        print()
        return False

def test_marabou():
    """Test Marabou installation."""
    print("=" * 60)
    print("Testing Marabou")
    print("=" * 60)
    try:
        from maraboupy import Marabou
        print("✓ Marabou imported successfully")

        # Test creating Marabou options
        options = Marabou.createOptions(numWorkers=4, timeoutInSeconds=60, verbosity=0)
        print("✓ Marabou options created successfully")

        print()
        return True
    except Exception as e:
        print(f"✗ Marabou test failed: {e}")
        print()
        return False

def test_gurobi():
    """Test Gurobi availability (optional)."""
    print("=" * 60)
    print("Testing Gurobi (Optional)")
    print("=" * 60)
    try:
        import gurobipy
        print(f"✓ Gurobi imported successfully: {gurobipy.gurobi.version()}")

        # Check for license
        license_file = os.environ.get('GRB_LICENSE_FILE', '/opt/gurobi/gurobi.lic')
        if os.path.exists(license_file):
            print(f"✓ Gurobi license file found: {license_file}")
        else:
            print(f"⚠ Gurobi license file not found: {license_file}")
            print("  Note: License required for Gurobi-accelerated solving")

        print()
        return True
    except ImportError:
        print("⚠ Gurobi not available (will use default LP solver)")
        print("  Note: This is optional, Marabou can work without Gurobi")
        print()
        return True
    except Exception as e:
        print(f"⚠ Gurobi test warning: {e}")
        print("  Note: Gurobi license may need configuration")
        print()
        return True

def test_verix_compatibility():
    """Test VeriX-specific compatibility."""
    print("=" * 60)
    print("Testing VeriX Compatibility")
    print("=" * 60)

    # Test that we can import all VeriX dependencies together
    try:
        from maraboupy import Marabou
        import tensorflow as tf
        import keras
        import onnx
        import onnxruntime as ort
        import numpy as np
        from skimage.color import label2rgb
        from matplotlib import pyplot as plt

        print("✓ All VeriX dependencies can be imported together")
        print()
        return True
    except Exception as e:
        print(f"✗ VeriX compatibility test failed: {e}")
        print()
        return False

def main():
    """Run all tests."""
    print("\n")
    print("#" * 60)
    print("# VeriX Docker Environment Validation")
    print("#" * 60)
    print()

    tests = [
        ("Python Version", test_python_version),
        ("TensorFlow", test_tensorflow),
        ("Keras", test_keras),
        ("ONNX", test_onnx),
        ("tf2onnx", test_tf2onnx),
        ("Scientific Packages", test_scientific_packages),
        ("Marabou", test_marabou),
        ("Gurobi", test_gurobi),
        ("VeriX Compatibility", test_verix_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print()
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Environment is ready for VeriX.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
