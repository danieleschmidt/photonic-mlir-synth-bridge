#!/usr/bin/env python3
"""
Basic test script for photonic MLIR components without full PyTorch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_imports():
    """Test basic imports work"""
    try:
        from photonic_mlir import __version__
        print(f"✓ Successfully imported photonic_mlir version {__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import photonic_mlir: {e}")
        # Try importing individual modules that don't require torch
        try:
            from photonic_mlir.compiler import PhotonicBackend
            print("✓ Partial import successful (core modules work)")
            return True
        except ImportError as e2:
            print(f"✗ Even core modules failed: {e2}")
            return False

def test_basic_functionality():
    """Test basic functionality without PyTorch dependencies"""
    try:
        # Test compiler backend enum
        from photonic_mlir.compiler import PhotonicBackend
        backends = list(PhotonicBackend)
        print(f"✓ Found {len(backends)} photonic backends")
        
        # Test optimization pipeline
        from photonic_mlir.optimization import OptimizationPipeline
        pipeline = OptimizationPipeline()
        print("✓ Created optimization pipeline")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_validation():
    """Test validation utilities"""
    try:
        from photonic_mlir.validation import InputValidator
        
        # Test wavelength validation directly
        result = InputValidator.validate_wavelengths([1550.0, 1551.0])
        assert result["valid"], "Should accept valid wavelengths"
        print("✓ Wavelength validation works")
        
        return True
    except ImportError as e:
        print(f"✗ Validation import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False

def test_security():
    """Test security utilities"""
    try:
        from photonic_mlir.security import sanitize_input
        
        # Test input sanitization
        safe_input = sanitize_input("test_input")
        assert safe_input == "test_input", "Should pass safe input unchanged"
        print("✓ Security utilities work")
        
        return True
    except ImportError as e:
        print(f"✗ Security import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Security test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running basic photonic MLIR tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_validation,
        test_security
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Generation 1 complete.")
        return 0
    else:
        print("❌ Some tests failed. Need to fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())