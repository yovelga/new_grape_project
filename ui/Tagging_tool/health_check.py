#!/usr/bin/env python3
"""
Quick health check for the Tagging Tool
Run this anytime to verify the application is ready to use
"""
import os
import sys

def quick_health_check():
    """Perform a quick health check of the Tagging Tool."""
    print("🔍 Tagging Tool - Quick Health Check")
    print("=" * 50)

    checks_passed = 0
    checks_total = 0

    # Check 1: Required files exist
    checks_total += 1
    required_files = ['app.py', 'back.py', 'requirements.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if not missing_files:
        print("✓ All required files present")
        checks_passed += 1
    else:
        print(f"✗ Missing files: {missing_files}")

    # Check 2: Can import backend
    checks_total += 1
    try:
        import back
        print("✓ Backend module imports successfully")
        checks_passed += 1
    except Exception as e:
        print(f"✗ Cannot import backend: {e}")
        return False

    # Check 3: Directories exist
    checks_total += 1
    if os.path.exists(back.tif_dir) and os.path.exists(back.image_dir):
        print("✓ Data directories accessible")
        checks_passed += 1
    else:
        print("✗ Data directories not found")

    # Check 4: TIF files available
    checks_total += 1
    try:
        tif_count = len(back.load_tif_files())
        if tif_count > 0:
            print(f"✓ Found {tif_count} TIF files")
            checks_passed += 1
        else:
            print("✗ No TIF files found")
    except Exception as e:
        print(f"✗ Error loading TIF files: {e}")

    # Summary
    print("=" * 50)
    print(f"Score: {checks_passed}/{checks_total} checks passed")

    if checks_passed == checks_total:
        print("✅ System healthy - ready to use!")
        print("\nStart the app with: streamlit run app.py")
        return True
    else:
        print("⚠️  Some issues detected")
        print("\nRun for detailed diagnostics: python validate_setup.py")
        return False

if __name__ == "__main__":
    success = quick_health_check()
    sys.exit(0 if success else 1)

