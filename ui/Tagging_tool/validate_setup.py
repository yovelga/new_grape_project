"""
Validation script to check if the Tagging_tool can access the required directories.
Run this before starting the main application to verify setup.
"""
import os
import sys

def validate_setup():
    """Validate that all required directories and files exist."""
    print("=" * 60)
    print("Tagging Tool - Setup Validation")
    print("=" * 60)

    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    print(f"\n📁 Script directory: {script_dir}")
    print(f"📁 Project root: {project_root}")

    # Check directories
    checks = []

    # Check masks directory
    masks_dir = os.path.join(project_root, 'src', 'preprocessing', 'items_for_cnn_train', 'masks')
    masks_exists = os.path.exists(masks_dir)
    checks.append(("Masks directory", masks_dir, masks_exists))

    if masks_exists:
        tif_files = [f for f in os.listdir(masks_dir) if f.endswith('.tif')]
        print(f"   ✓ Found {len(tif_files)} TIF files")

    # Check images directory
    images_dir = os.path.join(project_root, 'src', 'preprocessing', 'items_for_cnn_train', 'used')
    images_exists = os.path.exists(images_dir)
    checks.append(("Images directory", images_dir, images_exists))

    if images_exists:
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        print(f"   ✓ Found {len(image_files)} image files")

    # Check requirements
    requirements_path = os.path.join(script_dir, 'requirements.txt')
    requirements_exists = os.path.exists(requirements_path)
    checks.append(("Requirements file", requirements_path, requirements_exists))

    # Display results
    print("\n" + "=" * 60)
    print("Validation Results:")
    print("=" * 60)

    all_passed = True
    for name, path, exists in checks:
        status = "✓ PASS" if exists else "✗ FAIL"
        print(f"{status} - {name}")
        print(f"      {path}")
        if not exists:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed! You can run the application.")
        print("\nTo start the application, run:")
        print("  streamlit run app.py")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Make sure you're in the correct directory (ui/Tagging_tool)")
        print("  2. Verify that items_for_cnn_train exists in src/preprocessing/")
        print("  3. Check that mask files exist in the masks/ subdirectory")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)

