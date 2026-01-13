"""
Test script to verify the complete Tagging Tool workflow
"""
import os
import sys

def test_complete_workflow():
    """Test the complete workflow from loading to processing."""
    print("=" * 60)
    print("Testing Tagging Tool - Complete Workflow")
    print("=" * 60)

    try:
        # Import the backend module
        print("\n[1/5] Importing backend module...")
        import back
        print("✓ Backend module imported successfully")

        # Load TIF files
        print("\n[2/5] Loading TIF files...")
        tif_files = back.load_tif_files()
        print(f"✓ Found {len(tif_files)} TIF files")

        if len(tif_files) == 0:
            print("✗ No TIF files found. Cannot continue test.")
            return False

        # Get current TIF
        print("\n[3/5] Loading first TIF file...")
        current_tif = back.get_current_tif()
        if not current_tif:
            print("✗ Failed to load current TIF")
            return False
        print(f"✓ Loaded TIF: {os.path.basename(current_tif['path'])}")
        print(f"   Metadata keys: {list(current_tif['metadata'].keys())}")

        # Test processing
        print("\n[4/5] Testing image processing...")
        tif_path = current_tif["path"]
        results = back.process_tif_detection(tif_path)

        if not results:
            print("✗ Processing failed")
            return False

        required_keys = [
            "image", "cropped_image", "cropped_extended_image",
            "annotated_image", "masked_image_cropped"
        ]

        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            print(f"✗ Missing result keys: {missing_keys}")
            return False

        print("✓ Processing successful")
        print(f"   Generated images: {list(results.keys())}")

        # Test navigation
        print("\n[5/5] Testing navigation...")
        initial_index = back.current_tif_index
        back.next_tif()
        after_next = back.current_tif_index
        back.prev_tif()
        after_prev = back.current_tif_index

        if after_next == initial_index + 1 and after_prev == initial_index:
            print("✓ Navigation working correctly")
        else:
            print(f"✗ Navigation failed: {initial_index} -> {after_next} -> {after_prev}")
            return False

        print("\n" + "=" * 60)
        print("✓ All workflow tests passed!")
        print("=" * 60)
        print("\nThe Tagging Tool is ready to use.")
        print("Run 'streamlit run app_NEW.py' to start the application.")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_workflow()
    sys.exit(0 if success else 1)

