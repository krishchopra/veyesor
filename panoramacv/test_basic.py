#!/usr/bin/env python3
"""
Simple test script to verify basic computer vision functionality
"""
import cv2
import numpy as np
import imutils

def test_opencv():
    """Test basic OpenCV functionality"""
    print("Testing OpenCV...")
    
    # Test basic image creation and operations
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # Blue image
    
    # Test resize
    resized = imutils.resize(img, width=50)
    print(f"Original shape: {img.shape}, Resized shape: {resized.shape}")
    
    # Test color conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale shape: {gray.shape}")
    
    print("✓ OpenCV basic functionality working!")
    return True

def test_camera_fallback():
    """Test camera access with fallback to test image"""
    print("Testing camera access...")
    
    # Try to access camera (will likely fail in headless environment)
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("✓ Camera access working!")
            print(f"Frame shape: {frame.shape}")
        else:
            print("Camera opened but no frame read")
        cap.release()
    else:
        print("No camera available (expected in headless environment)")
    
    # Create a test pattern instead
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"✓ Test frame created: {test_frame.shape}")
    return True

if __name__ == "__main__":
    print("Running basic functionality tests...")
    print("=" * 50)
    
    try:
        test_opencv()
        print()
        test_camera_fallback()
        print()
        print("✓ All basic tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()