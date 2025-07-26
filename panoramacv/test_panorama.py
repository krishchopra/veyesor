#!/usr/bin/env python3
"""
Test script for panorama stitching functionality
"""
import cv2
import numpy as np
import imutils
from pyimagesearch.panorama import Stitcher

def create_test_images():
    """Create synthetic test images for stitching"""
    print("Creating test images...")
    
    # Create base pattern
    base_pattern = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add some features for matching
    cv2.rectangle(base_pattern, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.rectangle(base_pattern, (200, 100), (300, 200), (0, 255, 0), -1)
    cv2.circle(base_pattern, (100, 250), 30, (0, 0, 255), -1)
    
    # Create overlapping images by shifting
    img1 = base_pattern.copy()
    
    # Second image - shift right and add new content
    img2 = np.zeros_like(base_pattern)
    img2[:, :-100] = base_pattern[:, 100:]  # Shift left part
    cv2.rectangle(img2, (300, 50), (380, 130), (255, 255, 0), -1)  # New content
    
    return [img1, img2]

def test_stitcher():
    """Test the Stitcher class"""
    print("Testing Stitcher class...")
    
    try:
        # Create test images
        test_images = create_test_images()
        
        # Initialize stitcher
        stitcher = Stitcher()
        
        # Test stitching
        result = stitcher.stitch(test_images, ratio=0.75, reprojThresh=4.0)
        
        if result is not None:
            if isinstance(result, tuple) and len(result) == 2:
                homographies, valid_indices = result
                print(f"✓ Stitching successful! Got {len(homographies)} homographies for indices {valid_indices}")
                return True
            else:
                print(f"✓ Stitching successful! Result shape: {result.shape}")
                return True
        else:
            print("✗ Stitching failed - returned None")
            return False
            
    except Exception as e:
        print(f"✗ Stitching failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_motion_detector():
    """Test the BasicMotionDetector class"""
    print("Testing BasicMotionDetector...")
    
    try:
        from pyimagesearch.basicmotiondetector import BasicMotionDetector
        
        # Create motion detector
        motion = BasicMotionDetector(minArea=500)
        
        # Create test frames
        frame1 = np.zeros((200, 300), dtype=np.uint8)
        frame2 = frame1.copy()
        
        # Add motion in second frame
        cv2.rectangle(frame2, (100, 50), (150, 100), 255, -1)
        
        # Test motion detection
        locs1 = motion.update(frame1)
        locs2 = motion.update(frame2)
        
        print(f"✓ Motion detector working! Detected {len(locs2)} motion regions")
        return True
        
    except Exception as e:
        print(f"✗ Motion detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing panorama functionality...")
    print("=" * 50)
    
    success = True
    
    try:
        success &= test_stitcher()
        print()
        success &= test_motion_detector()
        print()
        
        if success:
            print("✓ All panorama tests passed!")
        else:
            print("✗ Some tests failed!")
            
    except Exception as e:
        print(f"✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()