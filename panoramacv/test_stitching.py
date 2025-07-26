import cv2
import numpy as np
from pyimagesearch.panorama import Stitcher

def create_mock_frame(camera_id, frame_count=0):
    """Create a mock camera frame with different colors and overlapping content"""
    width, height = 400, 300
    
    # Different background colors for each camera
    colors = [(50, 100, 200), (100, 200, 50), (200, 50, 100)]
    bg_color = colors[camera_id % len(colors)]
    
    # Create frame
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Add camera label
    cv2.putText(frame, f"Camera {camera_id}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add some features for stitching (circles and rectangles)
    if camera_id == 0:  # Left camera
        cv2.circle(frame, (300, 150), 50, (255, 255, 255), 3)
        cv2.rectangle(frame, (250, 100), (350, 200), (255, 255, 255), 3)
    elif camera_id == 1:  # Middle camera  
        cv2.circle(frame, (200, 150), 50, (255, 255, 255), 3)
        cv2.rectangle(frame, (150, 100), (250, 200), (255, 255, 255), 3)
        cv2.circle(frame, (100, 150), 30, (255, 255, 255), 3)
    elif camera_id == 2:  # Right camera
        cv2.circle(frame, (100, 150), 50, (255, 255, 255), 3)
        cv2.rectangle(frame, (50, 100), (150, 200), (255, 255, 255), 3)
    
    return frame

def test_stitching():
    print("Testing panorama stitching...")
    
    # Create mock frames
    frames = []
    for i in range(3):
        frame = create_mock_frame(i)
        frames.append(frame)
        cv2.imwrite(f"test_frame_{i}.jpg", frame)
        print(f"Created test frame {i}")
    
    # Initialize stitcher
    stitcher = Stitcher()
    
    # Try to stitch frames
    print("Attempting to stitch frames...")
    result = stitcher.stitch(frames, ratio=0.7, reprojThresh=5.0)
    
    if result is None:
        print("Stitching failed!")
        return False
    
    homographies, valid_indices = result
    print(f"Stitching successful! Valid indices: {valid_indices}")
    print(f"Number of homographies: {len(homographies)}")
    
    # Test the manual stitching process like in the server
    corners = []
    for i, H in zip(valid_indices, homographies):
        if i >= len(frames):
            continue
        h, w = frames[i].shape[:2]
        pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
        warped = cv2.perspectiveTransform(pts, H)
        corners.append(warped)
        
    all_corners = np.concatenate(corners)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # Blend images
    final_result = None
    for idx, i in enumerate(valid_indices):
        if i >= len(frames) or idx >= len(homographies):
            continue
            
        warped = cv2.warpPerspective(frames[i], Ht.dot(homographies[idx]), 
                                    (xmax-xmin, ymax-ymin))

        if final_result is None:
            final_result = warped
        else:
            # Simple blending - just overlay non-black pixels
            mask = (warped > 0).any(axis=2)
            final_result[mask] = warped[mask]
    
    if final_result is not None:
        cv2.imwrite("stitched_result.jpg", final_result)
        print("Stitched result saved as 'stitched_result.jpg'")
        print(f"Result dimensions: {final_result.shape}")
        return True
    else:
        print("Final blending failed!")
        return False

if __name__ == "__main__":
    success = test_stitching()
    if success:
        print("✅ Stitching test passed!")
    else:
        print("❌ Stitching test failed!")