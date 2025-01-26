from __future__ import print_function
from pyimagesearch.basicmotiondetector import BasicMotionDetector
from pyimagesearch.panorama import Stitcher
import numpy as np
import datetime
import imutils
import time
import cv2

# List all your camera URLs here in desired order (left to right).
camera_urls = [
    "http://10.217.5.80:4747/video",  # Benson
    "http://10.217.22.138:4747/video", # Krish
    "http://10.217.29.154:4747/video",  # Fahmi
    "http://10.217.13.254:4747/video", # Alex
]

print("[INFO] Starting camera streams...")
captures = [cv2.VideoCapture(url) for url in camera_urls]

# Verify that each capture stream is opened successfully
for i, cap in enumerate(captures):
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera stream at index {i}")
        exit(1)

time.sleep(2.0)

# Initialize the Stitcher and Motion Detector
stitcher = Stitcher()
motion = None  # We'll initialize this after first frame
totalFrames = 0
initial_frames = []
# Attempt to read a frame from each camera
for i, cap in enumerate(captures):
    ret, frame = cap.read()
    # Debugging prints
    print(f"[DEBUG] Stream {i} read success: {ret}")
    if not ret:
        print(f"[ERROR] Could not read frame from camera at index {i}")
        break
    frame = imutils.resize(frame, width=400)
    initial_frames.append(frame)

# If we didn't successfully get as many frames as cameras, stop.
if len(initial_frames) < len(captures):
    print("[INFO] Stopping due to read error.")
    exit(1)

(homographies, valid_indices) = stitcher.stitch(initial_frames, ratio=0.7, reprojThresh=5.0)

while True:
    frames = []
    # Attempt to read a frame from each camera
    for i, cap in enumerate(captures):
        ret, frame = cap.read()
        # Debugging prints
        print(f"[DEBUG] Stream {i} read success: {ret}")
        if not ret:
            print(f"[ERROR] Could not read frame from camera at index {i}")
            break
        frame = imutils.resize(frame, width=400)
        frames.append(frame)

    # If we didn't successfully get as many frames as cameras, stop.
    if len(frames) < len(captures):
        print("[INFO] Stopping due to read error.")
        break

    # Initialize result with the first frame
    # result = frames[0]
    # result = stitcher.stitch(frames, ratio=0.7, reprojThresh=5.0)

    # Calculate canvas dimensions
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
    result = None
    for idx, i in enumerate(valid_indices):
        if i >= len(frames) or idx >= len(homographies):
            continue
            
        try:
            warped = cv2.warpPerspective(frames[i], Ht.dot(homographies[idx]), 
                                        (xmax-xmin, ymax-ymin))
        except:
            print(f"Skipping invalid warp for camera {i}")
            continue

        if result is None:
            result = warped
        else:
            # Create blending mask
            mask = np.zeros_like(warped, dtype=np.uint8)
            mask[(warped[..., 0] > 0) | 
                    (warped[..., 1] > 0) | 
                    (warped[..., 2] > 0)] = 255
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            mask = mask.astype(float) / 255.0
            
            # Blend overlapping regions
            overlap = (result > 0) & (warped > 0)
            result[overlap] = (result[overlap] * (1.0 - mask[overlap]) + 
                                warped[overlap] * mask[overlap]).astype(np.uint8)
            
            # Add non-overlapping regions
            non_overlap = (result == 0) & (warped > 0)
            result[non_overlap] = warped[non_overlap]
    
    # Stitch remaining frames one by one
    """
    for i in range(1, len(frames)):
        print(f"stich {i}")
        result = stitcher.stitch([result, frames[i]], ratio=0.6, reprojThresh=2.0)
        if result is None:
            print(f"[INFO] Failed to stitch frame {i}")
            break
    """

    if result is None:
        break

    # Convert the panorama to grayscale and blur for motion detection
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Initialize or reinitialize motion detector if needed
    if motion is None:
        motion = BasicMotionDetector(minArea=500)
    elif motion.avg is not None and motion.avg.shape != gray.shape:
        motion = BasicMotionDetector(minArea=500)
    
    locs = motion.update(gray)

    # Process motion if enough frames have accumulated
    if totalFrames > 32 and len(locs) > 0:
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)
        for l in locs:
            (x, y, w, h) = cv2.boundingRect(l)
            minX, maxX = min(minX, x), max(maxX, x + w)
            minY, maxY = min(minY, y), max(maxY, y + h)
        # Draw bounding box
        cv2.rectangle(result, (minX, minY), (maxX, maxY),
                      (0, 0, 255), 3)

    # Increase total frames and annotate the result with a timestamp
    totalFrames += 1
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(result, ts, (10, result.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Display the stitched result and the individual frames
    cv2.imshow("Result", result)
    for i, frame in enumerate(frames):
        cv2.imshow(f"Camera {i}", frame)

    # Exit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("[INFO] Cleaning up...")
cv2.destroyAllWindows()
for cap in captures:
    cap.release()
