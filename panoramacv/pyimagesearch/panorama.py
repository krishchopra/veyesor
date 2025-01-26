import numpy as np
import cv2
import imutils

import numpy as np
import cv2
import imutils

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3()
        self.cached_homographies = {}
        self.stable_frames = 0
        self.last_valid_result = None
        self.reference_idx = 1  # Use middle camera as reference for 4-cam setup

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        if len(images) < 2:
            return images[0] if images else None

        # Use cached results if stable
        if self.stable_frames > 30 and self.last_valid_result is not None:
            return self.last_valid_result.copy()

        # Calculate homographies relative to reference camera
        homographies = []
        valid_indices = [self.reference_idx]
        ref_image = images[self.reference_idx]
        (kps_ref, features_ref) = self.detectAndDescribe(ref_image)

        # Calculate transformations to reference camera
        for i, image in enumerate(images):
            if i == self.reference_idx:
                homographies.append(np.eye(3))
                continue
                
            (kps, features) = self.detectAndDescribe(image)
            M = self.matchKeypoints(kps_ref, kps, features_ref, features, ratio, reprojThresh)
            
            if M is None:
                print(f"Warning: Couldn't stitch camera {i} to reference")
                continue
                
            (_, H, _) = M
            homographies.append(H)
            valid_indices.append(i)

        if len(valid_indices) < 2:
            return None

        return (homographies, valid_indices)

    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = matcher.knnMatch(featuresB, featuresA, k=2)
        
        matches = []
        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[i] for (_, i) in matches])
            H, status = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 
                                          reprojThresh, maxIters=2000, confidence=0.95)
            return (matches, H, status)
        return None
