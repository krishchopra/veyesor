import asyncio
import websockets
import json
import cv2
import numpy as np
import base64
import logging
import sys
import os
from pyimagesearch.panorama import Stitcher

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_ORIGINS = {
    "development": {"http://localhost:3000", "ws://localhost:3000"},
    "production": {"https://veyesor.vercel.app", "wss://veyesor.vercel.app"},
}

camera_urls = [
    "http://10.217.5.80:4747/video",  # Benson
    "http://10.217.22.138:4747/video",  # Krish
    "http://10.217.29.154:4747/video",  # Fahmi
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("websocket_server")


def resize_frame(frame, target_width=400):
    """Safely resize a frame maintaining aspect ratio"""
    if frame is None:
        return None
    height, width = frame.shape[:2]
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    return cv2.resize(
        frame, (target_width, target_height), interpolation=cv2.INTER_AREA
    )


async def stream_video(websocket):
    logger.info(f"New client connected from {websocket.remote_address}")

    try:
        # Initialize video captures
        captures = []
        for i, url in enumerate(camera_urls):
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                logger.error(f"Failed to open camera {i}")
                for c in captures:
                    c.release()
                await websocket.send(
                    json.dumps(
                        {"type": "error", "message": f"Failed to connect to camera {i}"}
                    )
                )
                return
            captures.append(cap)

        await websocket.send(json.dumps({"type": "connection", "status": "accepted"}))
        logger.info("Connected to all cameras")

        stitcher = Stitcher()
        homographies = None
        valid_indices = None

        while True:
            # Read frames from all cameras
            frames = []
            for i, cap in enumerate(captures):
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Failed to read from camera {i}")
                    continue

                # Resize frame to reduce processing time
                frame = resize_frame(frame, target_width=400)
                if frame is not None:
                    frames.append(frame)

            if len(frames) < len(captures):
                continue

            # Compute homographies for first frame or if stitching fails
            if homographies is None or valid_indices is None:
                homographies, valid_indices = stitcher.stitch(
                    frames, ratio=0.7, reprojThresh=5.0
                )
                if not valid_indices:
                    logger.error("Failed to compute initial homographies")
                    continue
                logger.info("Initial stitching completed")

            # Stitch frames using pre-computed homographies
            result = None
            corners = []

            # Calculate corners for warping
            for i, H in zip(valid_indices, homographies):
                if i >= len(frames):
                    continue
                h, w = frames[i].shape[:2]
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)
                warped = cv2.perspectiveTransform(pts, H)
                corners.append(warped)

            if not corners:
                continue

            # Calculate output dimensions
            all_corners = np.concatenate(corners)
            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            t = [-xmin, -ymin]
            Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

            # Warp and blend frames
            for idx, i in enumerate(valid_indices):
                if i >= len(frames) or idx >= len(homographies):
                    continue

                warped = cv2.warpPerspective(
                    frames[i], Ht.dot(homographies[idx]), (xmax - xmin, ymax - ymin)
                )

                if result is None:
                    result = warped
                else:
                    # Create and blur mask for smooth blending
                    mask = np.zeros_like(warped, dtype=np.uint8)
                    mask[
                        (warped[..., 0] > 0)
                        | (warped[..., 1] > 0)
                        | (warped[..., 2] > 0)
                    ] = 255
                    mask = cv2.GaussianBlur(mask, (21, 21), 0)
                    mask = mask.astype(float) / 255.0

                    # Blend overlapping regions
                    overlap = (result > 0) & (warped > 0)
                    result[overlap] = (
                        result[overlap] * (1.0 - mask[overlap])
                        + warped[overlap] * mask[overlap]
                    ).astype(np.uint8)

                    # Copy non-overlapping regions
                    non_overlap = (result == 0) & (warped > 0)
                    result[non_overlap] = warped[non_overlap]

            if result is not None:
                # Resize final result to reduce bandwidth
                result_h, result_w = result.shape[:2]
                target_w = 800
                target_h = int((target_w / result_w) * result_h)
                result = cv2.resize(
                    result, (target_w, target_h), interpolation=cv2.INTER_AREA
                )

                # Encode and send frame
                _, buffer = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpg_as_text = base64.b64encode(buffer).decode("utf-8")

                try:
                    await websocket.send(
                        json.dumps({"type": "frame", "data": jpg_as_text})
                    )
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Client disconnected")
                    break

            # Control frame rate
            await asyncio.sleep(0.1)  # 10 FPS

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Stream error: {str(e)}")
        try:
            await websocket.send(
                json.dumps({"type": "error", "message": f"Stream error: {str(e)}"})
            )
        except:
            pass
    finally:
        for cap in captures:
            cap.release()
        logger.info("Resources released")


async def main():
    host = "0.0.0.0"  # Changed from 127.0.0.1 to allow external connections
    port = 8765

    async def origin_check(origin, websocket):
        allowed = ALLOWED_ORIGINS[ENVIRONMENT]
        if origin not in allowed:
            logger.warning(f"Rejected connection from origin: {origin}")
            return False
        return True

    try:
        async with websockets.serve(
            stream_video,
            host,
            port,
            ping_interval=None,
            ping_timeout=None,
            origins=ALLOWED_ORIGINS[ENVIRONMENT],
        ) as server:
            logger.info(f"WebSocket server is running on ws://{host}:{port}")
            logger.info(f"Environment: {ENVIRONMENT}")
            logger.info(f"Allowed origins: {ALLOWED_ORIGINS[ENVIRONMENT]}")
            await asyncio.Future()  # run forever

    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServer shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
