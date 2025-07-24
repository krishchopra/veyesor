#!/usr/bin/env python3
"""
Simple test server to verify Flask functionality
"""
from flask import Flask, Response, jsonify
import cv2
import numpy as np
import time
import json

app = Flask(__name__)

def generate_test_frame():
    """Generate a test frame with timestamp"""
    # Create a test image with timestamp
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles
    cv2.rectangle(img, (50, 50), (200, 150), (255, 0, 0), -1)
    cv2.rectangle(img, (250, 100), (400, 200), (0, 255, 0), -1)
    cv2.rectangle(img, (450, 150), (600, 250), (0, 0, 255), -1)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Test Frame", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def generate_frames():
    """Generate video frames for streaming"""
    while True:
        frame = generate_test_frame()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.1)  # 10 FPS

@app.route('/')
def index():
    """Main page"""
    return """
    <html>
    <head>
        <title>Test Video Stream</title>
    </head>
    <body>
        <h1>Test Video Stream</h1>
        <img src="/video_feed" width="640" height="480">
        <br><br>
        <a href="/status">Check Status</a>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Status endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Test server is working!',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    print("Starting test server...")
    print("Visit http://localhost:5000 to see the test stream")
    print("Visit http://localhost:5000/status for status check")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\nServer stopped!")