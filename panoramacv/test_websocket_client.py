import asyncio
import websockets
import json
import base64
import cv2
import numpy as np

async def test_websocket():
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server")
            
            frame_count = 0
            while frame_count < 10:  # Receive 10 frames for testing
                message = await websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "connection":
                    print(f"📡 Connection status: {data['status']}")
                elif data["type"] == "frame":
                    frame_count += 1
                    print(f"🖼️  Received frame {frame_count}")
                    
                    # Decode and save a test frame
                    if frame_count == 5:  # Save the 5th frame as a test
                        img_data = base64.b64decode(data["data"])
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        cv2.imwrite("test_received_frame.jpg", img)
                        print("💾 Saved test frame as 'test_received_frame.jpg'")
                        
                elif data["type"] == "error":
                    print(f"❌ Server error: {data['message']}")
                    break
                    
            print("✅ WebSocket test completed successfully!")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())