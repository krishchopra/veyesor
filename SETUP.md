# Veyesor Setup Guide

## 🎯 Overview
Veyesor is a live panoramic video streaming system that captures video from multiple cameras, stitches them into a panoramic view, and streams the result via WebSocket to a web interface.

## 🏗️ Architecture
- **Python Backend** (`panoramacv/`): Computer vision processing with OpenCV for panoramic stitching
- **Next.js Frontend** (`webapp/`): Modern web interface for viewing the panoramic stream
- **Unity Component** (`unity/`): 3D visualization (optional)

## 🚀 Quick Start

### Prerequisites
- Python 3.13+ with venv support
- Node.js 18+ with npm
- Linux/macOS environment

### 1. Install Dependencies

#### Python Backend
```bash
cd panoramacv
python3 -m venv venv
source venv/bin/activate
pip install opencv-python imutils websockets numpy
```

#### Next.js Frontend
```bash
cd webapp
npm install
```

### 2. Start the System
From the project root:
```bash
./start_veyesor.sh
```

Or manually:
```bash
# Terminal 1: Start Python WebSocket server
cd panoramacv
source venv/bin/activate
python3 test_server.py

# Terminal 2: Start Next.js frontend
cd webapp
npm run dev
```

### 3. Access the Application
- **Web Interface**: http://localhost:3000
- **WebSocket Server**: ws://localhost:8765

## 🛠️ Configuration

### Camera Setup
For real cameras, edit `panoramacv/video_stream_server.py`:
```python
camera_urls = [
    "http://your-camera-1-ip:port/video",
    "http://your-camera-2-ip:port/video",
    "http://your-camera-3-ip:port/video",
]
```

### Mock Cameras (Default)
The system currently uses mock cameras with animated content for demonstration. This is configured in `test_server.py`.

## 🧪 Testing

### Test Panorama Stitching
```bash
cd panoramacv
source venv/bin/activate
python3 test_stitching.py
```

### Test WebSocket Connection
```bash
cd panoramacv
source venv/bin/activate
python3 test_websocket_client.py
```

## 📁 Project Structure
```
veyesor/
├── panoramacv/                 # Python computer vision backend
│   ├── pyimagesearch/         # Custom CV modules
│   │   ├── panorama.py        # Panoramic stitching algorithm
│   │   └── basicmotiondetector.py  # Motion detection
│   ├── video_stream_server.py # Main WebSocket server
│   ├── test_server.py         # Demo server with mock cameras
│   ├── test_stitching.py      # Stitching functionality test
│   └── requirements.txt       # Python dependencies
├── webapp/                     # Next.js frontend
│   ├── app/
│   │   ├── components/
│   │   │   └── VideoStream.tsx # WebSocket video component
│   │   └── page.tsx           # Main page
│   └── package.json           # Node.js dependencies
├── unity/                      # Unity 3D components
└── start_veyesor.sh           # Startup script
```

## 🔧 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Ensure Python server is running on port 8765
   - Check firewall settings
   - Verify WebSocket URL in `VideoStream.tsx`

2. **Panorama Stitching Failed**
   - Check camera feeds are accessible
   - Ensure sufficient overlap between camera views
   - Adjust stitching parameters in `panorama.py`

3. **Dependencies Missing**
   - Install system packages: `sudo apt update && sudo apt install python3-dev python3-pip`
   - Recreate virtual environment if needed

### Logs
- Python server logs appear in the terminal
- Browser console shows WebSocket connection status
- Check Network tab in browser dev tools for WebSocket messages

## 🎨 Features
- ✅ Real-time panoramic video stitching
- ✅ WebSocket streaming to web interface
- ✅ Motion detection with bounding boxes
- ✅ Responsive web UI
- ✅ Mock camera support for testing
- ✅ Configurable camera sources

## 🚧 Development

### Adding New Features
1. Backend changes go in `panoramacv/`
2. Frontend changes go in `webapp/app/`
3. Test changes with the provided test scripts

### Production Deployment
1. Update `ENVIRONMENT` variable to "production"
2. Configure proper camera URLs
3. Set up proper HTTPS/WSS endpoints
4. Update CORS settings in both servers