#!/bin/bash

echo "🚀 Starting Veyesor - Live Panoramic Video Streaming System"
echo "============================================================"

# Function to cleanup background processes
cleanup() {
    echo "🔄 Shutting down services..."
    pkill -f "python3 test_server.py" 2>/dev/null
    pkill -f "next dev" 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start Python WebSocket server
echo "📡 Starting Python WebSocket server..."
cd panoramacv
source venv/bin/activate
python3 test_server.py &
PYTHON_PID=$!
cd ..

# Wait a moment for the Python server to start
sleep 3

# Start Next.js development server
echo "🌐 Starting Next.js frontend..."
cd webapp
npm run dev &
NEXTJS_PID=$!
cd ..

echo ""
echo "✅ Veyesor is now running!"
echo "📡 WebSocket Server: ws://localhost:8765"
echo "🌐 Web Interface: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for background processes
wait $PYTHON_PID $NEXTJS_PID