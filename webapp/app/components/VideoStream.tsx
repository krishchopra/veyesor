'use client';

import React, { useEffect, useRef, useState } from 'react';

const WEBSOCKET_URL = 'ws://127.0.0.1:8765';

const VideoStream: React.FC = () => {
    const imageRef = useRef<HTMLImageElement>(null);
    const [error, setError] = useState<string | null>(null);
    const [isConnecting, setIsConnecting] = useState(true);
    const wsRef = useRef<WebSocket | null>(null);
    
    useEffect(() => {
        const ws = new WebSocket(WEBSOCKET_URL);
        wsRef.current = ws;
        
        ws.onopen = () => {
            console.log('WebSocket connection opened');
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'connection' && data.status === 'accepted') {
                    console.log('Connection accepted by server');
                    setIsConnecting(false);
                    setError(null);
                } else if (data.type === 'error') {
                    console.error('Server error:', data.message);
                    setError(data.message);
                    setIsConnecting(false);
                } else if (data.type === 'frame' && imageRef.current) {
                    imageRef.current.src = `data:image/jpeg;base64,${data.data}`;
                }
            } catch (e) {
                console.error('Error processing message:', e);
            }
        };
        
        ws.onerror = () => {
            setError('Failed to connect to video stream');
            setIsConnecting(false);
        };
        
        ws.onclose = () => {
            setError('Connection to video stream closed');
            setIsConnecting(false);
        };
        
        return () => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        };
    }, []);
    
    return (
        <div className="relative w-full aspect-video bg-gray-900 rounded-lg overflow-hidden">
            {isConnecting && (
                <div className="absolute inset-0 flex items-center justify-center text-white">
                    Connecting to video stream...
                </div>
            )}
            {error && (
                <div className="absolute inset-0 flex items-center justify-center text-red-500 px-4 text-center">
                    {error}
                </div>
            )}
            <img
                ref={imageRef}
                className="absolute inset-0 w-full h-full object-contain"
                alt="Video stream"
            />
        </div>
    );
}

export default VideoStream; 