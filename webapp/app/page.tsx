'use client';

import dynamic from 'next/dynamic';

// Use dynamic import to avoid SSR issues with WebSocket
const VideoStream = dynamic(() => import('./components/VideoStream'), {
	ssr: false
});

export default function Home() {
	return (
		<main className="flex min-h-screen flex-col items-center justify-between p-24">
			<div className="w-full max-w-5xl">
				<h1 className="text-4xl font-bold mb-8">Live Panoramic View</h1>
				<VideoStream />
			</div>
		</main>
	);
}
