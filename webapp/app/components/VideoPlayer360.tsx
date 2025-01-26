"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import type ReactPlayer from "react-player";

const ReactPlayerComponent = dynamic(() => import("react-player"), {
	ssr: false,
});

interface VideoPlayer360Props {
	wsUrl: string;
}

export default function VideoPlayer360({ wsUrl }: VideoPlayer360Props) {
	const playerRef = useRef<ReactPlayer>(null);
	const [streamUrl, setStreamUrl] = useState<string>("");

	useEffect(() => {
		// Create WebSocket connection
		const ws = new WebSocket(wsUrl);

		ws.onopen = () => {
			console.log("Connected to video stream");
		};

		ws.onmessage = (event) => {
			// Assuming the server sends the video stream URL
			const data = JSON.parse(event.data);
			if (data.streamUrl) {
				setStreamUrl(data.streamUrl);
			}
		};

		ws.onerror = (error) => {
			console.error("WebSocket error:", error);
		};

		return () => {
			ws.close();
		};
	}, [wsUrl]);

	if (!streamUrl) {
		return <div>Loading stream...</div>;
	}

	return (
		<div className="relative w-full h-screen">
			<ReactPlayerComponent
				ref={playerRef}
				url={streamUrl}
				width="100%"
				height="100%"
				playing={true}
				controls={true}
				config={{
					file: {
						attributes: {
							crossOrigin: "anonymous",
							playsInline: true,
						},
					},
				}}
			/>
		</div>
	);
}
