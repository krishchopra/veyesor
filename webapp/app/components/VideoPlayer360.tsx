"use client";

import dynamic from "next/dynamic";
import { useRef } from "react";
import type ReactPlayer from "react-player";

const ReactPlayerComponent = dynamic(() => import("react-player"), {
	ssr: false,
});

export default function VideoPlayer360() {
	const playerRef = useRef<ReactPlayer>(null);

	return (
		<div className="relative w-full h-screen">
			<ReactPlayerComponent
				ref={playerRef}
				url="/sample360.mp4"
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
